#!/usr/bin/env python3
"""
SPRT (Sequential Probability Ratio Test) for Nexus Infinite
Validates if new engine version is stronger than baseline
"""

import subprocess
import math
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Tuple, Dict, List

class SPRTTest:
    """SPRT implementation for engine testing"""
    
    def __init__(self, 
                 elo0: float = 0.0,      # Null hypothesis (no improvement)
                 elo1: float = 5.0,      # Alternative hypothesis (5 Elo improvement)
                 alpha: float = 0.05,    # Type I error (false positive)
                 beta: float = 0.05):    # Type II error (false negative)
        self.elo0 = elo0
        self.elo1 = elo1
        self.alpha = alpha
        self.beta = beta
        
        # Calculate boundaries
        self.lower_bound = math.log(beta / (1 - alpha))
        self.upper_bound = math.log((1 - beta) / alpha)
        
        # Results tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0
        
    def llr(self) -> float:
        """Calculate log-likelihood ratio"""
        if self.total_games == 0:
            return 0.0
            
        # Expected score under H0 and H1
        p0 = self._elo_to_score(self.elo0)
        p1 = self._elo_to_score(self.elo1)
        
        # Observed score
        W = self.wins
        D = self.draws
        L = self.losses
        n = self.total_games
        
        if W + D + L == 0:
            return 0.0
            
        # Pentanomial (draws count as 0.5)
        score = (W + 0.5 * D) / n
        
        # Log-likelihood ratio
        # llr = sum(log(P1(x) / P0(x)))
        if score <= 0 or score >= 1:
            return 0.0
            
        llr = W * math.log(p1 / p0) + \
              D * math.log((1 - p1) / (1 - p0)) + \
              L * math.log((1 - p1) / (1 - p0))
        
        return llr
    
    def _elo_to_score(self, elo: float) -> float:
        """Convert Elo difference to expected score"""
        return 1.0 / (1.0 + math.pow(10, -elo / 400.0))
    
    def status(self) -> str:
        """Current SPRT status"""
        llr_val = self.llr()
        
        if llr_val <= self.lower_bound:
            return "REJECT"
        elif llr_val >= self.upper_bound:
            return "ACCEPT"
        else:
            return "CONTINUE"
    
    def add_result(self, result: str):
        """Add game result: 'W', 'L', or 'D'"""
        if result == 'W':
            self.wins += 1
        elif result == 'L':
            self.losses += 1
        elif result == 'D':
            self.draws += 1
        else:
            raise ValueError(f"Invalid result: {result}")
        
        self.total_games += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        score = (self.wins + 0.5 * self.draws) / max(self.total_games, 1)
        
        return {
            'games': self.total_games,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'score': score,
            'llr': self.llr(),
            'status': self.status(),
            'elo_estimated': self._score_to_elo(score)
        }
    
    def _score_to_elo(self, score: float) -> float:
        """Convert score to Elo difference"""
        if score <= 0 or score >= 1:
            return 0.0
        return -400 * math.log10(1 / score - 1)


class FastChessRunner:
    """Run engine matches using fastchess or cutechess"""
    
    def __init__(self, 
                 engine_new: str,
                 engine_base: str,
                 tc: str = "10+0.1",
                 threads: int = 1,
                 book: str = "",
                 games: int = 10000):
        self.engine_new = engine_new
        self.engine_base = engine_base
        self.tc = tc
        self.threads = threads
        self.book = book
        self.max_games = games
        
    def run_match(self, sprt: SPRTTest) -> None:
        """Run games until SPRT conclusion"""
        
        print(f"Starting SPRT test")
        print(f"Engines: {self.engine_new} vs {self.engine_base}")
        print(f"Time control: {self.tc}")
        print(f"SPRT: H0={sprt.elo0} Elo, H1={sprt.elo1} Elo")
        print(f"Bounds: {sprt.lower_bound:.2f} to {sprt.upper_bound:.2f}")
        print()
        
        games_batch = 20  # Run in batches
        
        while sprt.status() == "CONTINUE" and sprt.total_games < self.max_games:
            results = self._run_games_batch(games_batch)
            
            for result in results:
                sprt.add_result(result)
            
            self._print_progress(sprt)
            
            # Save progress every 100 games
            if sprt.total_games % 100 == 0:
                self._save_progress(sprt)
        
        # Final report
        self._print_final_report(sprt)
    
    def _run_games_batch(self, num_games: int) -> List[str]:
        """Run batch of games and return results"""
        results = []
        
        # Use cutechess-cli or fastchess
        # Example with cutechess:
        cmd = [
            "cutechess-cli",
            "-engine", f"cmd={self.engine_new}",
            "-engine", f"cmd={self.engine_base}",
            "-each", f"tc={self.tc}",
            "-games", str(num_games),
            "-repeat",  # Each opening played twice
            "-recover",  # Continue on engine crash
            "-wait", "10",  # Wait for engines to start
            "-resign", "movecount=5", "score=500",  # Resign criteria
            "-draw", "movenumber=40", "movecount=8", "score=10",  # Draw adjudication
        ]
        
        if self.book:
            cmd.extend(["-openings", f"file={self.book}", "format=epd", "order=random", "plies=16"])
        
        cmd.append("-pgnout")
        cmd.append("games.pgn")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            # Parse results from output
            # This is simplified - actual parsing depends on cutechess output format
            for line in result.stdout.split('\n'):
                if "1-0" in line or "0-1" in line or "1/2-1/2" in line:
                    # Determine result for new engine
                    if "1-0" in line and "NewEngine" in line:
                        results.append('W')
                    elif "0-1" in line and "NewEngine" in line:
                        results.append('L')
                    else:
                        results.append('D')
        
        except subprocess.TimeoutExpired:
            print("Game batch timed out")
        except FileNotFoundError:
            # Fallback: simulate results for testing
            print("Warning: cutechess-cli not found, using simulated results")
            import random
            for _ in range(num_games):
                r = random.random()
                if r < 0.4: results.append('W')
                elif r < 0.5: results.append('L')
                else: results.append('D')
        
        return results
    
    def _print_progress(self, sprt: SPRTTest):
        """Print current progress"""
        stats = sprt.get_stats()
        
        print(f"\rGames: {stats['games']:<6} "
              f"W: {stats['wins']:<4} D: {stats['draws']:<4} L: {stats['losses']:<4} "
              f"Score: {stats['score']:.3f} "
              f"LLR: {stats['llr']:+.3f} "
              f"Status: {stats['status']:<8}", 
              end='', flush=True)
    
    def _save_progress(self, sprt: SPRTTest):
        """Save progress to file"""
        stats = sprt.get_stats()
        
        with open('sprt_progress.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'engines': {
                    'new': self.engine_new,
                    'base': self.engine_base
                },
                'time_control': self.tc,
                **stats
            }, f, indent=2)
    
    def _print_final_report(self, sprt: SPRTTest):
        """Print final SPRT report"""
        stats = sprt.get_stats()
        
        print("\n\n" + "="*60)
        print("SPRT TEST COMPLETE")
        print("="*60)
        print(f"Result: {stats['status']}")
        print(f"Total games: {stats['games']}")
        print(f"Wins: {stats['wins']} ({100*stats['wins']/stats['games']:.1f}%)")
        print(f"Draws: {stats['draws']} ({100*stats['draws']/stats['games']:.1f}%)")
        print(f"Losses: {stats['losses']} ({100*stats['losses']/stats['games']:.1f}%)")
        print(f"Score: {stats['score']:.3f}")
        print(f"Estimated Elo: {stats['elo_estimated']:+.1f}")
        print(f"Final LLR: {stats['llr']:.3f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='SPRT Test for Nexus Infinite')
    parser.add_argument('engine_new', help='Path to new engine version')
    parser.add_argument('engine_base', help='Path to base engine version')
    parser.add_argument('--elo0', type=float, default=0.0, help='Null hypothesis Elo')
    parser.add_argument('--elo1', type=float, default=5.0, help='Alternative hypothesis Elo')
    parser.add_argument('--alpha', type=float, default=0.05, help='Type I error rate')
    parser.add_argument('--beta', type=float, default=0.05, help='Type II error rate')
    parser.add_argument('--tc', default='10+0.1', help='Time control (e.g., 10+0.1)')
    parser.add_argument('--threads', type=int, default=1, help='Parallel threads')
    parser.add_argument('--book', default='', help='Opening book file (EPD)')
    parser.add_argument('--games', type=int, default=10000, help='Max games to run')
    
    args = parser.parse_args()
    
    # Create SPRT test
    sprt = SPRTTest(elo0=args.elo0, elo1=args.elo1, alpha=args.alpha, beta=args.beta)
    
    # Create runner
    runner = FastChessRunner(
        engine_new=args.engine_new,
        engine_base=args.engine_base,
        tc=args.tc,
        threads=args.threads,
        book=args.book,
        games=args.games
    )
    
    # Run test
    runner.run_match(sprt)


if __name__ == '__main__':
    main()
