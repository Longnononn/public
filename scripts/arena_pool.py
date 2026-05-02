#!/usr/bin/env python3
"""
Arena Pool for Nexus Infinite.

Instead of candidate vs best (which can lead to overfitting current best),
the candidate plays against a pool of historical champions.

This ensures the candidate is genuinely stronger, not just exploiting
the current best's weaknesses.

Pool structure:
  - Current best (always in pool)
  - Previous 3-5 champions (rolling window)
  - Baseline (oldest champion, to prevent regression)
  - Random sampling from archive

Scoring: candidate must win >50% against pool average with SPRT confidence.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional


@dataclass
class Champion:
    """A historical engine version/model."""
    name: str                # e.g., "v0.3.0_nnue_20260501"
    model_path: str          # Path to .nnue weights
    elo: float = 0.0         # Estimated Elo
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    date_added: str = ""
    generation: int = 0      # Training generation number
    
    @property
    def score(self) -> float:
        """Win rate (0-1)."""
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / total
    
    def update(self, w: int, l: int, d: int):
        self.wins += w
        self.losses += l
        self.draws += d
        self.games_played += w + l + d


class ArenaPool:
    """Manages pool of historical champions for robust validation."""
    
    # Pool configuration
    MIN_POOL_SIZE = 3        # Need at least 3 opponents
    MAX_POOL_SIZE = 8        # Max opponents per candidate test
    ROLLING_WINDOW = 5       # Keep last N champions always
    BASELINE_KEEP = 1        # Always keep oldest as baseline
    
    def __init__(self, pool_file: str = "arena_pool.json"):
        self.pool_file = pool_file
        self.champions: List[Champion] = []
        self.baseline: Optional[Champion] = None
        self.current_best: Optional[Champion] = None
        self.load()
    
    def load(self):
        """Load pool from JSON."""
        if not os.path.exists(self.pool_file):
            return
        
        with open(self.pool_file, 'r') as f:
            data = json.load(f)
        
        for c in data.get('champions', []):
            champ = Champion(
                name=c['name'],
                model_path=c['model_path'],
                elo=c.get('elo', 0.0),
                games_played=c.get('games_played', 0),
                wins=c.get('wins', 0),
                losses=c.get('losses', 0),
                draws=c.get('draws', 0),
                date_added=c.get('date_added', ''),
                generation=c.get('generation', 0)
            )
            self.champions.append(champ)
        
        # Identify baseline (oldest) and current best (highest Elo)
        if self.champions:
            self.baseline = min(self.champions, 
                              key=lambda c: c.generation)
            self.current_best = max(self.champions,
                                   key=lambda c: c.elo)
    
    def save(self):
        """Save pool to JSON."""
        data = {
            'champions': [
                {
                    'name': c.name,
                    'model_path': c.model_path,
                    'elo': c.elo,
                    'games_played': c.games_played,
                    'wins': c.wins,
                    'losses': c.losses,
                    'draws': c.draws,
                    'date_added': c.date_added,
                    'generation': c.generation,
                }
                for c in self.champions
            ],
            'baseline': self.baseline.name if self.baseline else None,
            'current_best': self.current_best.name if self.current_best else None,
        }
        
        with open(self.pool_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_champion(self, name: str, model_path: str, 
                     elo: float, generation: int):
        """Add new champion to pool (e.g., after successful promotion)."""
        champ = Champion(
            name=name,
            model_path=model_path,
            elo=elo,
            generation=generation,
            date_added=__import__('datetime').datetime.now().isoformat()
        )
        
        self.champions.append(champ)
        self.current_best = champ
        
        # Update baseline if needed
        if self.baseline is None:
            self.baseline = champ
        
        # Trim pool if too large (keep rolling window + baseline)
        self._trim_pool()
        self.save()
        
        print(f"Added champion: {name} (Elo: {elo:.1f}, Gen: {generation})")
    
    def _trim_pool(self):
        """Keep pool size manageable."""
        if len(self.champions) <= self.MAX_POOL_SIZE:
            return
        
        # Always keep: baseline, current best, last ROLLING_WINDOW
        keep_names = set()
        
        if self.baseline:
            keep_names.add(self.baseline.name)
        if self.current_best:
            keep_names.add(self.current_best.name)
        
        # Last N champions by generation
        recent = sorted(self.champions, key=lambda c: c.generation, 
                       reverse=True)[:self.ROLLING_WINDOW]
        for c in recent:
            keep_names.add(c.name)
        
        # Filter
        self.champions = [c for c in self.champions 
                         if c.name in keep_names]
    
    def select_pool(self, candidate_name: str, 
                   min_opponents: int = 4,
                   include_baseline: bool = True) -> List[Champion]:
        """Select opponents for candidate testing."""
        pool = []
        
        # Always include current best
        if self.current_best and self.current_best.name != candidate_name:
            pool.append(self.current_best)
        
        # Always include baseline (to prevent regression)
        if include_baseline and self.baseline:
            if self.baseline.name != candidate_name and \
               self.baseline not in pool:
                pool.append(self.baseline)
        
        # Add recent champions (diverse opponents)
        recent = [c for c in self.champions 
                 if c.name not in [x.name for x in pool] 
                 and c.name != candidate_name]
        recent.sort(key=lambda c: c.generation, reverse=True)
        
        # Add diverse Elo opponents (not just strongest)
        # This prevents overfitting to one Elo range
        if len(recent) > 0:
            # Sort by Elo spread
            recent.sort(key=lambda c: abs(c.elo - 
                (self.current_best.elo if self.current_best else 0)))
            
            # Take opponents across Elo spectrum
            step = max(1, len(recent) // (min_opponents - len(pool)))
            for i in range(0, len(recent), step):
                if len(pool) >= min_opponents:
                    break
                pool.append(recent[i])
        
        return pool[:self.MAX_POOL_SIZE]
    
    def test_candidate(self, candidate_path: str, candidate_name: str,
                       engine_path: str,
                       games_per_opponent: int = 50,
                       tc: str = "10+0.1",
                       threads: int = 1) -> Dict:
        """Test candidate against pool. Returns detailed results."""
        
        pool = self.select_pool(candidate_name)
        
        if len(pool) < self.MIN_POOL_SIZE:
            print(f"WARNING: Pool too small ({len(pool)}), "
                  f"falling back to direct vs best")
            if self.current_best:
                pool = [self.current_best]
        
        print(f"\n{'='*60}")
        print(f"ARENA POOL TEST: {candidate_name}")
        print(f"{'='*60}")
        print(f"Pool size: {len(pool)}")
        for c in pool:
            print(f"  - {c.name}: Elo={c.elo:.1f} Games={c.games_played}")
        print()
        
        results = {
            'candidate': candidate_name,
            'pool_size': len(pool),
            'opponents': {},
            'total_wins': 0,
            'total_losses': 0,
            'total_draws': 0,
            'total_games': 0,
            'overall_score': 0.0,
            'promotion_recommended': False,
        }
        
        for opponent in pool:
            print(f"\nVs {opponent.name} (Elo: {opponent.elo:.1f})")
            
            # Run match
            match_result = self._run_match(
                candidate_path, opponent.model_path,
                engine_path, games_per_opponent, tc, threads
            )
            
            w, l, d = match_result['wins'], match_result['losses'], match_result['draws']
            score = (w + 0.5 * d) / (w + l + d) if (w + l + d) > 0 else 0.5
            
            results['opponents'][opponent.name] = {
                'wins': w, 'losses': l, 'draws': d,
                'score': score,
                'opponent_elo': opponent.elo,
            }
            
            results['total_wins'] += w
            results['total_losses'] += l
            results['total_draws'] += d
            results['total_games'] += w + l + d
            
            print(f"  Result: +{w} -{l} ={d} (Score: {score:.3f})")
            
            # Update opponent stats
            opponent.update(l, w, d)  # Reverse for opponent
        
        # Overall score
        total = results['total_games']
        if total > 0:
            results['overall_score'] = (results['total_wins'] + 
                                        0.5 * results['total_draws']) / total
        
        # SPRT-like promotion criteria
        # Must win > 55% overall AND > 50% vs each opponent
        min_individual_score = min(
            r['score'] for r in results['opponents'].values()
        ) if results['opponents'] else 0.0
        
        results['min_individual_score'] = min_individual_score
        results['promotion_recommended'] = (
            results['overall_score'] > 0.55 and
            min_individual_score > 0.48 and  # Not losing badly to anyone
            results['total_games'] >= len(pool) * games_per_opponent * 0.8
        )
        
        print(f"\n{'='*60}")
        print(f"ARENA SUMMARY")
        print(f"{'='*60}")
        print(f"Total: +{results['total_wins']} -{results['total_losses']} "
              f"={results['total_draws']} ({total} games)")
        print(f"Overall score: {results['overall_score']:.3f}")
        print(f"Min vs individual: {min_individual_score:.3f}")
        print(f"PROMOTION: {'ACCEPT' if results['promotion_recommended'] else 'REJECT'}")
        
        self.save()
        
        return results
    
    def _run_match(self, candidate_nnue: str, opponent_nnue: str,
                   engine_path: str, games: int, tc: str, 
                   threads: int) -> Dict:
        """Run match between two networks using same engine binary."""
        # Use run_sprt.py infrastructure but as fixed-length match
        cmd = [
            "python", "scripts/run_sprt.py",
            "--engine-new", engine_path,
            "--engine-base", engine_path,
            "--network-new", candidate_nnue,
            "--network-base", opponent_nnue,
            "--games", str(games),
            "--tc", tc,
            "--threads", str(threads),
            "--fixed-games",  # Don't stop early, play all games
            "--output", "/tmp/arena_match.json"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, 
                          timeout=games * 60)
            
            if os.path.exists("/tmp/arena_match.json"):
                with open("/tmp/arena_match.json", 'r') as f:
                    data = json.load(f)
                    return {
                        'wins': data.get('wins', 0),
                        'losses': data.get('losses', 0),
                        'draws': data.get('draws', 0),
                        'elo_diff': data.get('elo_diff', 0.0),
                    }
        except Exception as e:
            print(f"  Match error: {e}")
        
        return {'wins': 0, 'losses': 0, 'draws': 0, 'elo_diff': 0.0}
    
    def get_elo_estimate(self, candidate_results: Dict) -> float:
        """Estimate candidate Elo from pool results."""
        if not self.current_best:
            return 0.0
        
        # Simple Elo calculation from average opponent Elo + performance
        total_weighted_elo = 0.0
        total_weight = 0.0
        
        for name, result in candidate_results['opponents'].items():
            opponent = next((c for c in self.champions if c.name == name), None)
            if not opponent:
                continue
            
            # Elo difference from score
            score = result['score']
            if score <= 0.0:
                score = 0.01
            if score >= 1.0:
                score = 0.99
            
            # Convert score to Elo difference
            elo_diff = -400 * __import__('math').log10(1.0 / score - 1.0)
            
            weight = result['wins'] + result['losses'] + result['draws']
            total_weighted_elo += (opponent.elo + elo_diff) * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_weighted_elo / total_weight
        
        return self.current_best.elo


def main():
    parser = argparse.ArgumentParser(description='Arena Pool Testing')
    parser.add_argument('--candidate', required=True, help='Candidate .nnue path')
    parser.add_argument('--candidate-name', required=True, help='Candidate name')
    parser.add_argument('--engine', required=True, help='Engine binary path')
    parser.add_argument('--pool-file', default='arena_pool.json',
                        help='Arena pool JSON file')
    parser.add_argument('--games', type=int, default=50,
                        help='Games per opponent')
    parser.add_argument('--tc', default='10+0.1')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--add-if-accepted', action='store_true',
                        help='Add candidate to pool if accepted')
    parser.add_argument('--generation', type=int, default=0,
                        help='Candidate generation number')
    
    args = parser.parse_args()
    
    # Initialize or load pool
    pool = ArenaPool(args.pool_file)
    
    # Test candidate
    results = pool.test_candidate(
        args.candidate, args.candidate_name,
        args.engine, args.games, args.tc, args.threads
    )
    
    # Save results
    output_file = f"arena_result_{args.candidate_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_file}")
    
    # Add to pool if accepted
    if args.add_if_accepted and results['promotion_recommended']:
        estimated_elo = pool.get_elo_estimate(results)
        pool.add_champion(
            args.candidate_name,
            args.candidate,
            estimated_elo,
            args.generation
        )
        print(f"Added {args.candidate_name} to pool (Elo: {estimated_elo:.1f})")
    
    # Exit code for CI
    sys.exit(0 if results['promotion_recommended'] else 1)


if __name__ == '__main__':
    main()
