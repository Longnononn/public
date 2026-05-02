#!/usr/bin/env python3
"""
Hard Position Mining for NNUE Training
Finds positions where Nexus fails but reference engine succeeds

Usage:
    python scripts/hard_mining.py \
        --nexus ./nexus \
        --reference ./stockfish \
        --positions positions.epd \
        --depth 20 \
        --output hard_positions.txt

Strategy:
    1. Run both engines on same positions
    2. Compare: move choice, eval difference, search consistency
    3. Queue "hard" positions for retraining
    4. Retrain with hard positions → improve weak areas
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class EngineResult:
    """Analysis result from single engine"""
    best_move: str
    score_cp: float
    depth: int
    nodes: int
    nps: int
    pv: List[str]
    time_ms: int
    multipv: List[Tuple[str, float]]  # (move, score) pairs


@dataclass
class PositionDiff:
    """Difference between Nexus and reference on same position"""
    fen: str
    nexus_move: str
    ref_move: str
    nexus_score: float
    ref_score: float
    score_diff: float  # ref - nexus
    move_agreement: bool
    hard_type: str  # "eval", "tactical", "strategic"
    
    def is_hard(self, threshold: float = 30.0) -> bool:
        """Position is hard if eval differs significantly"""
        return abs(self.score_diff) > threshold


class EngineInterface:
    """UCI engine wrapper for analysis"""
    
    def __init__(self, binary: str, name: str = "engine"):
        self.binary = binary
        self.name = name
        self.process = None
    
    def start(self):
        """Start engine process"""
        self.process = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for initialization
        self._read_until("uciok")
        self._send("isready")
        self._read_until("readyok")
    
    def stop(self):
        """Stop engine"""
        if self.process:
            self._send("quit")
            self.process.wait(timeout=5)
            self.process = None
    
    def _send(self, cmd: str):
        """Send command to engine"""
        if self.process:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
    
    def _read_until(self, target: str, timeout: float = 10.0) -> List[str]:
        """Read output until target string found"""
        lines = []
        start = time.time()
        
        while time.time() - start < timeout:
            line = self.process.stdout.readline().strip()
            if line:
                lines.append(line)
            if target in line:
                return lines
        
        return lines
    
    def analyze(self, fen: str, depth: int, movetime: int = 0,
                multipv: int = 1) -> EngineResult:
        """Analyze position to given depth"""
        
        # Set position
        self._send(f"position fen {fen}")
        
        # Search
        if movetime > 0:
            cmd = f"go movetime {movetime}"
        else:
            cmd = f"go depth {depth}"
        
        self._send(cmd)
        
        # Parse output
        best_move = ""
        score = 0.0
        nodes = 0
        nps = 0
        pv = []
        multipv_results = []
        
        start_time = time.time()
        
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                continue
            
            parts = line.split()
            
            if line.startswith("info"):
                # Parse info line
                i = 1
                while i < len(parts):
                    if parts[i] == "score" and i + 2 < len(parts):
                        score_type = parts[i + 1]
                        if score_type == "cp":
                            score = float(parts[i + 2])
                        elif score_type == "mate":
                            mate_in = int(parts[i + 2])
                            score = 10000.0 if mate_in > 0 else -10000.0
                    
                    elif parts[i] == "depth" and i + 1 < len(parts):
                        current_depth = int(parts[i + 1])
                    
                    elif parts[i] == "nodes" and i + 1 < len(parts):
                        nodes = int(parts[i + 1])
                    
                    elif parts[i] == "nps" and i + 1 < len(parts):
                        nps = int(parts[i + 1])
                    
                    elif parts[i] == "multipv" and i + 1 < len(parts):
                        mpv_idx = int(parts[i + 1])
                    
                    elif parts[i] == "pv":
                        pv_start = i + 1
                        pv = parts[pv_start:]
                    
                    i += 1
                
                # Store multipv result
                if "multipv" in line:
                    multipv_results.append((pv[0] if pv else "", score))
            
            elif line.startswith("bestmove"):
                best_move = parts[1] if len(parts) > 1 else ""
                break
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return EngineResult(
            best_move=best_move,
            score_cp=score,
            depth=current_depth if 'current_depth' in dir() else depth,
            nodes=nodes,
            nps=nps,
            pv=pv,
            time_ms=elapsed_ms,
            multipv=multipv_results
        )


class HardMiner:
    """Mine hard positions by comparing engines"""
    
    def __init__(self, nexus_binary: str, ref_binary: str,
                 nexus_depth: int = 20, ref_depth: int = 20,
                 score_threshold: float = 30.0):
        self.nexus = EngineInterface(nexus_binary, "nexus")
        self.reference = EngineInterface(ref_binary, "reference")
        self.nexus_depth = nexus_depth
        self.ref_depth = ref_depth
        self.threshold = score_threshold
        
        self.hard_positions: List[PositionDiff] = []
        self.stats = {
            'total': 0,
            'hard': 0,
            'tactical': 0,
            'strategic': 0,
            'eval_diff': 0,
            'move_agree': 0
        }
    
    def start(self):
        print("Starting engines...")
        self.nexus.start()
        self.reference.start()
        print("Engines ready.")
    
    def stop(self):
        print("Stopping engines...")
        self.nexus.stop()
        self.reference.stop()
    
    def analyze_position(self, fen: str, label: str = "") -> Optional[PositionDiff]:
        """Analyze position with both engines and compare"""
        
        print(f"\nAnalyzing: {label or fen[:60]}...")
        
        # Analyze with Nexus
        nexus_result = self.nexus.analyze(fen, self.nexus_depth)
        print(f"  Nexus: {nexus_result.best_move} ({nexus_result.score_cp:+.0f}) "
              f"d={nexus_result.depth} n={nexus_result.nodes:,}")
        
        # Analyze with reference
        ref_result = self.reference.analyze(fen, self.ref_depth)
        print(f"  Ref:   {ref_result.best_move} ({ref_result.score_cp:+.0f}) "
              f"d={ref_result.depth} n={ref_result.nodes:,}")
        
        # Calculate difference
        score_diff = ref_result.score_cp - nexus_result.score_cp
        move_agree = nexus_result.best_move == ref_result.best_move
        
        # Classify hard type
        if abs(score_diff) > 100 and not move_agree:
            hard_type = "tactical"
        elif abs(score_diff) > self.threshold:
            hard_type = "eval"
        elif not move_agree and abs(score_diff) > 20:
            hard_type = "strategic"
        else:
            hard_type = "normal"
        
        diff = PositionDiff(
            fen=fen,
            nexus_move=nexus_result.best_move,
            ref_move=ref_result.best_move,
            nexus_score=nexus_result.score_cp,
            ref_score=ref_result.score_cp,
            score_diff=score_diff,
            move_agreement=move_agree,
            hard_type=hard_type
        )
        
        # Update stats
        self.stats['total'] += 1
        self.stats['move_agree'] += 1 if move_agree else 0
        
        if diff.is_hard(self.threshold):
            self.stats['hard'] += 1
            self.stats['tactical'] += 1 if hard_type == 'tactical' else 0
            self.stats['strategic'] += 1 if hard_type == 'strategic' else 0
            self.stats['eval_diff'] += 1 if hard_type == 'eval' else 0
            
            self.hard_positions.append(diff)
            print(f"  => HARD ({hard_type}, diff={score_diff:+.0f})")
        else:
            print(f"  => OK (diff={score_diff:+.0f})")
        
        return diff
    
    def mine_from_epd(self, epd_file: str, max_positions: int = 0):
        """Mine hard positions from EPD file"""
        
        print(f"\n{'='*60}")
        print(f"Mining from: {epd_file}")
        print(f"{'='*60}")
        
        positions = []
        with open(epd_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract FEN from EPD
                    parts = line.split(';')
                    fen = parts[0].strip()
                    positions.append((fen, line))
                    
                    if max_positions > 0 and len(positions) >= max_positions:
                        break
        
        print(f"Loaded {len(positions)} positions")
        
        for i, (fen, label) in enumerate(positions):
            self.analyze_position(fen, f"#{i+1}")
            
            if (i + 1) % 10 == 0:
                self._print_progress()
        
        self._print_summary()
    
    def mine_from_pgn(self, pgn_file: str, max_games: int = 0):
        """Mine from PGN games - analyze critical positions"""
        
        print(f"\nMining from PGN: {pgn_file}")
        
        # Parse PGN and extract key positions
        # Simplified: extract positions after opening
        positions = []
        
        with open(pgn_file, 'r') as f:
            content = f.read()
        
        # Extract FENs from game headers or parse moves
        # For now, placeholder
        print("PGN mining: TODO - implement move parser")
    
    def generate_and_mine(self, num_positions: int = 1000,
                          random_depth: int = 8):
        """Generate random positions and mine for hard ones"""
        
        print(f"\nGenerating {num_positions} random positions...")
        
        # Start Nexus for self-play generation
        self.nexus._send("ucinewgame")
        
        for i in range(num_positions):
            # Generate random game to random depth
            # Placeholder: would use actual self-play
            pass
    
    def _print_progress(self):
        """Print current mining progress"""
        total = self.stats['total']
        if total == 0:
            return
        
        hard_pct = 100.0 * self.stats['hard'] / total
        agree_pct = 100.0 * self.stats['move_agree'] / total
        
        print(f"\n  Progress: {total} analyzed, "
              f"{self.stats['hard']} hard ({hard_pct:.1f}%), "
              f"{agree_pct:.1f}% move agreement")
    
    def _print_summary(self):
        """Print final mining summary"""
        print(f"\n{'='*60}")
        print("MINING SUMMARY")
        print(f"{'='*60}")
        
        s = self.stats
        total = s['total']
        
        print(f"\nTotal positions: {total}")
        print(f"Hard positions:  {s['hard']} ({100*s['hard']/total:.1f}%)")
        print(f"  - Tactical:    {s['tactical']}")
        print(f"  - Strategic:   {s['strategic']}")
        print(f"  - Eval diff:   {s['eval_diff']}")
        print(f"Move agreement:  {s['move_agree']}/{total} "
              f"({100*s['move_agree']/total:.1f}%)")
        print(f"\nQueued {len(self.hard_positions)} positions for retraining")
    
    def export_hard_positions(self, output_file: str):
        """Export hard positions for retraining"""
        
        with open(output_file, 'w') as f:
            f.write("# Hard positions for NNUE retraining\n")
            f.write("# Format: FEN | nexus_move | ref_move | "
                    "nexus_score | ref_score | diff | type\n\n")
            
            for diff in self.hard_positions:
                f.write(f"{diff.fen} | "
                        f"{diff.nexus_move} | "
                        f"{diff.ref_move} | "
                        f"{diff.nexus_score:.0f} | "
                        f"{diff.ref_score:.0f} | "
                        f"{diff.score_diff:+.0f} | "
                        f"{diff.hard_type}\n")
        
        print(f"\nExported {len(self.hard_positions)} hard positions "
              f"to {output_file}")
    
    def export_for_training(self, output_file: str,
                            blend_factor: float = 0.3):
        """Export positions in training format with reference scores"""
        
        with open(output_file, 'w') as f:
            for diff in self.hard_positions:
                # Use reference score as target (teacher forcing)
                # Add to retraining queue
                f.write(f"{diff.fen} | "
                        f"{diff.ref_score:.0f} | "
                        f"0 | "  # result unknown for single position
                        f"2 | "  # phase = middlegame
                        f"hard_mined\n")
        
        print(f"Exported {len(self.hard_positions)} positions "
              f"in training format to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Hard Position Mining for NNUE Training'
    )
    
    parser.add_argument('--nexus', required=True,
                        help='Nexus engine binary')
    parser.add_argument('--reference', required=True,
                        help='Reference engine (Stockfish/etc)')
    parser.add_argument('--positions', required=True,
                        help='EPD/PGN file with test positions')
    parser.add_argument('--depth', type=int, default=20,
                        help='Analysis depth')
    parser.add_argument('--movetime', type=int, default=0,
                        help='Analysis time in ms (0=depth-based)')
    parser.add_argument('--threshold', type=float, default=30.0,
                        help='Score difference threshold for "hard"')
    parser.add_argument('--max-positions', type=int, default=0,
                        help='Max positions to analyze (0=all)')
    parser.add_argument('--output', default='hard_positions.txt',
                        help='Output file for hard positions')
    parser.add_argument('--training-output', default='hard_for_training.txt',
                        help='Output in training format')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hard Position Mining")
    print("=" * 60)
    print(f"Nexus:     {args.nexus}")
    print(f"Reference: {args.reference}")
    print(f"Positions: {args.positions}")
    print(f"Depth:     {args.depth}")
    print(f"Threshold: {args.threshold} cp")
    print("=" * 60)
    
    # Create miner
    miner = HardMiner(
        nexus_binary=args.nexus,
        ref_binary=args.reference,
        nexus_depth=args.depth,
        ref_depth=args.depth,
        score_threshold=args.threshold
    )
    
    try:
        # Start engines
        miner.start()
        
        # Mine positions
        if args.positions.endswith('.pgn'):
            miner.mine_from_pgn(args.positions, args.max_positions)
        else:
            miner.mine_from_epd(args.positions, args.max_positions)
        
        # Export results
        miner.export_hard_positions(args.output)
        miner.export_for_training(args.training_output)
        
    except KeyboardInterrupt:
        print("\n\nMining interrupted by user")
    finally:
        # Stop engines
        miner.stop()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
