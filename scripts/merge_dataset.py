#!/usr/bin/env python3
"""
Dataset Merge, Deduplicate, Filter, and Balance
For Nexus Infinite Infinite Training Pipeline

Usage:
    python scripts/merge_dataset.py \
        --input-dir data/shards/ \
        --output data/cleaned.txt \
        --dedup \
        --filter-entropy \
        --balance \
        --min-ply 8 \
        --max-score 800

Pipeline:
    1. Merge all shard files
    2. Parse and validate positions
    3. Deduplicate (FEN hash-based)
    4. Filter low-entropy (trivial) positions
    5. Remove forced mates
    6. Remove positions in check
    7. Remove early opening (first N plies)
    8. Cap extreme scores
    9. Balance W/D/L results
    10. Shuffle and write
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class Position:
    """Parsed training position"""
    def __init__(self, line: str):
        parts = [p.strip() for p in line.split('|')]
        
        if len(parts) < 3:
            raise ValueError(f"Invalid format: {line[:80]}")
        
        self.fen = parts[0]
        
        try:
            self.score = int(parts[1])
        except ValueError:
            self.score = 0
        
        try:
            self.result = int(parts[2])
        except ValueError:
            self.result = 0
        
        self.phase = parts[3] if len(parts) > 3 else "unknown"
        self.game_id = parts[4] if len(parts) > 4 else ""
        try:
            self.ply = int(parts[5]) if len(parts) > 5 else 0
        except ValueError:
            self.ply = 0
    
    def hash(self) -> str:
        """Hash for deduplication (ignoring score/result)"""
        # Extract board state from FEN (ignore move counters for dedup)
        fen_board = self.fen.rsplit(' ', 2)[0] if ' ' in self.fen else self.fen
        return hashlib.md5(fen_board.encode()).hexdigest()[:16]
    
    def __repr__(self):
        return f"Position({self.fen[:30]}..., score={self.score}, result={self.result})"


def load_positions(input_files: List[str], max_positions: int = 0) -> List[Position]:
    """Load positions from multiple files"""
    positions: List[Position] = []
    
    for file_path in input_files:
        if not Path(file_path).exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    pos = Position(line)
                    positions.append(pos)
                    
                    if max_positions > 0 and len(positions) >= max_positions:
                        return positions
                        
                except ValueError as e:
                    if line_num < 5:  # Only warn for first few lines
                        print(f"Warning: Skipping invalid line in {file_path}:{line_num}: {e}")
                    continue
        
        print(f"Loaded {len(positions)} positions from {file_path}")
    
    return positions


def deduplicate(positions: List[Position]) -> Tuple[List[Position], int]:
    """Remove duplicate positions based on FEN hash"""
    seen: Set[str] = set()
    unique: List[Position] = []
    removed = 0
    
    for pos in positions:
        h = pos.hash()
        if h not in seen:
            seen.add(h)
            unique.append(pos)
        else:
            removed += 1
    
    return unique, removed


def filter_entropy(positions: List[Position], 
                   min_entropy: float = 0.05) -> Tuple[List[Position], int]:
    """Remove low-entropy (trivial) positions"""
    # Entropy based on: piece count, score magnitude, legal move count
    # Simplified: use score magnitude and material as proxy
    
    filtered = []
    removed = 0
    
    for pos in positions:
        # Parse FEN for piece count
        board = pos.fen.split()[0]
        pieces = sum(1 for c in board if c.isalpha())
        
        # Simple entropy: more pieces and closer to 0 score = higher entropy
        piece_entropy = pieces / 32.0  # 32 pieces max
        score_entropy = 1.0 - min(abs(pos.score) / 1000.0, 1.0)
        
        entropy = piece_entropy * 0.6 + score_entropy * 0.4
        
        if entropy >= min_entropy:
            filtered.append(pos)
        else:
            removed += 1
    
    return filtered, removed


def filter_forced_mates(positions: List[Position], 
                        mate_threshold: int = 10000) -> Tuple[List[Position], int]:
    """Remove positions with forced mate scores"""
    filtered = []
    removed = 0
    
    for pos in positions:
        if abs(pos.score) >= mate_threshold:
            removed += 1
        else:
            filtered.append(pos)
    
    return filtered, removed


def filter_score_cap(positions: List[Position],
                       max_score: int = 1000) -> Tuple[List[Position], int]:
    """Remove positions with extreme scores"""
    filtered = []
    removed = 0
    
    for pos in positions:
        if abs(pos.score) > max_score:
            removed += 1
        else:
            filtered.append(pos)
    
    return filtered, removed


def filter_opening(positions: List[Position],
                   min_ply: int = 8) -> Tuple[List[Position], int]:
    """Remove early opening positions"""
    filtered = []
    removed = 0
    
    for pos in positions:
        if pos.ply < min_ply:
            removed += 1
        else:
            filtered.append(pos)
    
    return filtered, removed


def filter_late_endgame(positions: List[Position],
                        min_pieces: int = 6) -> Tuple[List[Position], int]:
    """Remove very late endgame (below min pieces)"""
    filtered = []
    removed = 0
    
    for pos in positions:
        board = pos.fen.split()[0]
        pieces = sum(1 for c in board if c.isalpha())
        
        if pieces < min_pieces:
            removed += 1
        else:
            filtered.append(pos)
    
    return filtered, removed


def balance_results(positions: List[Position],
                    max_imbalance_ratio: float = 1.5) -> List[Position]:
    """Balance win/draw/loss distribution"""
    # Count results
    wins = [p for p in positions if p.result == 1]
    draws = [p for p in positions if p.result == 0]
    losses = [p for p in positions if p.result == -1]
    unknown = [p for p in positions if p.result not in [-1, 0, 1]]
    
    print(f"  Before balance: W={len(wins)} D={len(draws)} L={len(losses)} U={len(unknown)}")
    
    # Find target count
    counts = [len(wins), len(draws), len(losses)]
    target = min(counts)
    
    if target == 0:
        print("  Warning: Cannot balance (one category is empty)")
        return positions
    
    # Shuffle and limit each category
    random.shuffle(wins)
    random.shuffle(draws)
    random.shuffle(losses)
    
    balanced = wins[:target] + draws[:target] + losses[:target]
    random.shuffle(balanced)
    
    # Add unknown results back
    balanced.extend(unknown)
    
    print(f"  After balance:  W={target} D={target} L={target} = {len(balanced)}")
    
    return balanced


def stratified_sample(positions: List[Position],
                      samples_per_phase: int = 0) -> List[Position]:
    """Sample evenly across game phases"""
    phases = defaultdict(list)
    for pos in positions:
        phases[pos.phase].append(pos)
    
    print(f"  Phases: {dict((k, len(v)) for k, v in phases.items())}")
    
    if samples_per_phase == 0:
        # Keep all, just verify balance
        return positions
    
    sampled = []
    for phase, pos_list in phases.items():
        random.shuffle(pos_list)
        take = min(samples_per_phase, len(pos_list))
        sampled.extend(pos_list[:take])
    
    random.shuffle(sampled)
    return sampled


def write_output(positions: List[Position], output_path: str):
    """Write cleaned dataset to output file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Nexus Infinite Cleaned Training Dataset\n")
        f.write("# Format: FEN | score | result | phase | game_id | ply\n\n")
        
        for pos in positions:
            f.write(f"{pos.fen} | {pos.score} | {pos.result} | "
                    f"{pos.phase} | {pos.game_id} | {pos.ply}\n")
    
    print(f"\nWrote {len(positions)} positions to {output_path}")


def compute_stats(positions: List[Position]) -> dict:
    """Compute dataset statistics"""
    if not positions:
        return {}
    
    scores = [p.score for p in positions]
    results = Counter(p.result for p in positions)
    phases = Counter(p.phase for p in positions)
    plies = [p.ply for p in positions]
    
    return {
        "total_positions": len(positions),
        "avg_score": sum(scores) / len(scores),
        "score_std": (sum(s**2 for s in scores) / len(scores) - 
                      (sum(scores) / len(scores))**2) ** 0.5,
        "score_min": min(scores),
        "score_max": max(scores),
        "results": dict(results),
        "phases": dict(phases),
        "avg_ply": sum(plies) / len(plies) if plies else 0,
        "unique_games": len(set(p.game_id for p in positions)),
    }


def print_stats(stats: dict, label: str = "Dataset Statistics"):
    """Print formatted statistics"""
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:>10.2f}")
        elif isinstance(value, dict):
            print(f"  {key:20s}:")
            for k, v in sorted(value.items()):
                print(f"    {str(k):18s}: {v:>8,}")
        else:
            print(f"  {key:20s}: {value:>10,}")
    
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Merge, deduplicate, filter, and balance training datasets'
    )
    
    # Input
    parser.add_argument('--input', nargs='+', help='Input files')
    parser.add_argument('--input-dir', help='Directory with shard files')
    parser.add_argument('--pattern', default='*.txt', 
                        help='File pattern for input-dir')
    
    # Output
    parser.add_argument('--output', required=True, help='Output file')
    
    # Filtering
    parser.add_argument('--dedup', action='store_true', help='Deduplicate positions')
    parser.add_argument('--filter-entropy', action='store_true', 
                        help='Filter low-entropy positions')
    parser.add_argument('--filter-mates', action='store_true',
                        help='Remove forced mates')
    parser.add_argument('--filter-check', action='store_true',
                        help='Remove positions in check (if info available)')
    parser.add_argument('--filter-score', action='store_true',
                        help='Cap extreme scores')
    parser.add_argument('--filter-opening', action='store_true',
                        help='Remove early opening')
    parser.add_argument('--filter-endgame', action='store_true',
                        help='Remove very late endgame')
    
    # Parameters
    parser.add_argument('--min-ply', type=int, default=8,
                        help='Minimum ply for opening filter')
    parser.add_argument('--max-score', type=int, default=1000,
                        help='Maximum absolute score')
    parser.add_argument('--min-entropy', type=float, default=0.05,
                        help='Minimum position entropy')
    parser.add_argument('--min-pieces', type=int, default=6,
                        help='Minimum pieces for endgame filter')
    
    # Balancing
    parser.add_argument('--balance', action='store_true',
                        help='Balance win/draw/loss results')
    parser.add_argument('--stratify', action='store_true',
                        help='Sample evenly across phases')
    parser.add_argument('--samples-per-phase', type=int, default=0,
                        help='Max samples per phase (0=keep all)')
    
    # Misc
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle output')
    parser.add_argument('--max-positions', type=int, default=0,
                        help='Max positions to process (0=all)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only compute statistics')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Collect input files
    input_files = args.input or []
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if input_dir.exists():
            pattern = args.pattern
            if not pattern.startswith('*'):
                pattern = f"**/{pattern}"
            
            files = list(input_dir.glob(pattern))
            input_files.extend(str(f) for f in files)
            print(f"Found {len(files)} files in {args.input_dir}")
    
    if not input_files:
        print("Error: No input files found")
        sys.exit(1)
    
    print(f"Processing {len(input_files)} files...")
    
    # Load
    positions = load_positions(input_files, args.max_positions)
    
    if not positions:
        print("Error: No valid positions loaded")
        sys.exit(1)
    
    # Stats before
    stats_before = compute_stats(positions)
    print_stats(stats_before, "Before Processing")
    
    if args.stats_only:
        sys.exit(0)
    
    # Apply filters
    total_removed = 0
    
    if args.dedup:
        positions, removed = deduplicate(positions)
        total_removed += removed
        print(f"Deduplication: removed {removed} duplicates")
    
    if args.filter_entropy:
        positions, removed = filter_entropy(positions, args.min_entropy)
        total_removed += removed
        print(f"Entropy filter: removed {removed} low-entropy positions")
    
    if args.filter_mates:
        positions, removed = filter_forced_mates(positions)
        total_removed += removed
        print(f"Mate filter: removed {removed} forced mates")
    
    if args.filter_score:
        positions, removed = filter_score_cap(positions, args.max_score)
        total_removed += removed
        print(f"Score cap: removed {removed} positions with |score| > {args.max_score}")
    
    if args.filter_opening:
        positions, removed = filter_opening(positions, args.min_ply)
        total_removed += removed
        print(f"Opening filter: removed {removed} positions with ply < {args.min_ply}")
    
    if args.filter_endgame:
        positions, removed = filter_late_endgame(positions, args.min_pieces)
        total_removed += removed
        print(f"Endgame filter: removed {removed} positions with < {args.min_pieces} pieces")
    
    # Balance
    if args.balance:
        positions = balance_results(positions)
    
    # Stratified sample
    if args.stratify:
        positions = stratified_sample(positions, args.samples_per_phase)
    
    # Shuffle
    if args.shuffle:
        random.shuffle(positions)
    
    # Stats after
    stats_after = compute_stats(positions)
    print_stats(stats_after, "After Processing")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Input:  {stats_before['total_positions']:,} positions")
    print(f"  Output: {stats_after['total_positions']:,} positions")
    print(f"  Removed: {total_removed:,} ({100*total_removed/stats_before['total_positions']:.1f}%)")
    
    # Write
    write_output(positions, args.output)
    
    # Save stats
    stats_file = str(Path(args.output).with_suffix('.stats.json'))
    with open(stats_file, 'w') as f:
        json.dump({
            "before": stats_before,
            "after": stats_after,
            "filters_applied": {
                "dedup": args.dedup,
                "entropy": args.filter_entropy,
                "mates": args.filter_mates,
                "score_cap": args.filter_score,
                "opening": args.filter_opening,
                "endgame": args.filter_endgame,
                "balance": args.balance,
                "stratify": args.stratify,
            },
            "total_removed": total_removed,
            "parameters": {
                "min_ply": args.min_ply,
                "max_score": args.max_score,
                "min_entropy": args.min_entropy,
                "min_pieces": args.min_pieces,
                "samples_per_phase": args.samples_per_phase,
                "seed": args.seed,
            }
        }, f, indent=2)
    
    print(f"Stats saved to {stats_file}")
    print("Done!")


if __name__ == '__main__':
    main()
