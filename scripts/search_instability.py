#!/usr/bin/env python3
"""
Search Instability Detector for Nexus Infinite.

Identifies positions where search results oscillate wildly with depth.
These positions are extremely valuable for training because they indicate:
  - Tactical oversights (engine misses moves)
  - Evaluation horizon issues
  - Positions near phase boundaries
  - Material imbalance complexities

Usage:
    python scripts/search_instability.py --engine ./nexus --positions file.fen \
        --min-depth 8 --max-depth 20 --output unstable.fen
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional


@dataclass
class SearchResult:
    """Search result at a specific depth."""
    depth: int
    score: int          # Centipawns
    best_move: str      # UCI move
    nodes: int
    time_ms: int
    pv: List[str] = field(default_factory=list)
    
    @property
    def is_mate(self) -> bool:
        return abs(self.score) > 30000  # Mate score threshold
    
    @property
    def mate_distance(self) -> Optional[int]:
        if self.is_mate:
            return (32000 - abs(self.score)) // 2
        return None


@dataclass 
class InstabilityReport:
    """Full analysis of search instability for one position."""
    fen: str
    results: List[SearchResult] = field(default_factory=list)
    
    # Instability metrics
    eval_range: int = 0           # Max - Min score across depths
    eval_variance: float = 0.0    # Variance of scores
    flip_count: int = 0           # Number of sign flips (winning -> losing)
    best_move_changes: int = 0      # How many times best move changed
    mate_appears: bool = False    # Mate found at deeper depth
    mate_disappears: bool = False # Mate lost at deeper depth
    trend_inconsistent: bool = False # Score doesn't stabilize
    
    # Classification
    instability_type: str = "unknown"
    severity: float = 0.0         # 0-10 scale
    
    # Training value
    training_priority: float = 0.0  # Higher = more valuable to train on
    recommended_depth: int = 0      # Depth where search seems most reliable


class SearchInstabilityDetector:
    """Detects unstable search behavior across depths."""
    
    # Thresholds
    SIGNIFICANT_EVAL_CHANGE = 100   # cp: score change this much = unstable
    MATE_SCORE = 32000
    FLIP_THRESHOLD = 50             # cp: crossing this boundary = flip
    STABILIZATION_DEPTH = 3         # Need this many depths without change = stable
    
    def __init__(self, engine_path: str, threads: int = 1, hash_mb: int = 64):
        self.engine_path = engine_path
        self.threads = threads
        self.hash_mb = hash_mb
    
    def search_position(self, fen: str, depth: int) -> SearchResult:
        """Run engine search at fixed depth."""
        proc = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send UCI commands
        commands = [
            "uci",
            f"setoption name Hash value {self.hash_mb}",
            f"setoption name Threads value {self.threads}",
            "isready",
            f"position fen {fen}",
            f"go depth {depth}",
        ]
        
        stdout, stderr = proc.communicate("\n".join(commands) + "\n", 
                                          timeout=depth * 30)
        
        # Parse output
        result = SearchResult(depth=depth, score=0, best_move="", nodes=0, time_ms=0)
        
        for line in stdout.split("\n"):
            if line.startswith("info"):
                parts = line.split()
                
                # Parse score
                if "score" in parts:
                    score_idx = parts.index("score")
                    if score_idx + 2 < len(parts):
                        if parts[score_idx + 1] == "cp":
                            result.score = int(parts[score_idx + 2])
                        elif parts[score_idx + 1] == "mate":
                            mate_in = int(parts[score_idx + 2])
                            result.score = self.MATE_SCORE - abs(mate_in) * 2
                            if mate_in < 0:
                                result.score = -result.score
                
                # Parse nodes
                if "nodes" in parts:
                    nodes_idx = parts.index("nodes")
                    if nodes_idx + 1 < len(parts):
                        result.nodes = int(parts[nodes_idx + 1])
                
                # Parse time
                if "time" in parts:
                    time_idx = parts.index("time")
                    if time_idx + 1 < len(parts):
                        result.time_ms = int(parts[time_idx + 1])
                
                # Parse PV
                if "pv" in parts:
                    pv_idx = parts.index("pv") + 1
                    result.pv = parts[pv_idx:pv_idx + 5]  # First 5 moves
            
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    result.best_move = parts[1]
        
        return result
    
    def analyze_position(self, fen: str, 
                         min_depth: int = 8,
                         max_depth: int = 20) -> InstabilityReport:
        """Analyze position across depth range."""
        report = InstabilityReport(fen=fen)
        
        print(f"\nAnalyzing: {fen[:60]}...")
        
        prev_score = None
        prev_move = None
        
        for depth in range(min_depth, max_depth + 1):
            print(f"  Depth {depth}...", end=" ", flush=True)
            
            try:
                result = self.search_position(fen, depth)
                report.results.append(result)
                
                print(f"score={result.score} move={result.best_move} "
                      f"nodes={result.nodes:,}")
                
                # Detect best move change
                if prev_move and result.best_move != prev_move:
                    report.best_move_changes += 1
                prev_move = result.best_move
                
                # Detect mate appears/disappears
                if prev_score and not self._is_mate(prev_score) and self._is_mate(result.score):
                    report.mate_appears = True
                if prev_score and self._is_mate(prev_score) and not self._is_mate(result.score):
                    report.mate_disappears = True
                
                # Detect sign flip
                if prev_score and self._is_significant_flip(prev_score, result.score):
                    report.flip_count += 1
                
                prev_score = result.score
                
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT at depth {depth}")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                break
        
        # Compute metrics
        if len(report.results) >= 2:
            scores = [r.score for r in report.results]
            report.eval_range = max(scores) - min(scores)
            
            # Variance (excluding mate scores for stability)
            cp_scores = [s for s in scores if abs(s) < 30000]
            if len(cp_scores) >= 2:
                mean = sum(cp_scores) / len(cp_scores)
                report.eval_variance = sum((s - mean) ** 2 for s in cp_scores) / len(cp_scores)
            
            # Trend inconsistency: does score stabilize?
            # Check last STABILIZATION_DEPTH consecutive depths
            if len(scores) >= self.STABILIZATION_DEPTH + 1:
                recent_changes = [
                    abs(scores[i] - scores[i-1]) 
                    for i in range(-self.STABILIZATION_DEPTH, 0)
                ]
                report.trend_inconsistent = any(
                    c > self.SIGNIFICANT_EVAL_CHANGE for c in recent_changes
                )
            
            # Classify instability type
            report.instability_type = self._classify(report)
            report.severity = self._compute_severity(report)
            report.training_priority = self._training_priority(report)
            
            # Recommended depth: where score first stabilizes
            report.recommended_depth = self._find_stable_depth(report)
        
        return report
    
    def _is_mate(self, score: int) -> bool:
        return abs(score) > 30000
    
    def _is_significant_flip(self, old: int, new: int) -> bool:
        """Detect if evaluation crossed a critical boundary."""
        # Significant if: both non-mate, opposite signs, and magnitude > threshold
        if self._is_mate(old) or self._is_mate(new):
            return False
        
        if (old > self.FLIP_THRESHOLD and new < -self.FLIP_THRESHOLD) or \
           (old < -self.FLIP_THRESHOLD and new > self.FLIP_THRESHOLD):
            return True
        
        return False
    
    def _classify(self, report: InstabilityReport) -> str:
        """Classify type of instability."""
        if report.mate_appears or report.mate_disappears:
            if report.mate_appears and not report.mate_disappears:
                return "missed_mate"  # Engine found mate late
            elif report.mate_disappears:
                return "phantom_mate"  # Engine thought mate, was wrong
            return "mate_instability"
        
        if report.flip_count >= 2:
            return "eval_flips"  # Multiple winning/losing reversals
        
        if report.best_move_changes >= 3:
            return "move_changes"  # Can't decide best move
        
        if report.trend_inconsistent:
            return "unstable_trend"  # Score keeps changing even at high depth
        
        if report.eval_range > 300:
            return "large_range"  # Big score swing, but consistent direction
        
        return "minor_instability"
    
    def _compute_severity(self, report: InstabilityReport) -> float:
        """Compute severity score 0-10."""
        severity = 0.0
        
        # Mate issues are most severe
        if report.mate_appears:
            severity += 3.0
        if report.mate_disappears:
            severity += 5.0  # Phantom mate is worse
        
        # Eval flips
        severity += report.flip_count * 1.5
        
        # Best move changes
        severity += report.best_move_changes * 0.5
        
        # Range
        severity += min(report.eval_range / 200.0, 3.0)
        
        # Variance
        severity += min(report.eval_variance / 5000.0, 2.0)
        
        # Trend
        if report.trend_inconsistent:
            severity += 1.0
        
        return min(severity, 10.0)
    
    def _training_priority(self, report: InstabilityReport) -> float:
        """How valuable is this position for training?"""
        # Higher severity = more valuable
        priority = report.severity
        
        # But also: simpler positions with mate issues are very valuable
        if report.instability_type in ("missed_mate", "phantom_mate"):
            priority *= 1.5
        
        # Eval flips at high depth are extremely valuable
        if report.flip_count > 0 and len(report.results) > 15:
            priority *= 1.3
        
        return min(priority, 10.0)
    
    def _find_stable_depth(self, report: InstabilityReport) -> int:
        """Find depth where evaluation stabilizes."""
        scores = [r.score for r in report.results]
        
        for i in range(len(scores) - self.STABILIZATION_DEPTH + 1):
            window = scores[i:i + self.STABILIZATION_DEPTH]
            if all(abs(window[j] - window[j+1]) < self.SIGNIFICANT_EVAL_CHANGE 
                   for j in range(len(window)-1)):
                return report.results[i].depth
        
        return report.results[-1].depth if report.results else 0
    
    def find_unstable_positions(self, position_file: str, 
                                 min_depth: int = 8,
                                 max_depth: int = 20,
                                 min_severity: float = 5.0,
                                 max_positions: int = 100) -> List[InstabilityReport]:
        """Find unstable positions from file."""
        reports = []
        
        with open(position_file, 'r') as f:
            for i, line in enumerate(f):
                if max_positions > 0 and len(reports) >= max_positions:
                    break
                
                fen = line.strip()
                if not fen or fen.startswith('#'):
                    continue
                
                report = self.analyze_position(fen, min_depth, max_depth)
                
                if report.severity >= min_severity:
                    reports.append(report)
                    print(f"  *** UNSTABLE (severity={report.severity:.1f}, "
                          f"type={report.instability_type}) ***")
        
        # Sort by training priority
        reports.sort(key=lambda r: r.training_priority, reverse=True)
        
        return reports
    
    def export_training_data(self, reports: List[InstabilityReport],
                           output_file: str,
                           score_override: Optional[int] = None):
        """Export unstable positions for priority training.
        
        score_override: if provided, use this score instead of engine's
        (useful for positions with known correct evaluation)
        """
        with open(output_file, 'w') as f:
            for report in reports:
                # Use score at recommended depth (most reliable)
                final_score = score_override
                if final_score is None:
                    final_result = next(
                        (r for r in report.results 
                         if r.depth == report.recommended_depth),
                        report.results[-1] if report.results else None
                    )
                    if final_result:
                        final_score = final_result.score
                    else:
                        final_score = 0
                
                # Weight by training priority (repeat position multiple times)
                weight = max(1, int(report.training_priority / 2))
                
                for _ in range(weight):
                    f.write(f"{report.fen} | {final_score} | 0\n")
        
        print(f"\nExported {len(reports)} positions to {output_file}")
        print(f"Total training samples (with weighting): "
              f"{sum(int(r.training_priority / 2) for r in reports)}")
    
    def export_json_report(self, reports: List[InstabilityReport],
                          output_file: str):
        """Export detailed JSON report for analysis."""
        data = {
            'positions': [
                {
                    'fen': r.fen,
                    'instability_type': r.instability_type,
                    'severity': r.severity,
                    'training_priority': r.training_priority,
                    'eval_range': r.eval_range,
                    'eval_variance': r.eval_variance,
                    'flip_count': r.flip_count,
                    'best_move_changes': r.best_move_changes,
                    'mate_appears': r.mate_appears,
                    'mate_disappears': r.mate_disappears,
                    'trend_inconsistent': r.trend_inconsistent,
                    'recommended_depth': r.recommended_depth,
                    'depths_searched': len(r.results),
                    'scores_by_depth': [
                        {'depth': res.depth, 'score': res.score,
                         'best_move': res.best_move, 'nodes': res.nodes}
                        for res in r.results
                    ]
                }
                for r in reports
            ],
            'summary': {
                'total_positions': len(reports),
                'avg_severity': sum(r.severity for r in reports) / len(reports) if reports else 0,
                'avg_priority': sum(r.training_priority for r in reports) / len(reports) if reports else 0,
                'type_distribution': self._type_distribution(reports),
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"JSON report saved: {output_file}")
    
    def _type_distribution(self, reports: List[InstabilityReport]) -> Dict[str, int]:
        dist = {}
        for r in reports:
            dist[r.instability_type] = dist.get(r.instability_type, 0) + 1
        return dist


def main():
    parser = argparse.ArgumentParser(description='Search Instability Detector')
    parser.add_argument('--engine', required=True, help='Engine binary path')
    parser.add_argument('--positions', required=True, help='FEN file')
    parser.add_argument('--min-depth', type=int, default=8)
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--min-severity', type=float, default=5.0)
    parser.add_argument('--max-positions', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--hash', type=int, default=64)
    parser.add_argument('--output-training', help='Output training data file')
    parser.add_argument('--output-json', help='Output JSON report')
    
    args = parser.parse_args()
    
    detector = SearchInstabilityDetector(
        args.engine, args.threads, args.hash
    )
    
    reports = detector.find_unstable_positions(
        args.positions,
        args.min_depth,
        args.max_depth,
        args.min_severity,
        args.max_positions
    )
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Found {len(reports)} unstable positions")
    print(f"{'='*60}")
    
    if reports:
        avg_severity = sum(r.severity for r in reports) / len(reports)
        avg_priority = sum(r.training_priority for r in reports) / len(reports)
        print(f"Average severity: {avg_severity:.1f}/10")
        print(f"Average training priority: {avg_priority:.1f}/10")
        
        types = {}
        for r in reports:
            types[r.instability_type] = types.get(r.instability_type, 0) + 1
        print(f"\nInstability types:")
        for t, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")
        
        if args.output_training:
            detector.export_training_data(reports, args.output_training)
        
        if args.output_json:
            detector.export_json_report(reports, args.output_json)


if __name__ == '__main__':
    main()
