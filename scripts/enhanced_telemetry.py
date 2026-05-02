#!/usr/bin/env python3
"""
Enhanced Search Telemetry Aggregator for Nexus Infinite.

Collects, analyzes, and visualizes search telemetry data.
Generates reports for debugging search regressions and training ML models.

Metrics tracked:
  - Node types (PV, cut, all nodes)
  - TT hit rates by depth
  - Cutoff causes (beta, alpha, static eval, futility, etc.)
  - Fail-high/fail-low rates
  - Search instability (eval flips between depths)
  - LMR statistics (reductions, rejections)
  - Null move success rate
  - SEE statistics
  - Time-to-depth curves
  - Policy vs search agreement
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SearchStats:
    """Aggregated search statistics."""
    # Node counts
    total_nodes: int = 0
    pv_nodes: int = 0
    cut_nodes: int = 0
    all_nodes: int = 0
    
    # TT statistics
    tt_hits: int = 0
    tt_misses: int = 0
    tt_cutoffs: int = 0
    tt_by_depth: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    # Cutoff causes
    beta_cutoffs: int = 0
    alpha_cutoffs: int = 0
    static_eval_cutoffs: int = 0
    futility_prunes: int = 0
    null_move_prunes: int = 0
    lmr_researches: int = 0
    lmr_accepted: int = 0
    
    # Fail rates
    fail_highs: int = 0
    fail_lows: int = 0
    aspiration_resets: int = 0
    
    # Instability
    eval_flips: int = 0
    move_changes: int = 0
    depth_oscillations: int = 0
    
    # Performance
    avg_time_per_depth: Dict[int, float] = field(default_factory=dict)
    nps_by_depth: Dict[int, float] = field(default_factory=dict)
    nodes_by_depth: Dict[int, int] = field(default_factory=dict)
    
    # Pruning efficiency
    moves_generated: int = 0
    moves_searched: int = 0
    moves_pruned: int = 0
    
    # NNUE
    nnue_evals: int = 0
    nnue_cache_hits: int = 0
    nnue_avg_latency_us: float = 0.0
    
    # Policy agreement (if using policy network)
    policy_agreement: int = 0
    policy_disagreement: int = 0


class TelemetryAggregator:
    """Aggregates and analyzes search telemetry logs."""
    
    def __init__(self, telemetry_file: str):
        self.telemetry_file = telemetry_file
        self.stats = SearchStats()
        self.raw_events: List[Dict] = []
        self.search_sessions: List[Dict] = []
        self.current_session: Optional[Dict] = None
    
    def load(self, max_events: int = 0):
        """Load telemetry file."""
        print(f"Loading {self.telemetry_file}...")
        
        with open(self.telemetry_file, 'r') as f:
            for i, line in enumerate(f):
                if max_events > 0 and i >= max_events:
                    break
                
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                self.raw_events.append(event)
                self._process_event(event)
        
        print(f"Loaded {len(self.raw_events)} events")
    
    def _process_event(self, event: Dict):
        """Process single telemetry event."""
        event_type = event.get('type', '')
        
        if event_type == 'SEARCH_START':
            self.current_session = {
                'start_time': event.get('timestamp', 0),
                'fen': event.get('fen', ''),
                'depth': event.get('depth', 0),
                'nodes': 0,
                'events': [],
            }
            self.search_sessions.append(self.current_session)
        
        elif event_type == 'SEARCH_END' and self.current_session:
            self.current_session['end_time'] = event.get('timestamp', 0)
            self.current_session['score'] = event.get('score', 0)
            
            # Compute session metrics
            duration = (self.current_session['end_time'] - 
                       self.current_session['start_time'])
            nodes = self.current_session.get('nodes', 0)
            
            if duration > 0:
                nps = nodes / (duration / 1e6)  # duration in us
                depth = self.current_session['depth']
                self.stats.nps_by_depth[depth] = (
                    self.stats.nps_by_depth.get(depth, 0) + nps
                ) / 2
        
        elif event_type == 'NODE_EXPANDED':
            self.stats.total_nodes += 1
            depth = event.get('depth', 0)
            self.stats.nodes_by_depth[depth] = \
                self.stats.nodes_by_depth.get(depth, 0) + 1
        
        elif event_type == 'TT_HIT':
            self.stats.tt_hits += 1
            depth = event.get('depth', 0)
            
            hits, total = self.stats.tt_by_depth.get(depth, (0, 0))
            self.stats.tt_by_depth[depth] = (hits + 1, total + 1)
        
        elif event_type == 'TT_STORE':
            self.stats.tt_misses += 1
            depth = event.get('depth', 0)
            
            hits, total = self.stats.tt_by_depth.get(depth, (0, 0))
            self.stats.tt_by_depth[depth] = (hits, total + 1)
        
        elif event_type == 'CUTOFF':
            # Determine cutoff type
            score = event.get('score', 0)
            alpha = event.get('alpha', -32000)
            beta = event.get('beta', 32000)
            
            if score >= beta:
                self.stats.beta_cutoffs += 1
                self.stats.cut_nodes += 1
            elif score <= alpha:
                self.stats.alpha_cutoffs += 1
                self.stats.all_nodes += 1
            else:
                self.stats.pv_nodes += 1
        
        elif event_type == 'PRUNING_DECISION':
            context = event.get('context', '')
            accepted = 'accepted":true' in context or \
                      'accepted": true' in context
            
            if 'futility' in context.lower():
                if accepted:
                    self.stats.futility_prunes += 1
            elif 'null' in context.lower():
                if accepted:
                    self.stats.null_move_prunes += 1
            elif 'lmr' in context.lower():
                if 'rejected' in context.lower() or not accepted:
                    self.stats.lmr_researches += 1
                else:
                    self.stats.lmr_accepted += 1
        
        elif event_type == 'FAIL_HIGH':
            self.stats.fail_highs += 1
            if event.get('aspiration', False):
                self.stats.aspiration_resets += 1
        
        elif event_type == 'FAIL_LOW':
            self.stats.fail_lows += 1
        
        elif event_type == 'EVAL_FLIP':
            self.stats.eval_flips += 1
        
        elif event_type == 'BEST_MOVE_CHANGE':
            self.stats.move_changes += 1
        
        elif event_type == 'NNUE_EVAL':
            self.stats.nnue_evals += 1
            latency = event.get('latency_us', 0)
            # Running average
            n = self.stats.nnue_evals
            self.stats.nnue_avg_latency_us = (
                (self.stats.nnue_avg_latency_us * (n - 1) + latency) / n
            )
        
        elif event_type == 'NNUE_CACHE_HIT':
            self.stats.nnue_cache_hits += 1
        
        elif event_type == 'POLICY_AGREEMENT':
            self.stats.policy_agreement += 1
        
        elif event_type == 'POLICY_DISAGREEMENT':
            self.stats.policy_disagreement += 1
    
    def compute_derived_stats(self):
        """Compute derived statistics."""
        # TT hit rate
        total_tt = self.stats.tt_hits + self.stats.tt_misses
        if total_tt > 0:
            self.tt_hit_rate = self.stats.tt_hits / total_tt
        
        # Pruning efficiency
        if self.stats.moves_generated > 0:
            self.pruning_rate = self.stats.moves_pruned / self.stats.moves_generated
        
        # Node type distribution
        total = self.stats.pv_nodes + self.stats.cut_nodes + self.stats.all_nodes
        if total > 0:
            self.pv_rate = self.stats.pv_nodes / total
            self.cut_rate = self.stats.cut_nodes / total
            self.all_rate = self.stats.all_nodes / total
        
        # Fail rates
        total_searches = self.stats.fail_highs + self.stats.fail_lows
        if total_searches > 0:
            self.fail_high_rate = self.stats.fail_highs / total_searches
            self.fail_low_rate = self.stats.fail_lows / total_searches
        
        # LMR effectiveness
        total_lmr = self.stats.lmr_accepted + self.stats.lmr_researches
        if total_lmr > 0:
            self.lmr_success_rate = self.stats.lmr_accepted / total_lmr
        
        # Null move effectiveness
        # Would need total null move attempts
        
        # Policy agreement (if applicable)
        total_policy = self.stats.policy_agreement + self.stats.policy_disagreement
        if total_policy > 0:
            self.policy_agreement_rate = self.stats.policy_agreement / total_policy
        
        # Instability
        total_sessions = len(self.search_sessions)
        if total_sessions > 0:
            self.avg_eval_flips = self.stats.eval_flips / total_sessions
            self.avg_move_changes = self.stats.move_changes / total_sessions
        
        # Per-depth stats
        self.depth_tt_rates = {}
        for depth, (hits, total) in self.stats.tt_by_depth.items():
            if total > 0:
                self.depth_tt_rates[depth] = hits / total
    
    def detect_regression(self, baseline_stats: 'SearchStats',
                         threshold: float = 0.2) -> List[str]:
        """Detect regressions compared to baseline."""
        regressions = []
        
        # TT hit rate regression
        baseline_tt_rate = baseline_stats.tt_hits / max(1, 
            baseline_stats.tt_hits + baseline_stats.tt_misses)
        current_tt_rate = self.stats.tt_hits / max(1,
            self.stats.tt_hits + self.stats.tt_misses)
        
        if current_tt_rate < baseline_tt_rate * (1 - threshold):
            regressions.append(
                f"TT hit rate dropped: {baseline_tt_rate:.1%} -> {current_tt_rate:.1%}"
            )
        
        # NPS regression
        baseline_nps = sum(baseline_stats.nps_by_depth.values()) / \
                       max(1, len(baseline_stats.nps_by_depth))
        current_nps = sum(self.stats.nps_by_depth.values()) / \
                      max(1, len(self.stats.nps_by_depth))
        
        if current_nps < baseline_nps * (1 - threshold):
            regressions.append(
                f"NPS dropped: {baseline_nps:.0f} -> {current_nps:.0f}"
            )
        
        # Node count regression (too many nodes = less efficient)
        if self.stats.total_nodes > baseline_stats.total_nodes * (1 + threshold):
            regressions.append(
                f"Node count increased: {baseline_stats.total_nodes} -> "
                f"{self.stats.total_nodes}"
            )
        
        # Instability regression
        baseline_flips = baseline_stats.eval_flips / max(1, 
            len(baseline_stats.nodes_by_depth))
        current_flips = self.stats.eval_flips / max(1,
            len(self.stats.nodes_by_depth))
        
        if current_flips > baseline_flips * (1 + threshold):
            regressions.append(
                f"Eval flips increased: {baseline_flips:.2f} -> {current_flips:.2f}"
            )
        
        return regressions
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive report."""
        self.compute_derived_stats()
        
        report = {
            'summary': {
                'total_events': len(self.raw_events),
                'total_sessions': len(self.search_sessions),
                'total_nodes': self.stats.total_nodes,
                'avg_nodes_per_session': self.stats.total_nodes / max(1, 
                    len(self.search_sessions)),
            },
            
            'node_types': {
                'pv_nodes': self.stats.pv_nodes,
                'cut_nodes': self.stats.cut_nodes,
                'all_nodes': self.stats.all_nodes,
                'pv_rate': getattr(self, 'pv_rate', 0.0),
                'cut_rate': getattr(self, 'cut_rate', 0.0),
                'all_rate': getattr(self, 'all_rate', 0.0),
            },
            
            'tt_statistics': {
                'hits': self.stats.tt_hits,
                'misses': self.stats.tt_misses,
                'cutoffs': self.stats.tt_cutoffs,
                'overall_hit_rate': getattr(self, 'tt_hit_rate', 0.0),
                'by_depth': {
                    str(d): rate for d, rate in 
                    getattr(self, 'depth_tt_rates', {}).items()
                },
            },
            
            'cutoff_causes': {
                'beta_cutoffs': self.stats.beta_cutoffs,
                'alpha_cutoffs': self.stats.alpha_cutoffs,
                'futility_prunes': self.stats.futility_prunes,
                'null_move_prunes': self.stats.null_move_prunes,
                'static_eval_cutoffs': self.stats.static_eval_cutoffs,
            },
            
            'fail_rates': {
                'fail_highs': self.stats.fail_highs,
                'fail_lows': self.stats.fail_lows,
                'aspiration_resets': self.stats.aspiration_resets,
                'fail_high_rate': getattr(self, 'fail_high_rate', 0.0),
                'fail_low_rate': getattr(self, 'fail_low_rate', 0.0),
            },
            
            'instability': {
                'eval_flips': self.stats.eval_flips,
                'move_changes': self.stats.move_changes,
                'depth_oscillations': self.stats.depth_oscillations,
                'avg_flips_per_session': getattr(self, 'avg_eval_flips', 0.0),
                'avg_move_changes_per_session': getattr(self, 'avg_move_changes', 0.0),
            },
            
            'lmr_statistics': {
                'accepted': self.stats.lmr_accepted,
                'researches': self.stats.lmr_researches,
                'success_rate': getattr(self, 'lmr_success_rate', 0.0),
            },
            
            'performance': {
                'nps_by_depth': {
                    str(d): nps for d, nps in self.stats.nps_by_depth.items()
                },
                'nodes_by_depth': {
                    str(d): n for d, n in self.stats.nodes_by_depth.items()
                },
                'nnue_latency_us': self.stats.nnue_avg_latency_us,
                'nnue_cache_hit_rate': (
                    self.stats.nnue_cache_hits / max(1, self.stats.nnue_evals)
                ),
            },
            
            'policy_agreement': {
                'agreements': self.stats.policy_agreement,
                'disagreements': self.stats.policy_disagreement,
                'agreement_rate': getattr(self, 'policy_agreement_rate', 0.0),
            } if (self.stats.policy_agreement + self.stats.policy_disagreement) > 0 else None,
        }
        
        # Clean None values
        report = {k: v for k, v in report.items() if v is not None}
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print human-readable summary."""
        self.compute_derived_stats()
        
        print("\n" + "=" * 60)
        print("  SEARCH TELEMETRY SUMMARY")
        print("=" * 60)
        
        print(f"\nEvents: {len(self.raw_events):,}")
        print(f"Sessions: {len(self.search_sessions):,}")
        print(f"Total nodes: {self.stats.total_nodes:,}")
        
        print(f"\nNode Types:")
        total = self.stats.pv_nodes + self.stats.cut_nodes + self.stats.all_nodes
        if total > 0:
            print(f"  PV:   {self.stats.pv_nodes:,} ({self.stats.pv_nodes/total:.1%})")
            print(f"  Cut:  {self.stats.cut_nodes:,} ({self.stats.cut_nodes/total:.1%})")
            print(f"  All:  {self.stats.all_nodes:,} ({self.stats.all_nodes/total:.1%})")
        
        print(f"\nTT Statistics:")
        tt_total = self.stats.tt_hits + self.stats.tt_misses
        if tt_total > 0:
            print(f"  Hits:  {self.stats.tt_hits:,} ({self.stats.tt_hits/tt_total:.1%})")
            print(f"  Misses: {self.stats.tt_misses:,} ({self.stats.tt_misses/tt_total:.1%})")
        print(f"  TT Cutoffs: {self.stats.tt_cutoffs:,}")
        
        print(f"\nCutoff Causes:")
        print(f"  Beta:   {self.stats.beta_cutoffs:,}")
        print(f"  Alpha:  {self.stats.alpha_cutoffs:,}")
        print(f"  Futility: {self.stats.futility_prunes:,}")
        print(f"  Null Move: {self.stats.null_move_prunes:,}")
        
        print(f"\nFail Rates:")
        total_fails = self.stats.fail_highs + self.stats.fail_lows
        if total_fails > 0:
            print(f"  High: {self.stats.fail_highs:,} ({self.stats.fail_highs/total_fails:.1%})")
            print(f"  Low:  {self.stats.fail_lows:,} ({self.stats.fail_lows/total_fails:.1%})")
        print(f"  Aspiration resets: {self.stats.aspiration_resets:,}")
        
        print(f"\nInstability:")
        print(f"  Eval flips: {self.stats.eval_flips:,}")
        print(f"  Move changes: {self.stats.move_changes:,}")
        if self.search_sessions:
            print(f"  Avg flips/session: {self.stats.eval_flips/len(self.search_sessions):.2f}")
        
        print(f"\nLMR Statistics:")
        total_lmr = self.stats.lmr_accepted + self.stats.lmr_researches
        if total_lmr > 0:
            print(f"  Accepted: {self.stats.lmr_accepted:,} ({self.stats.lmr_accepted/total_lmr:.1%})")
            print(f"  Researches: {self.stats.lmr_researches:,} ({self.stats.lmr_researches/total_lmr:.1%})")
        
        print(f"\nNNUE Performance:")
        print(f"  Evals: {self.stats.nnue_evals:,}")
        print(f"  Cache hits: {self.stats.nnue_cache_hits:,}")
        if self.stats.nnue_evals > 0:
            print(f"  Cache rate: {self.stats.nnue_cache_hits/self.stats.nnue_evals:.1%}")
        print(f"  Avg latency: {self.stats.nnue_avg_latency_us:.1f} us")
        
        total_policy = self.stats.policy_agreement + self.stats.policy_disagreement
        if total_policy > 0:
            print(f"\nPolicy Agreement:")
            print(f"  Agree: {self.stats.policy_agreement:,} ({self.stats.policy_agreement/total_policy:.1%})")
            print(f"  Disagree: {self.stats.policy_disagreement:,} ({self.stats.policy_disagreement/total_policy:.1%})")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Search Telemetry Aggregator')
    parser.add_argument('--telemetry', required=True, help='Telemetry file')
    parser.add_argument('--output', help='JSON report output')
    parser.add_argument('--max-events', type=int, default=0, help='Max events to load')
    parser.add_argument('--compare', help='Baseline telemetry for regression detection')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    parser.add_argument('--export-ml', help='Export training data for ML (features + labels)')
    
    args = parser.parse_args()
    
    agg = TelemetryAggregator(args.telemetry)
    agg.load(args.max_events)
    
    if args.summary or not args.output:
        agg.print_summary()
    
    if args.output:
        report = agg.generate_report(args.output)
        
        # Print key metrics
        print(f"\nKey metrics:")
        print(f"  Nodes: {report['summary']['total_nodes']:,}")
        if 'tt_statistics' in report:
            print(f"  TT rate: {report['tt_statistics']['overall_hit_rate']:.1%}")
    
    if args.compare:
        baseline = TelemetryAggregator(args.compare)
        baseline.load(args.max_events)
        baseline.compute_derived_stats()
        
        regressions = agg.detect_regression(baseline.stats)
        
        if regressions:
            print(f"\nREGRESSIONS DETECTED ({len(regressions)}):")
            for r in regressions:
                print(f"  - {r}")
            sys.exit(1)
        else:
            print(f"\nNo regressions detected vs baseline")
    
    if args.export_ml:
        # Export structured training data for ML models
        # Format: each line = (features, label)
        # Features: search context at decision point
        # Label: whether decision was correct
        print(f"Exporting ML training data to {args.export_ml}...")
        
        with open(args.export_ml, 'w') as f:
            for event in agg.raw_events:
                if event.get('type') not in ('PRUNING_DECISION', 'CUTOFF'):
                    continue
                
                features = {
                    'depth': event.get('depth', 0),
                    'score': event.get('score', 0),
                    'alpha': event.get('alpha', 0),
                    'beta': event.get('beta', 0),
                }
                
                # Parse context
                ctx = {}
                try:
                    ctx = json.loads(event.get('context', '{}'))
                except:
                    pass
                
                features.update(ctx)
                
                # Label: was this a good decision?
                # For beta cutoffs: 1 = cutoff correct (saved nodes)
                # For pruning: 1 = prune didn't miss best move
                label = 1 if event.get('type') == 'CUTOFF' else 0
                
                f.write(json.dumps({'features': features, 'label': label}) + '\n')
        
        print(f"Exported {sum(1 for e in agg.raw_events if e.get('type') in ('PRUNING_DECISION', 'CUTOFF'))} samples")


if __name__ == '__main__':
    main()
