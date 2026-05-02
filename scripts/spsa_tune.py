#!/usr/bin/env python3
"""
SPSA (Simultaneous Perturbation Stochastic Approximation) Auto Parameter Tuning.

Automatically optimizes engine search parameters using minimal games.
Based on Stockfish's tuning methodology.

Usage:
    python scripts/spsa_tune.py --engine ./nexus --games 100 --iterations 1000
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path


@dataclass
class TunableParam:
    """A single tunable parameter."""
    name: str
    value: int  # Current value
    min_val: int
    max_val: int
    step: int = 1
    c: float = 0.0  # SPSA perturbation scale (auto-computed)
    a: float = 0.0  # SPSA gain sequence (auto-computed)
    
    def perturb(self, delta: float) -> int:
        """Perturb value by delta * c."""
        new_val = self.value + int(delta * self.c * self.step)
        return max(self.min_val, min(self.max_val, new_val))
    
    def update(self, gradient_estimate: float, iteration: int, A: float, alpha: float, a0: float):
        """Update value based on gradient estimate."""
        ak = a0 / ((iteration + 1 + A) ** alpha)
        self.value = int(self.value - ak * gradient_estimate * self.c)
        self.value = max(self.min_val, min(self.max_val, self.value))


# Default parameter set for chess engine search tuning
DEFAULT_PARAMS = [
    TunableParam("razor_margin", 500, 300, 700, step=10),
    TunableParam("futility_margin_base", 200, 100, 400, step=5),
    TunableParam("futility_margin_factor", 100, 50, 200, step=5),
    TunableParam("null_move_reduction", 3, 2, 6, step=1),
    TunableParam("null_move_verification_depth", 12, 8, 20, step=1),
    TunableParam("lmr_base", 100, 50, 200, step=5),
    TunableParam("lmr_divisor", 200, 100, 400, step=10),
    TunableParam("lmr_pv_node_bonus", 100, 0, 300, step=10),
    TunableParam("lmr_improving_bonus", 50, 0, 200, step=10),
    TunableParam("probcut_margin", 100, 50, 250, step=5),
    TunableParam("singular_extension_margin", 50, 20, 100, step=2),
    TunableParam("aspiration_window", 15, 5, 50, step=1),
    TunableParam("aspiration_delta", 20, 10, 50, step=1),
    TunableParam("seep_quiet_threshold", -15, -30, 0, step=1),
    TunableParam("seep_noisy_threshold", -45, -100, -20, step=2),
    TunableParam("history_prune_threshold", -2000, -5000, -500, step=100),
    TunableParam("countermove_bonus", 500, 200, 1000, step=50),
    TunableParam("extensions_check", 1, 0, 2, step=1),
    TunableParam("extensions_one_reply", 1, 0, 2, step=1),
    TunableParam("extensions_recapture", 1, 0, 2, step=1),
    TunableParam("extensions_pawn_push", 1, 0, 2, step=1),
    TunableParam("eval_trend_threshold", 30, 10, 100, step=2),
    TunableParam("time_management_mtg", 30, 20, 60, step=1),
    TunableParam("time_management_movestogo", 25, 15, 40, step=1),
]


class SPSA:
    """SPSA optimizer for engine parameters."""
    
    def __init__(self, 
                 params: List[TunableParam],
                 A: float = 100.0,
                 alpha: float = 0.602,  # Standard SPSA: 0.602
                 gamma: float = 0.101,  # Standard SPSA: 0.101
                 c0: float = 2.0,       # Initial perturbation scale
                 a0: float = 1.0,       # Initial gain
                 ):
        self.params = params
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.c0 = c0
        self.a0 = a0
        
        # Compute c and a for each parameter
        p = len(params)
        for param in params:
            param.c = c0 * (param.max_val - param.min_val)
            param.a = a0 / p  # Scale by number of params
    
    def get_ck(self, iteration: int) -> float:
        """Perturbation magnitude at iteration k."""
        return self.c0 / ((iteration + 1) ** self.gamma)
    
    def generate_perturbation(self) -> List[float]:
        """Generate random delta vector (Bernoulli +/- 1)."""
        return [random.choice([-1.0, 1.0]) for _ in self.params]
    
    def apply_params(self, param_values: Dict[str, int], 
                     engine_path: str) -> List[str]:
        """Build command line with parameter overrides."""
        cmd = [engine_path]
        for name, value in param_values.items():
            # Engine should accept --param-name value format
            cmd.extend([f"--{name}", str(value)])
        return cmd
    
    def evaluate(self, engine_path: str, opponent_path: str,
                 param_values: Dict[str, int],
                 games: int, tc: str = "10+0.1",
                 threads: int = 1) -> float:
        """Run games and return score (0-1, higher is better)."""
        # Use fastchess or cutechess-cli for game evaluation
        cmd = [
            "python", "scripts/run_sprt.py",
            "--engine-new", engine_path,
            "--engine-base", opponent_path,
            "--games", str(games),
            "--tc", tc,
            "--threads", str(threads),
            "--param-overrides", json.dumps(param_values),
            "--output", "/tmp/spsa_eval.json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   timeout=games * 30)
            
            if os.path.exists("/tmp/spsa_eval.json"):
                with open("/tmp/spsa_eval.json", 'r') as f:
                    data = json.load(f)
                    # Return win rate (0-1)
                    wins = data.get('wins', 0)
                    losses = data.get('losses', 0)
                    draws = data.get('draws', 0)
                    total = wins + losses + draws
                    if total > 0:
                        return (wins + 0.5 * draws) / total
            
            return 0.5  # Default: draw
            
        except subprocess.TimeoutExpired:
            print(f"WARNING: Evaluation timeout for {param_values}")
            return 0.5
        except Exception as e:
            print(f"ERROR: Evaluation failed: {e}")
            return 0.5
    
    def step(self, iteration: int, engine_path: str, 
             opponent_path: str, games: int, tc: str):
        """Single SPSA iteration."""
        # Generate perturbation
        delta = self.generate_perturbation()
        ck = self.get_ck(iteration)
        
        # Perturb parameters: theta + ck * delta
        params_plus = {}
        params_minus = {}
        base_values = {}
        
        for i, param in enumerate(self.params):
            base_values[param.name] = param.value
            perturb = int(delta[i] * ck * param.c / param.step) * param.step
            params_plus[param.name] = max(param.min_val, 
                                          min(param.max_val, param.value + perturb))
            params_minus[param.name] = max(param.min_val, 
                                           min(param.max_val, param.value - perturb))
        
        # Evaluate both directions
        print(f"  Iteration {iteration+1}: evaluating +perturbation...")
        y_plus = self.evaluate(engine_path, opponent_path, params_plus, 
                              games // 2, tc)
        
        print(f"  Iteration {iteration+1}: evaluating -perturbation...")
        y_minus = self.evaluate(engine_path, opponent_path, params_minus,
                               games // 2, tc)
        
        # Gradient estimate
        gradient = (y_plus - y_minus) / (2 * ck)
        
        # Update parameters
        print(f"  Gradient: {gradient:.4f} (y+={y_plus:.3f}, y-={y_minus:.3f})")
        
        for i, param in enumerate(self.params):
            # SPSA update: theta_{k+1} = theta_k - a_k * g_k
            ak = param.a / ((iteration + 1 + self.A) ** self.alpha)
            
            # Gradient component for this param
            gk = gradient * delta[i]  # Approximate gradient component
            
            old_value = param.value
            param.value = int(param.value - ak * gk)
            param.value = max(param.min_val, min(param.max_val, param.value))
            
            if param.value != old_value:
                print(f"    {param.name}: {old_value} -> {param.value}")
        
        return (y_plus + y_minus) / 2  # Average performance
    
    def optimize(self, iterations: int, engine_path: str,
                opponent_path: str, games_per_eval: int,
                tc: str = "10+0.1", save_interval: int = 50):
        """Run full SPSA optimization."""
        
        print("=" * 60)
        print("SPSA Parameter Optimization")
        print("=" * 60)
        print(f"Parameters: {len(self.params)}")
        print(f"Iterations: {iterations}")
        print(f"Games per eval: {games_per_eval}")
        print(f"Time control: {tc}")
        print()
        
        best_params = None
        best_score = 0.5
        
        for k in range(iterations):
            start_time = time.time()
            
            score = self.step(k, engine_path, opponent_path, 
                            games_per_eval, tc)
            
            elapsed = time.time() - start_time
            
            print(f"Iteration {k+1}/{iterations}: score={score:.4f} "
                  f"time={elapsed:.1f}s")
            
            if score > best_score:
                best_score = score
                best_params = {p.name: p.value for p in self.params}
                print(f"  *** New best: {score:.4f} ***")
            
            # Periodic save
            if (k + 1) % save_interval == 0:
                self.save_checkpoint(k + 1, score)
        
        # Final save
        self.save_checkpoint(iterations, best_score, best_params, 
                            final=True)
        
        return best_params, best_score
    
    def save_checkpoint(self, iteration: int, score: float,
                       best_params: Optional[Dict] = None,
                       final: bool = False):
        """Save current parameter values."""
        data = {
            'iteration': iteration,
            'score': score,
            'params': {p.name: p.value for p in self.params},
        }
        
        if best_params:
            data['best_params'] = best_params
            data['best_score'] = score
        
        suffix = "_final" if final else f"_iter{iteration}"
        filename = f"spsa_params{suffix}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved checkpoint: {filename}")
    
    def export_cpp(self, filename: str):
        """Export as C++ constexpr header."""
        lines = [
            "#pragma once",
            "// Auto-generated SPSA-tuned parameters",
            "",
        ]
        
        for param in self.params:
            lines.append(f"constexpr int {param.name.upper()} = {param.value};")
        
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"Exported C++ header: {filename}")


def load_params_from_json(filename: str) -> List[TunableParam]:
    """Load parameters from existing tuning result."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    params = []
    for p in DEFAULT_PARAMS:
        if p.name in data.get('best_params', data.get('params', {})):
            p.value = data['best_params'].get(p.name, 
                                              data['params'].get(p.name, p.value))
        params.append(p)
    
    return params


def main():
    parser = argparse.ArgumentParser(description='SPSA Parameter Tuning')
    parser.add_argument('--engine', required=True, help='Engine binary path')
    parser.add_argument('--opponent', required=True, 
                        help='Opponent engine (or same for self-play)')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--games', type=int, default=100,
                        help='Games per evaluation (split between +/-)')
    parser.add_argument('--tc', default='10+0.1', help='Time control')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--resume', help='Resume from checkpoint JSON')
    parser.add_argument('--output', default='tuned_params.json')
    parser.add_argument('--cpp-header', help='Export C++ header')
    parser.add_argument('--param-filter', help='Comma-separated param names to tune')
    
    args = parser.parse_args()
    
    # Select parameters
    if args.param_filter:
        allowed = set(args.param_filter.split(','))
        params = [p for p in DEFAULT_PARAMS if p.name in allowed]
    elif args.resume:
        params = load_params_from_json(args.resume)
    else:
        params = DEFAULT_PARAMS
    
    print(f"Tuning {len(params)} parameters")
    
    # Run SPSA
    spsa = SPSA(params)
    
    best_params, best_score = spsa.optimize(
        iterations=args.iterations,
        engine_path=args.engine,
        opponent_path=args.opponent,
        games_per_eval=args.games,
        tc=args.tc
    )
    
    # Export
    with open(args.output, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score,
            'tuned_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }, f, indent=2)
    
    print(f"\nTuning complete!")
    print(f"Best score: {best_score:.4f}")
    print(f"Parameters saved: {args.output}")
    
    if args.cpp_header:
        spsa.export_cpp(args.cpp_header)


if __name__ == '__main__':
    main()
