#!/usr/bin/env python3
"""
Fishtest-style Patch Queue for Nexus Infinite
Automated build → test → SPRT → accept/reject pipeline

Usage:
    python scripts/fishtest_queue.py \
        --base-branch main \
        --test-branch feature-xxx \
        --tc "10+0.1" \
        --elo0 0 --elo1 5 \
        --threads 4

Architecture:
    ┌─────────────┐    ┌──────────────┐    ┌──────────┐
    │ Git Checkout│───>│ Build Engine │───>│ Quicktest│
    └─────────────┘    └──────────────┘    └────┬─────┘
                                                  │
    ┌─────────────┐    ┌──────────────┐    ┌─────▼─────┐
    │ Accept/Reject│<───│ SPRT Results  │<───│ Gauntlet │
    └─────────────┘    └──────────────┘    └───────────┘
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import threading
import queue as threading_queue


@dataclass
class PatchConfig:
    name: str
    base_branch: str
    test_branch: str
    elo0: float
    elo1: float
    tc: str
    threads: int
    games: int
    priority: int  # 1-10, higher = more important
    description: str = ""


@dataclass
class PatchResult:
    patch_name: str
    status: str  # "pending", "building", "testing", "accept", "reject", "error"
    build_time: float = 0.0
    test_time: float = 0.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    llr: float = 0.0
    estimated_elo: float = 0.0
    error_message: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PatchQueue:
    """Queue manager for patch testing"""
    
    def __init__(self, config_file: str = "fishtest_config.json"):
        self.config_file = config_file
        self.queue: threading_queue.PriorityQueue = threading_queue.PriorityQueue()
        self.results: List[PatchResult] = []
        self.running = False
        self.workers: List[threading.Thread] = []
        
        self.build_dir = Path("build_fishtest")
        self.results_dir = Path("fishtest_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def add_patch(self, config: PatchConfig):
        """Add patch to queue (lower priority value = higher priority)"""
        self.queue.put((-config.priority, time.time(), config))
        print(f"Added patch '{config.name}' to queue (priority: {config.priority})")
    
    def load_queue(self):
        """Load pending patches from config"""
        if not Path(self.config_file).exists():
            return
        
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        for patch_data in data.get('queue', []):
            config = PatchConfig(**patch_data)
            self.add_patch(config)
    
    def save_results(self):
        """Save all results"""
        filename = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Also save cumulative
        cumulative = self.results_dir / "cumulative_results.json"
        with open(cumulative, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
    
    def run_single_test(self, config: PatchConfig) -> PatchResult:
        """Run complete test pipeline for one patch"""
        result = PatchResult(
            patch_name=config.name,
            status="building",
            timestamp=datetime.now().isoformat()
        )
        
        build_start = time.time()
        
        try:
            # Step 1: Checkout and build
            print(f"\n{'='*60}")
            print(f"Testing patch: {config.name}")
            print(f"{'='*60}")
            
            # Clean build directory
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
            self.build_dir.mkdir()
            
            # Checkout test branch
            print(f"Checking out {config.test_branch}...")
            subprocess.run(
                ["git", "checkout", config.test_branch],
                capture_output=True, check=True
            )
            
            # Build test version
            print("Building test version...")
            build_result = subprocess.run(
                ["cmake", "-B", str(self.build_dir), "-S", ".", 
                 "-DCMAKE_BUILD_TYPE=Release"],
                capture_output=True, text=True
            )
            if build_result.returncode != 0:
                raise RuntimeError(f"CMake failed: {build_result.stderr}")
            
            build_result = subprocess.run(
                ["cmake", "--build", str(self.build_dir), "-j"],
                capture_output=True, text=True
            )
            if build_result.returncode != 0:
                raise RuntimeError(f"Build failed: {build_result.stderr}")
            
            test_binary = self.build_dir / "nexus"
            if not test_binary.exists():
                raise RuntimeError("Binary not found after build")
            
            result.build_time = time.time() - build_start
            print(f"Build completed in {result.build_time:.1f}s")
            
            # Step 2: Quick validation
            print("Running quicktest...")
            quicktest = subprocess.run(
                [str(test_binary)],
                input="quicktest\nquit\n",
                capture_output=True, text=True, timeout=60
            )
            if "FAILED" in quicktest.stdout or quicktest.returncode != 0:
                raise RuntimeError("Quicktest failed - patch has critical bugs")
            
            print("Quicktest passed!")
            
            # Step 3: Checkout base and build
            print(f"Building base version ({config.base_branch})...")
            base_dir = Path("build_base")
            if base_dir.exists():
                shutil.rmtree(base_dir)
            base_dir.mkdir()
            
            subprocess.run(
                ["git", "checkout", config.base_branch],
                capture_output=True, check=True
            )
            
            subprocess.run(
                ["cmake", "-B", str(base_dir), "-S", ".",
                 "-DCMAKE_BUILD_TYPE=Release"],
                capture_output=True, check=True
            )
            subprocess.run(
                ["cmake", "--build", str(base_dir), "-j"],
                capture_output=True, check=True
            )
            
            base_binary = base_dir / "nexus"
            
            # Step 4: Run SPRT
            result.status = "testing"
            test_start = time.time()
            
            print(f"\nRunning SPRT: H0={config.elo0} Elo, H1={config.elo1} Elo")
            print(f"Time control: {config.tc}")
            print(f"Max games: {config.games}")
            print(f"Threads: {config.threads}")
            
            sprt_cmd = [
                sys.executable, "scripts/sprt_test.py",
                str(test_binary), str(base_binary),
                "--elo0", str(config.elo0),
                "--elo1", str(config.elo1),
                "--tc", config.tc,
                "--games", str(config.games),
                "--threads", str(config.threads),
            ]
            
            sprt_result = subprocess.run(
                sprt_cmd,
                capture_output=True, text=True,
                timeout=86400  # 24 hours max
            )
            
            result.test_time = time.time() - test_start
            
            # Parse SPRT output
            output = sprt_result.stdout + sprt_result.stderr
            
            if "ACCEPT" in output:
                result.status = "accept"
            elif "REJECT" in output:
                result.status = "reject"
            else:
                result.status = "inconclusive"
            
            # Try to extract stats
            for line in output.split('\n'):
                if "Games:" in line:
                    try:
                        parts = line.split()
                        result.games_played = int(parts[1].rstrip(','))
                        result.wins = int(parts[3].rstrip(','))
                        result.draws = int(parts[5].rstrip(','))
                        result.losses = int(parts[7].rstrip(','))
                    except (ValueError, IndexError):
                        pass
                
                if "LLR:" in line:
                    try:
                        result.llr = float(line.split("LLR:")[1].split()[0])
                    except (ValueError, IndexError):
                        pass
                
                if "Estimated Elo:" in line:
                    try:
                        result.estimated_elo = float(
                            line.split("Estimated Elo:")[1].split()[0]
                        )
                    except (ValueError, IndexError):
                        pass
            
        except subprocess.TimeoutExpired:
            result.status = "timeout"
            result.error_message = "SPRT test timed out (24h limit)"
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            print(f"ERROR testing {config.name}: {e}")
        
        return result
    
    def worker(self, worker_id: int):
        """Worker thread that processes patches from queue"""
        while self.running:
            try:
                _, _, config = self.queue.get(timeout=5)
            except threading_queue.Empty:
                continue
            
            print(f"\n[Worker {worker_id}] Processing: {config.name}")
            
            result = self.run_single_test(config)
            self.results.append(result)
            self.save_results()
            
            print(f"\n[Worker {worker_id}] Result: {result.status.upper()}")
            print(f"  Elo estimate: {result.estimated_elo:+.1f}")
            print(f"  LLR: {result.llr:.3f}")
            print(f"  Games: {result.games_played}")
            
            self.queue.task_done()
    
    def run(self, num_workers: int = 1):
        """Start processing queue with multiple workers"""
        self.running = True
        
        print(f"\n{'='*60}")
        print("Fishtest-style Patch Queue for Nexus Infinite")
        print(f"{'='*60}")
        print(f"Workers: {num_workers}")
        print(f"Queue size: {self.queue.qsize()}")
        print(f"Results dir: {self.results_dir}")
        
        # Start workers
        for i in range(num_workers):
            t = threading.Thread(target=self.worker, args=(i,))
            t.daemon = True
            t.start()
            self.workers.append(t)
        
        try:
            # Wait for all tasks
            self.queue.join()
        except KeyboardInterrupt:
            print("\n\nShutting down workers...")
            self.running = False
            for t in self.workers:
                t.join(timeout=10)
        
        # Final save
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary of all results"""
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        status_counts = {}
        for r in self.results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        
        print(f"\nResults by status:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status:15s}: {count}")
        
        print(f"\nAccepted patches:")
        for r in self.results:
            if r.status == "accept":
                print(f"  {r.patch_name}: +{r.estimated_elo:.1f} Elo "
                      f"({r.games_played} games)")
        
        print(f"\nRejected patches:")
        for r in self.results:
            if r.status == "reject":
                print(f"  {r.patch_name}: {r.estimated_elo:+.1f} Elo "
                      f"({r.games_played} games)")
        
        print(f"\nTotal patches tested: {len(self.results)}")
        print(f"Total games played: {sum(r.games_played for r in self.results)}")


def main():
    parser = argparse.ArgumentParser(
        description='Fishtest-style Patch Queue for Nexus Infinite'
    )
    
    parser.add_argument('--config', default='fishtest_config.json',
                        help='Queue configuration file')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel test workers')
    parser.add_argument('--add', action='store_true',
                        help='Add patch interactively')
    parser.add_argument('--status', action='store_true',
                        help='Show queue status')
    
    args = parser.parse_args()
    
    queue = PatchQueue(args.config)
    
    if args.add:
        # Interactive patch addition
        print("Add new patch to queue:")
        name = input("Patch name: ")
        base = input("Base branch [main]: ") or "main"
        test = input("Test branch: ")
        elo0 = float(input("Null hypothesis Elo [0]: ") or "0")
        elo1 = float(input("Target Elo gain [5]: ") or "5")
        tc = input("Time control [10+0.1]: ") or "10+0.1"
        games = int(input("Max games [10000]: ") or "10000")
        threads = int(input("Threads [4]: ") or "4")
        priority = int(input("Priority [5]: ") or "5")
        desc = input("Description: ")
        
        config = PatchConfig(
            name=name, base_branch=base, test_branch=test,
            elo0=elo0, elo1=elo1, tc=tc, threads=threads,
            games=games, priority=priority, description=desc
        )
        queue.add_patch(config)
        
        # Save to config
        with open(args.config, 'w') as f:
            json.dump({
                'queue': [asdict(config)]
            }, f, indent=2)
        
        print(f"Patch added to queue!")
        return
    
    if args.status:
        # Show status
        print(f"Queue: {queue.queue.qsize()} pending")
        if Path(queue.results_dir / "cumulative_results.json").exists():
            with open(queue.results_dir / "cumulative_results.json") as f:
                results = json.load(f)
            print(f"Completed: {len(results)} patches")
            
            for r in results[-5:]:  # Last 5
                print(f"  {r['patch_name']}: {r['status']} "
                      f"({r.get('estimated_elo', 0):+.1f} Elo)")
        return
    
    # Load queue and run
    queue.load_queue()
    queue.run(num_workers=args.workers)


if __name__ == '__main__':
    main()
