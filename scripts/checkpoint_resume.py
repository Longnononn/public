#!/usr/bin/env python3
"""
Checkpoint Manager for Micro-Training Pipeline
Handles checkpoint persistence via GitHub artifacts for infinite training.

Usage:
    python scripts/checkpoint_resume.py \
        --download \
        --artifact-prefix checkpoint \
        --output-dir checkpoints/
    
    python scripts/checkpoint_resume.py \
        --upload \
        --checkpoint-file checkpoints/latest.ckpt \
        --artifact-name checkpoint-$(date +%s)

Architecture:
    Each training job:
    1. Download latest checkpoint artifact from previous run
    2. Resume training for 1 epoch (micro-train)
    3. Save new checkpoint as artifact
    4. Trigger next workflow with new artifact name

This enables infinite training despite GitHub's 6h job limit.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CheckpointManager:
    """Manages checkpoint lifecycle for GitHub Actions micro-training"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_history.json"
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if checkpoints:
            print(f"Latest checkpoint: {checkpoints[0]} (epoch {self._extract_epoch(checkpoints[0])})")
            return checkpoints[0]
        
        # Also check for .pt files (PyTorch format)
        pt_files = sorted(
            self.checkpoint_dir.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if pt_files:
            print(f"Latest checkpoint: {pt_files[0]}")
            return pt_files[0]
        
        print("No checkpoint found")
        return None
    
    def _extract_epoch(self, path: Path) -> int:
        """Extract epoch number from checkpoint filename"""
        name = path.stem
        # Try patterns: epoch_5, checkpoint_epoch_3, best_epoch7
        import re
        match = re.search(r'epoch[_-]?(\d+)', name)
        if match:
            return int(match.group(1))
        return 0
    
    def save_checkpoint(self, 
                        source_path: str,
                        epoch: int,
                        loss: float,
                        val_loss: float,
                        run_id: str = "") -> Path:
        """Save checkpoint with metadata"""
        
        source = Path(source_path)
        
        # Generate checkpoint name with epoch and timestamp
        timestamp = int(time.time())
        name = f"checkpoint_epoch{epoch:04d}_run{run_id}_{timestamp}.ckpt"
        dest = self.checkpoint_dir / name
        
        # Copy checkpoint
        if source != dest:
            import shutil
            shutil.copy2(source, dest)
        
        # Update metadata
        metadata = self.load_metadata()
        
        entry = {
            "file": str(dest.name),
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "timestamp": timestamp,
            "run_id": run_id,
            "size_bytes": dest.stat().st_size,
        }
        
        metadata["checkpoints"].append(entry)
        metadata["latest"] = entry
        
        self.save_metadata(metadata)
        
        print(f"Checkpoint saved: {dest}")
        print(f"  Epoch: {epoch}, Loss: {loss:.6f}, Val: {val_loss:.6f}")
        
        return dest
    
    def load_metadata(self) -> Dict:
        """Load checkpoint metadata history"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        
        return {
            "checkpoints": [],
            "latest": None,
            "total_epochs": 0,
        }
    
    def save_metadata(self, metadata: Dict):
        """Save checkpoint metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_training_state(self) -> Dict:
        """Get current training state for resume"""
        metadata = self.load_metadata()
        latest = metadata.get("latest")
        
        if latest:
            return {
                "resume_from": str(self.checkpoint_dir / latest["file"]),
                "start_epoch": latest["epoch"] + 1,
                "total_epochs": metadata.get("total_epochs", 0),
                "last_loss": latest.get("loss", 0.0),
                "last_val_loss": latest.get("val_loss", 0.0),
            }
        
        return {
            "resume_from": None,
            "start_epoch": 0,
            "total_epochs": 0,
            "last_loss": 0.0,
            "last_val_loss": 0.0,
        }
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keep only N most recent"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        removed = 0
        for old in checkpoints[keep_last:]:
            old.unlink()
            removed += 1
        
        if removed > 0:
            print(f"Cleaned up {removed} old checkpoints, kept {keep_last}")
        
        return removed
    
    def estimate_progress(self, target_epochs: int = 100) -> Dict:
        """Estimate training progress"""
        state = self.get_training_state()
        
        if target_epochs <= 0:
            return {"progress": 0.0, "remaining": 0}
        
        completed = state["total_epochs"]
        progress = min(100.0, 100.0 * completed / target_epochs)
        remaining = max(0, target_epochs - completed)
        
        return {
            "completed_epochs": completed,
            "target_epochs": target_epochs,
            "progress_percent": progress,
            "remaining_epochs": remaining,
        }


def download_artifact(artifact_name: str, output_dir: str = "checkpoints") -> bool:
    """Download checkpoint artifact using gh CLI or curl"""
    
    print(f"Attempting to download artifact: {artifact_name}")
    
    # Try gh CLI first
    gh = which("gh")
    if gh:
        try:
            # List artifacts
            result = subprocess.run(
                [gh, "run", "list", "--json", "databaseId,status,headBranch"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                runs = json.loads(result.stdout)
                if runs:
                    latest_run = runs[0]
                    run_id = latest_run.get("databaseId")
                    
                    # Download specific artifact
                    subprocess.run(
                        [gh, "run", "download", str(run_id), 
                         "--name", artifact_name, "--dir", output_dir],
                        check=True, timeout=60
                    )
                    
                    print(f"Downloaded artifact to {output_dir}")
                    return True
        
        except Exception as e:
            print(f"gh CLI download failed: {e}")
    
    # Fallback: look for artifact in current directory
    # (In GitHub Actions, artifacts are downloaded via actions/download-artifact)
    print("Note: Use actions/download-artifact in workflow for automatic download")
    return False


def upload_artifact(artifact_name: str, files: List[str]) -> bool:
    """Upload files as artifact (for use in GitHub Actions)"""
    
    print(f"Artifact '{artifact_name}' would contain:")
    for f in files:
        p = Path(f)
        if p.exists():
            size = p.stat().st_size
            print(f"  {f} ({size:,} bytes)")
    
    print("\nIn GitHub Actions, use:")
    print(f"  - uses: actions/upload-artifact@v4")
    print(f"    with:")
    print(f"      name: {artifact_name}")
    print(f"      path: {'\n'.join(files)}")
    
    return True


def which(cmd: str) -> Optional[str]:
    """Find command in PATH"""
    for path in os.environ.get("PATH", "").split(os.pathsep):
        full = os.path.join(path, cmd)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Checkpoint Manager for Micro-Training'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Download command
    dl_parser = subparsers.add_parser('download', help='Download latest checkpoint')
    dl_parser.add_argument('--artifact-prefix', default='checkpoint',
                           help='Artifact name prefix')
    dl_parser.add_argument('--output-dir', default='checkpoints',
                           help='Output directory')
    
    # Upload command
    up_parser = subparsers.add_parser('upload', help='Prepare upload manifest')
    up_parser.add_argument('--checkpoint-file', required=True,
                           help='Checkpoint file to upload')
    up_parser.add_argument('--artifact-name',
                           help='Artifact name (auto-generated if not set)')
    up_parser.add_argument('--epoch', type=int, default=0,
                           help='Training epoch')
    up_parser.add_argument('--loss', type=float, default=0.0,
                           help='Training loss')
    up_parser.add_argument('--val-loss', type=float, default=0.0,
                           help='Validation loss')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show training status')
    status_parser.add_argument('--target-epochs', type=int, default=100,
                               help='Target total epochs')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove old checkpoints')
    cleanup_parser.add_argument('--keep', type=int, default=5,
                                help='Number of checkpoints to keep')
    
    # Init command (for first run)
    init_parser = subparsers.add_parser('init', help='Initialize checkpoint directory')
    init_parser.add_argument('--checkpoint-dir', default='checkpoints',
                             help='Checkpoint directory')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        # In GitHub Actions, artifacts are downloaded via actions/download-artifact
        # This is a helper for local use
        success = download_artifact(args.artifact_prefix, args.output_dir)
        sys.exit(0 if success else 1)
    
    elif args.command == 'upload':
        manager = CheckpointManager()
        
        artifact_name = args.artifact_name or f"checkpoint-epoch{args.epoch}-{int(time.time())}"
        
        # Save checkpoint with metadata
        manager.save_checkpoint(
            args.checkpoint_file,
            args.epoch,
            args.loss,
            args.val_loss,
            run_id=os.environ.get('GITHUB_RUN_ID', 'local')
        )
        
        # Show upload manifest
        upload_artifact(artifact_name, [args.checkpoint_file])
    
    elif args.command == 'status':
        manager = CheckpointManager()
        state = manager.get_training_state()
        progress = manager.estimate_progress(args.target_epochs)
        
        print(f"\n{'='*50}")
        print("Training Status")
        print(f"{'='*50}")
        
        if state["resume_from"]:
            print(f"Latest checkpoint: {state['resume_from']}")
            print(f"  Epoch: {state['start_epoch'] - 1}")
            print(f"  Loss: {state['last_loss']:.6f}")
            print(f"  Val Loss: {state['last_val_loss']:.6f}")
            print(f"  Total epochs completed: {state['total_epochs']}")
        else:
            print("No checkpoint found - starting from scratch")
        
        print(f"\nProgress: {progress['progress_percent']:.1f}%")
        print(f"  {progress['completed_epochs']}/{progress['target_epochs']} epochs")
        print(f"  Remaining: {progress['remaining_epochs']} epochs")
        
        # Estimate time (assuming ~30 min per epoch on GitHub CPU)
        minutes_per_epoch = 30
        remaining_minutes = progress['remaining_epochs'] * minutes_per_epoch
        remaining_hours = remaining_minutes / 60
        
        print(f"\nEstimated remaining: ~{remaining_hours:.1f} hours")
        print(f"  ({progress['remaining_epochs']} cycles on GitHub Actions)")
        
        print(f"{'='*50}\n")
    
    elif args.command == 'cleanup':
        manager = CheckpointManager()
        removed = manager.cleanup_old_checkpoints(args.keep)
        print(f"Removed {removed} old checkpoints")
    
    elif args.command == 'init':
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Create initial metadata
        metadata = {
            "checkpoints": [],
            "latest": None,
            "total_epochs": 0,
            "initialized": int(time.time()),
        }
        
        with open(Path(args.checkpoint_dir) / "checkpoint_history.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Initialized checkpoint directory: {args.checkpoint_dir}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
