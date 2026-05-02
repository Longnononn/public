#!/usr/bin/env python3
"""
Auto Promotion Script for Nexus Infinite
Promotes candidate NNUE model to 'best' if SPRT passes.
Manages model archive and versioning.

Usage:
    python scripts/promote.py \
        --candidate models/candidate.nnue \
        --best models/best.nnue \
        --backup-dir models/archive \
        --elo-gain 5.2 \
        --git-tag v0.4.5 \
        --yes
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def compute_hash(filepath: str) -> str:
    """Compute SHA256 hash of file"""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def get_model_info(filepath: str) -> dict:
    """Get model metadata"""
    path = Path(filepath)
    
    info = {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "hash": compute_hash(filepath),
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    }
    
    return info


def load_elo_history(path: str = "elo_history.json") -> list:
    """Load Elo history from JSON"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []


def save_elo_history(history: list, path: str = "elo_history.json"):
    """Save Elo history to JSON"""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def promote(candidate: str, best: str, backup_dir: Optional[str] = None,
            elo_gain: float = 0.0, tag: Optional[str] = None,
            force: bool = False) -> bool:
    """Promote candidate model to best"""
    
    candidate_path = Path(candidate)
    best_path = Path(best)
    
    # Validate candidate exists
    if not candidate_path.exists():
        print(f"ERROR: Candidate model not found: {candidate}")
        return False
    
    # Get model info
    candidate_info = get_model_info(candidate)
    print(f"Candidate: {candidate_info['filename']}")
    print(f"  Size: {candidate_info['size_mb']} MB")
    print(f"  Hash: {candidate_info['hash']}")
    
    # Backup current best if exists
    if best_path.exists() and not force:
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Generate backup name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"nnue_{timestamp}_{compute_hash(str(best_path))[:8]}.nnue"
            backup_file = backup_path / backup_name
            
            shutil.copy2(best_path, backup_file)
            print(f"  Backed up old best -> {backup_file}")
    
    # Promote candidate to best
    shutil.copy2(candidate_path, best_path)
    print(f"\nPromoted candidate -> {best_path}")
    
    # Update Elo history
    if elo_gain != 0.0 or tag:
        history = load_elo_history()
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "promote",
            "version": tag or "unknown",
            "elo_gain": elo_gain,
            "candidate": candidate_info,
        }
        
        if best_path.exists():
            entry["previous_best"] = get_model_info(str(best_path))
        
        history.append(entry)
        save_elo_history(history)
        print(f"  Elo history updated ({len(history)} entries)")
    
    # Create symlink if on Unix
    if os.name != 'nt':  # Not Windows
        latest_link = best_path.parent / "latest.nnue"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(best_path.name)
        print(f"  Updated symlink: {latest_link}")
    
    return True


def list_models(directory: str = "models"):
    """List all models in directory"""
    models_dir = Path(directory)
    
    if not models_dir.exists():
        print(f"Models directory not found: {directory}")
        return
    
    print(f"\n{'='*60}")
    print(f"Models in {directory}")
    print(f"{'='*60}")
    
    models = sorted(models_dir.glob("*.nnue"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    for i, model in enumerate(models, 1):
        info = get_model_info(str(model))
        marker = " <- BEST" if model.name == "best.nnue" else ""
        print(f"\n{i}. {model.name}{marker}")
        print(f"   Size: {info['size_mb']} MB")
        print(f"   Hash: {info['hash']}")
        print(f"   Modified: {info['modified']}")
    
    if not models:
        print("  No models found")
    
    print(f"{'='*60}")


def rollback(backup_dir: str = "models/archive", target: str = "models/best.nnue"):
    """Rollback to previous model"""
    archive = Path(backup_dir)
    
    if not archive.exists():
        print(f"Archive directory not found: {backup_dir}")
        return False
    
    # Get most recent backup
    backups = sorted(archive.glob("*.nnue"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not backups:
        print("No backups found in archive")
        return False
    
    print("Available backups:")
    for i, backup in enumerate(backups[:5], 1):
        info = get_model_info(str(backup))
        print(f"  {i}. {backup.name} ({info['size_mb']} MB, {info['modified'][:10]})")
    
    choice = input("\nSelect backup to restore (1-5, or 'q' to cancel): ").strip()
    
    if choice.lower() == 'q':
        print("Cancelled")
        return False
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(backups):
            print("Invalid selection")
            return False
    except ValueError:
        print("Invalid input")
        return False
    
    # Restore
    selected = backups[idx]
    target_path = Path(target)
    
    # Backup current
    if target_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"nnue_{timestamp}_before_rollback.nnue"
        shutil.copy2(target_path, archive / backup_name)
    
    shutil.copy2(selected, target_path)
    print(f"\nRestored: {selected.name} -> {target_path}")
    
    # Update history
    history = load_elo_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "event": "rollback",
        "restored_from": str(selected),
        "target": str(target_path),
    })
    save_elo_history(history)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Auto Promotion for Nexus Infinite NNUE Models'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote candidate to best')
    promote_parser.add_argument('--candidate', required=True, help='Candidate model path')
    promote_parser.add_argument('--best', default='models/best.nnue', help='Best model path')
    promote_parser.add_argument('--backup-dir', default='models/archive', help='Backup directory')
    promote_parser.add_argument('--elo-gain', type=float, default=0.0, help='Measured Elo gain')
    promote_parser.add_argument('--git-tag', help='Git version tag')
    promote_parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    promote_parser.add_argument('--force', action='store_true', help='Force without backup')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all models')
    list_parser.add_argument('--dir', default='models', help='Models directory')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous model')
    rollback_parser.add_argument('--backup-dir', default='models/archive', help='Backup directory')
    rollback_parser.add_argument('--target', default='models/best.nnue', help='Target model path')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show promotion history')
    
    args = parser.parse_args()
    
    if args.command == 'promote':
        print("=" * 60)
        print("Nexus Infinite - Model Promotion")
        print("=" * 60)
        
        if not args.yes:
            confirm = input(f"\nPromote {args.candidate} to {args.best}? [y/N]: ")
            if confirm.lower() != 'y':
                print("Cancelled")
                sys.exit(0)
        
        success = promote(
            candidate=args.candidate,
            best=args.best,
            backup_dir=args.backup_dir,
            elo_gain=args.elo_gain,
            tag=args.git_tag,
            force=args.force
        )
        
        if success:
            print("\nPromotion successful!")
        else:
            print("\nPromotion failed!")
            sys.exit(1)
    
    elif args.command == 'list':
        list_models(args.dir)
    
    elif args.command == 'rollback':
        print("=" * 60)
        print("Nexus Infinite - Model Rollback")
        print("=" * 60)
        rollback(args.backup_dir, args.target)
    
    elif args.command == 'history':
        history = load_elo_history()
        
        print(f"\n{'='*60}")
        print("Promotion History")
        print(f"{'='*60}")
        
        if not history:
            print("  No history found")
        else:
            for i, entry in enumerate(history[-20:], 1):  # Last 20
                ts = entry.get('timestamp', '?')[:19]
                event = entry.get('event', '?')
                version = entry.get('version', 'N/A')
                elo = entry.get('elo_gain', 0)
                
                if event == 'promote':
                    print(f"\n{i}. {ts} - PROMOTE v{version} (+{elo:.1f} Elo)")
                    if 'candidate' in entry:
                        c = entry['candidate']
                        print(f"   Model: {c['filename']} ({c['size_mb']} MB)")
                elif event == 'rollback':
                    print(f"\n{i}. {ts} - ROLLBACK")
                    print(f"   Restored: {entry.get('restored_from', '?')}")
                else:
                    print(f"\n{i}. {ts} - {event}")
        
        print(f"\n{'='*60}")
        print(f"Total events: {len(history)}")
        print(f"{'='*60}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
