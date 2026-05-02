#!/usr/bin/env python3
"""
Learned Pruning Network for Nexus Infinite.

Trains a small neural network to predict whether a node should be pruned.
Replaces handcrafted margins (futility, null move, LMR) with data-driven decisions.

Features:
  - Node depth remaining
  - Alpha-beta window size
  - Static evaluation
  - Move history score
  - SEE value
  - Material difference
  - King safety
  - Pawn structure
  - Phase (opening/middlegame/endgame)
  - TT hit rate in subtree
  - Prior move was check/capture/promotion
  
Output: prune_probability [0, 1]

Training data: Search telemetry logs with decisions and outcomes.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch required")
    sys.exit(1)


class PruningFeatures:
    """Feature vector for pruning decision."""
    
    FEATURE_NAMES = [
        'depth_remaining',      # 0: How many plies left
        'alpha_beta_gap',       # 1: beta - alpha
        'static_eval',          # 2: Position evaluation
        'eval_margin',          # 3: eval - alpha
        'move_history_score',   # 4: History heuristic value
        'move_counter_score',   # 5: Countermove heuristic
        'see_score',            # 6: Static exchange evaluation
        'material_diff',        # 7: Material imbalance
        'king_safety',          # 8: King safety estimate
        'pawn_structure',       # 9: Pawn structure score
        'piece_activity',       # 10: Piece mobility/activity
        'phase',                # 11: 0=opening, 1=endgame
        'is_check',             # 12: Position is check
        'is_capture',           # 13: Move is capture
        'is_promotion',         # 14: Move is promotion
        'tt_hit_rate',          # 15: TT hits in recent nodes
        'fail_high_ratio',      # 16: Recent fail-high rate
        'node_count_subtree',   # 17: Estimated subtree size
        'null_move_verified',   # 18: Null move search result
        'first_move',           # 19: Is first move in list
    ]
    
    NUM_FEATURES = len(FEATURE_NAMES)
    
    @classmethod
    def from_search_telemetry(cls, entry: Dict) -> np.ndarray:
        """Extract features from a telemetry log entry."""
        f = np.zeros(cls.NUM_FEATURES, dtype=np.float32)
        
        # Parse context JSON if available
        ctx = {}
        if 'context' in entry:
            try:
                ctx = json.loads(entry['context'])
            except:
                pass
        
        f[0] = entry.get('depth', 0) / 30.0  # Normalize
        f[1] = (entry.get('beta', 0) - entry.get('alpha', 0)) / 2000.0
        f[2] = entry.get('score', 0) / 1000.0
        f[3] = (entry.get('score', 0) - entry.get('alpha', 0)) / 1000.0
        f[4] = ctx.get('history_score', 0) / 10000.0
        f[5] = ctx.get('counter_score', 0) / 10000.0
        f[6] = ctx.get('see', 0) / 1000.0
        f[7] = ctx.get('material', 0) / 10.0
        f[8] = ctx.get('king_safety', 0) / 500.0
        f[9] = ctx.get('pawn_structure', 0) / 200.0
        f[10] = ctx.get('activity', 0) / 500.0
        f[11] = ctx.get('phase', 0.5)
        f[12] = 1.0 if ctx.get('is_check', False) else 0.0
        f[13] = 1.0 if ctx.get('is_capture', False) else 0.0
        f[14] = 1.0 if ctx.get('is_promotion', False) else 0.0
        f[15] = ctx.get('tt_rate', 0.5)
        f[16] = ctx.get('fail_high', 0.3)
        f[17] = np.log1p(ctx.get('node_estimate', 1000)) / 15.0
        f[18] = 1.0 if ctx.get('null_verified', False) else 0.0
        f[19] = 1.0 if ctx.get('move_index', 0) == 0 else 0.0
        
        return f


class PruningDataset(Dataset):
    """Dataset of (features, should_prune, actual_outcome)."""
    
    def __init__(self, telemetry_file: str, max_samples: int = 0):
        self.samples = []
        
        with open(telemetry_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples > 0 and i >= max_samples:
                    break
                
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    entry = json.loads(line)
                except:
                    continue
                
                # Only PRUNING_DECISION and CUTOFF events
                event_type = entry.get('type', '')
                if event_type not in ('PRUNING_DECISION', 'CUTOFF', 'NODE_EXPANDED'):
                    continue
                
                features = PruningFeatures.from_search_telemetry(entry)
                
                # Label: 1 = prune was correct (didn't miss better move)
                #       0 = prune was wrong (missed better move)
                #       We determine this from search outcome
                if event_type == 'CUTOFF':
                    label = 1.0  # Cutoff succeeded
                elif event_type == 'PRUNING_DECISION':
                    accepted = entry.get('accepted', True)
                    # If accepted and no fail-high later, it's correct
                    # Simplified: use accepted as proxy
                    label = 1.0 if accepted else 0.0
                else:
                    label = 0.5  # Unclear
                
                # Weight: wrong pruning decisions are more important
                weight = 2.0 if label < 0.5 else 1.0
                
                self.samples.append((features, label, weight))
        
        print(f"Loaded {len(self.samples)} pruning samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, label, weight = self.samples[idx]
        return (torch.from_numpy(features),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32))


class LearnedPruningNet(nn.Module):
    """Small MLP for pruning decisions.
    
    Input: 20 features
    Hidden: 64 -> 32
    Output: prune_probability + value_estimate
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()
    
    def should_prune(self, features: np.ndarray, threshold: float = 0.7) -> Tuple[bool, float]:
        """Returns (should_prune, confidence)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).float()
            prob = self.forward(x).item()
            return prob > threshold, prob
    
    def get_margin(self, features: np.ndarray) -> float:
        """Get dynamic margin for pruning (replaces static futility margin)."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).float()
            prob = self.forward(x).item()
            # Higher prob = more confident to prune = larger margin
            return 100.0 + prob * 400.0  # 100-500 cp dynamic margin


class PruningTrainer:
    """Trainer with weighted loss (penalize false negatives more)."""
    
    def __init__(self, model: LearnedPruningNet, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss(reduction='none')
        
    def train_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        total_weight = 0.0
        
        for features, labels, weights in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)
            
            optimizer.zero_grad()
            
            preds = self.model(features)
            loss = self.criterion(preds, labels)
            loss = (loss * weights).mean()  # Weighted loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            total_weight += features.size(0)
        
        return total_loss / total_weight if total_weight > 0 else 0.0
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        correct = 0
        total = 0
        false_negatives = 0  # Shouldn't prune but did
        false_positives = 0  # Should prune but didn't
        
        with torch.no_grad():
            for features, labels, _ in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                preds = self.model(features) > 0.5
                correct += (preds == (labels > 0.5)).sum().item()
                total += labels.size(0)
                
                false_negatives += ((preds == 1) & (labels == 0)).sum().item()
                false_positives += ((preds == 0) & (labels == 1)).sum().item()
        
        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'false_negative_rate': false_negatives / total if total > 0 else 0.0,
            'false_positive_rate': false_positives / total if total > 0 else 0.0,
        }


def generate_telemetry_summary(telemetry_file: str, output_file: str):
    """Analyze telemetry to find pruning patterns."""
    stats = {
        'total_prunes': 0,
        'successful_prunes': 0,
        'failed_prunes': 0,
        'by_depth': {},
        'by_phase': {},
    }
    
    with open(telemetry_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except:
                continue
            
            if entry.get('type') != 'PRUNING_DECISION':
                continue
            
            depth = entry.get('depth', 0)
            accepted = entry.get('accepted', False)
            
            stats['total_prunes'] += 1
            if accepted:
                stats['successful_prunes'] += 1
            else:
                stats['failed_prunes'] += 1
            
            d_key = f"d{depth}"
            if d_key not in stats['by_depth']:
                stats['by_depth'][d_key] = {'success': 0, 'fail': 0}
            if accepted:
                stats['by_depth'][d_key]['success'] += 1
            else:
                stats['by_depth'][d_key]['fail'] += 1
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Pruning summary written to {output_file}")
    print(f"Total: {stats['total_prunes']}, "
          f"Success: {stats['successful_prunes']}, "
          f"Fail: {stats['failed_prunes']}")


def export_cpp_header(model: LearnedPruningNet, output_file: str):
    """Export model weights as C++ header for engine integration."""
    state = model.state_dict()
    
    lines = [
        "#pragma once",
        "// Auto-generated learned pruning weights",
        "",
        f"constexpr int PRUNING_INPUT_DIM = {PruningFeatures.NUM_FEATURES};",
        "",
    ]
    
    for name, param in state.items():
        if 'weight' in name:
            arr = param.cpu().numpy().flatten()
            shape = list(param.shape)
            lines.append(f"// {name}: {shape}")
            lines.append(f"constexpr float {name}_data[] = {{")
            
            # Format in rows of 8
            for i in range(0, len(arr), 8):
                row = arr[i:i+8]
                vals = ", ".join(f"{v:.6f}f" for v in row)
                lines.append(f"    {vals},")
            
            lines.append("};")
            lines.append("")
    
    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"C++ header exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Learned Pruning Trainer')
    parser.add_argument('--telemetry', required=True, help='Search telemetry JSONL file')
    parser.add_argument('--output', default='pruning_net.pt', help='Output model')
    parser.add_argument('--cpp-header', help='Export C++ header')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--summary', action='store_true', help='Generate pruning summary')
    
    args = parser.parse_args()
    
    if args.summary:
        generate_telemetry_summary(args.telemetry, 'pruning_summary.json')
        return
    
    # Load data
    dataset = PruningDataset(args.telemetry, args.max_samples)
    
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_size > 0 else None
    
    # Model
    model = LearnedPruningNet(input_dim=PruningFeatures.NUM_FEATURES, hidden_dim=args.hidden)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = PruningTrainer(model, device='cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(train_loader, optimizer)
        
        if val_loader:
            metrics = trainer.validate(val_loader)
            scheduler.step(1.0 - metrics['accuracy'])
            
            print(f"Epoch {epoch+1}: loss={loss:.4f} "
                  f"acc={metrics['accuracy']:.3f} "
                  f"FN={metrics['false_negative_rate']:.3f} "
                  f"FP={metrics['false_positive_rate']:.3f}")
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                torch.save(model.state_dict(), args.output)
                print(f"  -> Saved best model")
        else:
            print(f"Epoch {epoch+1}: loss={loss:.4f}")
    
    # Export
    if args.cpp_header:
        model.load_state_dict(torch.load(args.output))
        export_cpp_header(model, args.cpp_header)
    
    print(f"Training complete. Best accuracy: {best_acc:.3f}")


if __name__ == '__main__':
    main()
