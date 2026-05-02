#!/usr/bin/env python3
"""
Neural Move Ordering Policy Network for Nexus Infinite.

Predicts move probabilities from position features.
Used for:
  - Root move ordering (reduce tree size)
  - LMR guidance (reduce less promising moves more)
  - Search pruning (skip moves with p < threshold)

Architecture: Tiny transformer/CNN over board state.
Input: 8x8xN channels (piece types, attack maps, etc.)
Output: move probabilities over all legal moves.

Training data: search tree trajectories with proven best moves.
"""

import argparse
import os
import sys
import json
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch required. pip install torch")
    sys.exit(1)


# ======== Board Encoding for Policy Network ========

def fen_to_policy_input(fen: str) -> np.ndarray:
    """Convert FEN to 8x8x20 input tensor for policy network.
    
    Channels:
      0-5:  White pieces (P,N,B,R,Q,K) one-hot
      6-11: Black pieces (P,N,B,R,Q,K) one-hot
      12:    White attack map
      13:    Black attack map  
      14:    Side to move (1=white, 0=black)
      15:    Castling rights
      16:    En passant target
      17:    Check status
      18:    Material count (normalized)
      19:    Phase (opening/middlegame/endgame)
    """
    parts = fen.split()
    board = parts[0]
    side = 1 if parts[1] == 'w' else 0
    
    # Initialize channels
    channels = np.zeros((20, 8, 8), dtype=np.float32)
    
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                 'p':0,'n':1,'b':2,'r':3,'q':4,'k':5}
    
    sq = 56  # a8
    for char in board:
        if char == '/':
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        else:
            r, f = sq // 8, sq % 8
            if char.isupper():
                channels[piece_map[char], 7-r, f] = 1.0
            else:
                channels[6 + piece_map[char], 7-r, f] = 1.0
            sq += 1
    
    # Attack maps (simplified: count attackers per square)
    # Full impl would use actual attack generation
    for r in range(8):
        for f in range(8):
            # Placeholder - real impl needs move generation
            channels[12, r, f] = 0.0  # white attacks
            channels[13, r, f] = 0.0  # black attacks
    
    # Global features broadcast to all squares
    channels[14] = side  # side to move
    
    # Castling
    castling = parts[2] if len(parts) > 2 else '-'
    channels[15] = (1.0 if 'K' in castling else 0.0 +
                    1.0 if 'Q' in castling else 0.0 +
                    1.0 if 'k' in castling else 0.0 +
                    1.0 if 'q' in castling else 0.0) / 4.0
    
    # Check (simplified)
    channels[17] = 0.0  # Would need actual check detection
    
    # Material (simplified count)
    material = np.sum(channels[:12])
    channels[18] = material / 32.0
    
    # Phase
    total_pieces = np.sum(channels[:12])
    channels[19] = 1.0 - (total_pieces / 32.0)  # 1=endgame, 0=opening
    
    return channels


def encode_move(from_sq: int, to_sq: int, promotion: int = 0) -> int:
    """Encode move as flat index: 64*64 + 4 promotions = 4096+?"""
    # Simplified: from*64 + to = 4096 moves
    # Promotion: add 64*64 + piece_type*64 + to
    return from_sq * 64 + to_sq


def decode_move(idx: int) -> Tuple[int, int, int]:
    """Decode flat index to (from_sq, to_sq, promotion)"""
    if idx < 4096:
        return idx // 64, idx % 64, 0
    # Promotion moves
    promo_idx = idx - 4096
    piece = promo_idx // 64
    to_sq = promo_idx % 64
    return 0, to_sq, piece  # from_sq needs context


# ======== Policy Network Model ========

class PolicyConvBlock(nn.Module):
    """Residual convolution block for spatial feature extraction."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
        
    def forward(self, x):
        residual = self.skip(x) if self.skip else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class NexusPolicyNetwork(nn.Module):
    """Tiny policy network for move ordering.
    
    Architecture (inspired by AlphaZero but tiny for CPU):
      Input:  20 channels x 8x8
      Conv:   32 -> 64 -> 64 (3x3 residual blocks)
      Policy head: flatten -> 512 -> 4672 moves (64*64 + promotions)
      Value head:  flatten -> 256 -> 1 (for training signal)
    """
    
    NUM_MOVES = 64 * 64  # Simplified: all square-to-square moves
    
    def __init__(self, input_channels: int = 20, 
                 conv_channels: int = 64,
                 num_residual: int = 4):
        super().__init__()
        
        # Initial conv
        self.input_conv = nn.Conv2d(input_channels, conv_channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(conv_channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            PolicyConvBlock(conv_channels, conv_channels)
            for _ in range(num_residual)
        ])
        
        # Policy head: spatial -> moves
        self.policy_conv = nn.Conv2d(conv_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, self.NUM_MOVES)
        
        # Value head (auxiliary training signal)
        self.value_conv = nn.Conv2d(conv_channels, 16, 1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 20, 8, 8] board state
        Returns:
            policy_logits: [batch, NUM_MOVES] move probabilities (before softmax)
            value: [batch, 1] position value estimate
        """
        # Feature extraction
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def get_move_probs(self, board_input: np.ndarray, 
                       legal_moves: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], float]]:
        """Get move probabilities for legal moves only.
        
        Args:
            board_input: [20, 8, 8] numpy array
            legal_moves: list of (from_sq, to_sq) tuples
        Returns:
            list of ((from_sq, to_sq), probability) sorted by prob desc
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(board_input).unsqueeze(0)
            logits, _ = self.forward(x)
            
            # Get indices for legal moves
            legal_indices = [f * 64 + t for f, t in legal_moves]
            legal_logits = logits[0, legal_indices]
            
            # Softmax over legal moves only
            probs = F.softmax(legal_logits, dim=0).numpy()
            
            return sorted(
                [(move, float(p)) for move, p in zip(legal_moves, probs)],
                key=lambda x: x[1],
                reverse=True
            )
    
    def save_quantized(self, filepath: str):
        """Save INT8 quantized weights for engine inference."""
        state = {
            'input_conv_weight': self.input_conv.weight.data.cpu().numpy(),
            'input_conv_bias': self.input_conv.bias.data.cpu().numpy(),
            'policy_fc_weight': self.policy_fc.weight.data.cpu().numpy(),
            'policy_fc_bias': self.policy_fc.bias.data.cpu().numpy(),
            'value_fc2_weight': self.value_fc2.weight.data.cpu().numpy(),
            'value_fc2_bias': self.value_fc2.bias.data.cpu().numpy(),
        }
        
        # Quantize conv weights to INT8
        quantized = {}
        for name, arr in state.items():
            if 'weight' in name:
                max_val = np.max(np.abs(arr))
                scale = max_val / 127.0 if max_val > 0 else 1.0
                q_arr = np.round(arr / scale).astype(np.int8)
                quantized[name] = q_arr
                quantized[name + '_scale'] = np.float32(scale)
            else:
                quantized[name] = arr.astype(np.float32)
        
        np.savez_compressed(filepath, **quantized)
        print(f"Saved quantized policy net: {filepath}")


# ======== Dataset ========

class PolicyDataset(Dataset):
    """Dataset of (position, best_move, search_value) for policy training."""
    
    def __init__(self, data_file: str, max_positions: int = 0):
        self.positions = []
        
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if max_positions > 0 and i >= max_positions:
                    break
                
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) < 3:
                    continue
                
                fen = parts[0].strip()
                best_move_str = parts[1].strip()
                value = float(parts[2].strip())
                
                # Parse best move (e.g., "e2e4" or "g1f3")
                if len(best_move_str) >= 4:
                    # Simplified UCI parse
                    files = 'abcdefgh'
                    ranks = '12345678'
                    try:
                        from_f = files.index(best_move_str[0])
                        from_r = ranks.index(best_move_str[1])
                        to_f = files.index(best_move_str[2])
                        to_r = ranks.index(best_move_str[3])
                        from_sq = from_r * 8 + from_f
                        to_sq = to_r * 8 + to_f
                        
                        board_input = fen_to_policy_input(fen)
                        move_idx = from_sq * 64 + to_sq
                        
                        self.positions.append((board_input, move_idx, value))
                    except (ValueError, IndexError):
                        continue
        
        print(f"Loaded {len(self.positions)} policy training positions")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        board, move_idx, value = self.positions[idx]
        return (torch.from_numpy(board),
                torch.tensor(move_idx, dtype=torch.long),
                torch.tensor(value, dtype=torch.float32))


# ======== Training ========

class PolicyTrainer:
    def __init__(self, model: NexusPolicyNetwork, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
    def train_epoch(self, loader: DataLoader, optimizer) -> Tuple[float, float]:
        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for boards, moves, values in loader:
            boards = boards.to(self.device)
            moves = moves.to(self.device)
            values = values.to(self.device)
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = self.model(boards)
            
            policy_loss = self.policy_criterion(policy_logits, moves)
            value_loss = self.value_criterion(value_pred.squeeze(), values)
            
            # Combined loss: policy dominates (90%), value guides (10%)
            loss = 0.9 * policy_loss + 0.1 * value_loss
            
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        n = len(loader)
        return total_policy_loss / n, total_value_loss / n
    
    def validate(self, loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for boards, moves, values in loader:
                boards = boards.to(self.device)
                moves = moves.to(self.device)
                values = values.to(self.device)
                
                policy_logits, value_pred = self.model(boards)
                
                policy_loss = self.policy_criterion(policy_logits, moves)
                value_loss = self.value_criterion(value_pred.squeeze(), values)
                loss = 0.9 * policy_loss + 0.1 * value_loss
                
                total_loss += loss.item()
                
                # Top-1 accuracy
                pred_moves = policy_logits.argmax(dim=1)
                correct += (pred_moves == moves).sum().item()
                total += moves.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return total_loss / len(loader), accuracy, correct


def main():
    parser = argparse.ArgumentParser(description='Policy Network Trainer')
    parser.add_argument('--data', required=True, help='Training data file')
    parser.add_argument('--output', default='policy_net.pt', help='Output model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max-positions', type=int, default=0)
    parser.add_argument('--val-split', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Load data
    dataset = PolicyDataset(args.data, args.max_positions)
    
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_size > 0 else None
    
    # Model
    model = NexusPolicyNetwork(conv_channels=32, num_residual=2)  # Tiny for CPU
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = PolicyTrainer(model, device='cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        p_loss, v_loss = trainer.train_epoch(train_loader, optimizer)
        
        if val_loader:
            val_loss, acc, correct = trainer.validate(val_loader)
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}: P-loss={p_loss:.4f} V-loss={v_loss:.4f} "
                  f"Val-loss={val_loss:.4f} Acc={acc:.3f} ({correct}/{val_size})")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.output)
                print(f"  -> Saved best model (acc={acc:.3f})")
        else:
            print(f"Epoch {epoch+1}: P-loss={p_loss:.4f} V-loss={v_loss:.4f}")
    
    # Export quantized
    model.load_state_dict(torch.load(args.output))
    model.save_quantized(args.output.replace('.pt', '_quantized.npz'))
    print(f"Training complete. Best accuracy: {best_acc:.3f}")


if __name__ == '__main__':
    main()
