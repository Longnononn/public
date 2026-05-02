#!/usr/bin/env python3
"""
NNUE Training Pipeline for Nexus Infinite
PyTorch-based trainer with Stockfish-compatible architecture

Architecture:
- Input: HalfKP (64 king squares × 640 piece positions)
- Hidden: 256 neurons (configurable)
- Output: 1 scalar (centipawn score)
- Activation: Clipped ReLU (max=1.0)
- Quantization: INT8 for inference

Usage:
    python scripts/train_nnue.py \
        --data training_data.txt \
        --output nexus.nnue \
        --epochs 100 \
        --batch-size 8192 \
        --lr 0.001
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Install: pip install torch torchvision tensorboard")
    sys.exit(1)


# ============== Architecture Constants ==============

NUM_SQ = 64
NUM_PT = 10  # 6 pieces × 2 colors - 1 (no king)
NUM_KING_SQ = 64
INPUT_SIZE = NUM_KING_SQ * NUM_PT * (NUM_SQ - 1)  # ~410k
SPACE_CONTROL_SIZE = 128  # 64 squares x 2 colors (heatmap)
HIDDEN_SIZE = 128  # Tiny NNUE for CPU training (configurable)
OUTPUT_BUCKETS = 8  # Phase-aware

# Attack directions for space control
KNIGHT_DIRS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
KING_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
BISHOP_DIRS = [(-1,-1),(-1,1),(1,-1),(1,1)]
ROOK_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS

# Feature transformation
KING_BUCKETS = [
    [0, 1, 2, 3, 3, 2, 1, 0],  # Rank-based buckets
    [4, 5, 6, 7, 7, 6, 5, 4],
    [8, 9, 10, 11, 11, 10, 9, 8],
    [8, 9, 10, 11, 11, 10, 9, 8],
    [8, 9, 10, 11, 11, 10, 9, 8],
    [8, 9, 10, 11, 11, 10, 9, 8],
    [12, 13, 14, 15, 15, 14, 13, 12],
    [12, 13, 14, 15, 15, 14, 13, 12],
]


def square_to_index(sq: int) -> int:
    """Convert square (0-63) to bucket index"""
    rank = sq // 8
    file = sq % 8
    return KING_BUCKETS[rank][file]


def piece_to_index(piece: str, color: str) -> int:
    """Map piece char to index (0-9, excluding king)"""
    pieces = 'PNBRQKpnbrqk'
    idx = pieces.index(piece)
    # White: 0-5, Black: 6-11
    # Remove king indices (5, 11)
    if idx == 5:  # White king
        return -1
    if idx == 11:  # Black king
        return -1
    # Adjust after removing kings
    if idx < 5:
        return idx
    return idx - 1  # Skip black king


def compute_space_control(board_str: str, side_to_move: str) -> np.ndarray:
    """Compute space control heatmap for both sides.
    
    Returns: (128,) vector = [64 white_attack_counts + 64 black_attack_counts]
    normalized by max attacks per square.
    """
    # Parse board to piece placement
    pieces = {}  # sq -> (color, piece_type)
    sq = 56  # a8
    for char in board_str:
        if char == '/':
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        else:
            color = 'w' if char.isupper() else 'b'
            piece = char.upper()
            pieces[sq] = (color, piece)
            sq += 1
    
    white_ctrl = np.zeros(64, dtype=np.float32)
    black_ctrl = np.zeros(64, dtype=np.float32)
    
    def file_rank(s):
        return s % 8, s // 8
    
    def add_attacks(from_sq, dirs, max_dist=1, sliding=False):
        f, r = file_rank(from_sq)
        color = pieces[from_sq][0]
        target = white_ctrl if color == 'w' else black_ctrl
        for df, dr in dirs:
            for dist in range(1, max_dist + 1):
                nf, nr = f + df * dist, r + dr * dist
                if not (0 <= nf <= 7 and 0 <= nr <= 7):
                    break
                to_sq = nr * 8 + nf
                target[to_sq] += 1.0
                if to_sq in pieces and not sliding:
                    break
                if to_sq in pieces and sliding:
                    break
    
    for sq, (color, piece) in pieces.items():
        if piece == 'N':
            add_attacks(sq, KNIGHT_DIRS, max_dist=1)
        elif piece == 'K':
            add_attacks(sq, KING_DIRS, max_dist=1)
        elif piece == 'P':
            # Pawns attack diagonally forward
            f, r = file_rank(sq)
            dr = 1 if color == 'w' else -1
            target = white_ctrl if color == 'w' else black_ctrl
            for df in (-1, 1):
                nf = f + df
                nr = r + dr
                if 0 <= nf <= 7 and 0 <= nr <= 7:
                    target[nr * 8 + nf] += 1.0
        elif piece == 'B':
            add_attacks(sq, BISHOP_DIRS, max_dist=7, sliding=True)
        elif piece == 'R':
            add_attacks(sq, ROOK_DIRS, max_dist=7, sliding=True)
        elif piece == 'Q':
            add_attacks(sq, QUEEN_DIRS, max_dist=7, sliding=True)
    
    # Normalize by typical max attacks per square (~8)
    white_ctrl = np.clip(white_ctrl / 8.0, 0.0, 1.0)
    black_ctrl = np.clip(black_ctrl / 8.0, 0.0, 1.0)
    
    # Flip if black to move (relative)
    if side_to_move == 'b':
        return np.concatenate([black_ctrl, white_ctrl])
    return np.concatenate([white_ctrl, black_ctrl])


# ============== Dataset ==============

class NexusDataset(Dataset):
    """Dataset for NNUE training from self-play data"""
    
    def __init__(self, data_file: str, max_positions: int = 0):
        self.positions: List[Tuple[np.ndarray, float]] = []
        self.load_data(data_file, max_positions)
    
    def load_data(self, filename: str, max_positions: int):
        """Load positions from training file
        
        Format: FEN | score | result | phase | game_id
        """
        print(f"Loading data from {filename}...")
        
        count = 0
        with open(filename, 'r') as f:
            for line in f:
                if count >= max_positions > 0:
                    break
                    
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) < 3:
                    continue
                
                fen = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                    result = int(parts[2].strip())
                except (ValueError, IndexError):
                    continue
                
                # Convert FEN to features
                features = self.fen_to_features(fen)
                if features is not None:
                    # Combine score and result for target
                    # WDL: blend search score with game outcome
                    target = self.compute_target(score, result)
                    self.positions.append((features, target))
                    count += 1
        
        print(f"Loaded {len(self.positions)} positions")
    
    def fen_to_features(self, fen: str) -> Optional[np.ndarray]:
        """Convert FEN to HalfKP feature vector"""
        # Parse FEN
        parts = fen.split()
        if len(parts) < 4:
            return None
        
        board = parts[0]
        side = parts[1]  # 'w' or 'b'
        
        # Initialize features
        # White perspective: [white_king_sq × piece_positions]
        # Black perspective: [black_king_sq × piece_positions]
        white_features = np.zeros((NUM_KING_SQ, NUM_PT * NUM_SQ), dtype=np.float32)
        black_features = np.zeros((NUM_KING_SQ, NUM_PT * NUM_SQ), dtype=np.float32)
        
        # Parse board
        sq = 56  # Start from a8
        white_king_sq = -1
        black_king_sq = -1
        
        for char in board:
            if char == '/':
                sq -= 16  # Move to next rank down
            elif char.isdigit():
                sq += int(char)
            else:
                piece_idx = piece_to_index(char, 'w' if char.isupper() else 'b')
                
                if piece_idx == -1:  # King
                    if char == 'K':
                        white_king_sq = sq
                    else:
                        black_king_sq = sq
                else:
                    # Add to both perspectives
                    # White's view: piece at sq, indexed by white king
                    if white_king_sq >= 0:
                        feat_idx = piece_idx * NUM_SQ + sq
                        white_features[white_king_sq, feat_idx] = 1.0
                    
                    # Black's view: mirrored square
                    mirrored_sq = sq ^ 56  # Flip rank
                    if black_king_sq >= 0:
                        # For black, flip piece color
                        piece_idx_flipped = piece_idx
                        if piece_idx < 5:  # White piece -> black piece
                            piece_idx_flipped += 5
                        else:  # Black piece -> white piece
                            piece_idx_flipped -= 5
                        feat_idx = piece_idx_flipped * NUM_SQ + mirrored_sq
                        black_features[black_king_sq, feat_idx] = 1.0
                
                sq += 1
        
        # Combine both perspectives
        halfkp_features = np.concatenate([
            white_features.flatten(),
            black_features.flatten()
        ])
        
        # Flip if black to move
        if side == 'b':
            halfkp_features = np.roll(halfkp_features, len(halfkp_features) // 2)
        
        # Global Board Vision: Space Control Heatmap
        space_control = compute_space_control(board, side)
        
        # Concatenate HalfKP + Space Control
        features = np.concatenate([halfkp_features, space_control])
        
        return features
    
    def compute_target(self, score: float, result: int) -> float:
        """Blend search score with game result
        
        result: -1=loss, 0=draw, 1=win
        """
        # Normalize score to [-1, 1]
        cp = score / 600.0  # Scale: 600cp = 1.0
        cp = np.clip(cp, -1.0, 1.0)
        
        # Blend with WDL
        wdl_score = result  # -1, 0, 1
        
        # Weight: 0.7 search, 0.3 WDL
        alpha = 0.7
        target = alpha * cp + (1 - alpha) * wdl_score
        
        return float(target)
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, target = self.positions[idx]
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.float32)


# ============== Model ==============

class NexusNNUE(nn.Module):
    """NNUE model with Space Control (Global Board Vision)
    
    Architecture: HalfKP (relative) + Space Control Heatmap -> hidden -> output
    """
    
    def __init__(self, input_size: int = INPUT_SIZE + SPACE_CONTROL_SIZE,
                 hidden_size: int = HIDDEN_SIZE,
                 space_control_size: int = SPACE_CONTROL_SIZE):
        super().__init__()
        
        self.input_size = input_size
        self.halfkp_size = input_size - space_control_size
        self.space_control_size = space_control_size
        self.hidden_size = hidden_size
        
        # Feature transformer for HalfKP (sparse in real engine)
        self.ft = nn.Linear(self.halfkp_size // 2, hidden_size, bias=True)
        
        # Space control projector: 128 -> 32 (lightweight)
        self.sc_proj = nn.Linear(space_control_size, 32, bias=True)
        
        # Output layers: accumulator + space control -> output
        self.l1 = nn.Linear(hidden_size * 2 + 32, 32, bias=True)
        self.l2 = nn.Linear(32, 32, bias=True)
        self.output = nn.Linear(32, 1, bias=True)
        
        # Activation: Clipped ReLU for INT8 quantization
        self.activation = nn.ReLU()
        self.clipped_relu = lambda x: torch.clamp(x, 0.0, 1.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        features: [batch, input_size] = [HalfKP + SpaceControl]
        """
        batch_size = features.size(0)
        half = self.halfkp_size // 2
        
        # Split HalfKP (relative, side-to-move first)
        halfkp = features[:, :self.halfkp_size]
        space_ctrl = features[:, self.halfkp_size:]
        
        # Split into white/black perspectives
        white = halfkp[:, :half]
        black = halfkp[:, half:]
        
        # Feature transformer (accumulator)
        w_acc = self.ft(white)
        b_acc = self.ft(black)
        
        # Clipped ReLU (0 to 1 for INT8 quantization)
        w_acc = self.clipped_relu(w_acc)
        b_acc = self.clipped_relu(b_acc)
        
        # Project space control to smaller dimension
        sc = self.activation(self.sc_proj(space_ctrl))
        
        # Concatenate: [white_acc | black_acc | space_control]
        acc = torch.cat([w_acc, b_acc, sc], dim=1)
        
        # Hidden layers
        x = self.activation(self.l1(acc))
        x = self.activation(self.l2(x))
        
        # Output (centipawns, scaled)
        out = self.output(x)
        
        # Scale: map [-1, 1] to centipawns
        return out * 600.0
    
    def save_quantized(self, filepath: str):
        """Save model with INT8 quantization for engine inference"""
        state = {
            'ft_weight': self.ft.weight.data.cpu().numpy(),
            'ft_bias': self.ft.bias.data.cpu().numpy(),
            'sc_proj_weight': self.sc_proj.weight.data.cpu().numpy(),
            'sc_proj_bias': self.sc_proj.bias.data.cpu().numpy(),
            'l1_weight': self.l1.weight.data.cpu().numpy(),
            'l1_bias': self.l1.bias.data.cpu().numpy(),
            'l2_weight': self.l2.weight.data.cpu().numpy(),
            'l2_bias': self.l2.bias.data.cpu().numpy(),
            'output_weight': self.output.weight.data.cpu().numpy(),
            'output_bias': self.output.bias.data.cpu().numpy(),
        }
        
        # Quantize to INT8
        quantized = {}
        for name, arr in state.items():
            if 'weight' in name:
                # Find scale factor
                max_val = np.max(np.abs(arr))
                scale = max_val / 127.0
                
                # Quantize
                q_arr = np.round(arr / scale).astype(np.int8)
                quantized[name] = q_arr
                quantized[name + '_scale'] = np.float32(scale)
            else:
                quantized[name] = arr.astype(np.float32)
        
        # Save
        with open(filepath, 'wb') as f:
            np.savez_compressed(f, **quantized)
        
        print(f"Saved quantized model to {filepath}")
        
        # Also save as plain weights for debugging
        plain_path = filepath.replace('.nnue', '_plain.npz')
        with open(plain_path, 'wb') as f:
            np.savez_compressed(f, **state)
    
    @classmethod
    def load_quantized(cls, filepath: str):
        """Load quantized model"""
        data = np.load(filepath)
        
        model = cls()
        
        # Dequantize
        state_dict = {}
        for key in data.files:
            if '_scale' in key:
                continue
            
            arr = data[key]
            scale_key = key + '_scale'
            if scale_key in data:
                scale = float(data[scale_key])
                arr = arr.astype(np.float32) * scale
            
            # Map to PyTorch state dict keys
            torch_key = key.replace('_', '.')
            state_dict[torch_key] = torch.from_numpy(arr)
        
        model.load_state_dict(state_dict, strict=False)
        return model


# ============== Training ==============

class Trainer:
    def __init__(self, model: NexusNNUE, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter('runs/nnue_training')
        
        self.best_loss = float('inf')
        self.best_model_path = ''
    
    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader],
              epochs: int,
              lr: float,
              weight_decay: float,
              checkpoint_dir: str = 'checkpoints',
              resume_checkpoint: Optional[str] = None,
              start_epoch: int = 0):
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Resume from checkpoint if provided
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Adjust scheduler if needed
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"  Resuming from epoch {start_epoch}")
            if 'loss' in checkpoint:
                self.best_loss = checkpoint['loss']
                print(f"  Previous best loss: {self.best_loss:.6f}")
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Starting from epoch {start_epoch}, training {epochs} epoch(s)")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            start_time = time.time()
            
            # Training
            train_loss = self._train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                val_loss = self._validate(val_loader)
                scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Logging
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if val_loader:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Checkpoint: save every epoch for micro-training
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss if val_loss > 0 else train_loss,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  -> Saved checkpoint (epoch {epoch+1})")
            
            # Also save best model separately
            if val_loss > 0 and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_path = os.path.join(
                    checkpoint_dir, f'best_model_epoch{epoch+1}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, self.best_model_path)
                print(f"  -> Saved best model (val_loss={val_loss:.6f})")
        
        self.writer.close()
        print(f"\nTraining complete. Best model: {self.best_model_path}")
    
    def _train_epoch(self, loader: DataLoader, optimizer: optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        
        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.6f}", 
                      end='\r')
        
        return total_loss / count
    
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count
    
    def export_for_engine(self, filepath: str):
        """Export final model for engine use"""
        self.model.eval()
        self.model.save_quantized(filepath)
        print(f"Exported model to {filepath}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description='NNUE Trainer for Nexus Infinite')
    
    # Data
    parser.add_argument('--data', required=True, help='Training data file')
    parser.add_argument('--val-data', help='Validation data file')
    parser.add_argument('--max-positions', type=int, default=0, 
                        help='Max positions to load (0=all)')
    
    # Model
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer size (tiny NNUE for CPU)')
    parser.add_argument('--no-space-control', action='store_true',
                        help='Disable space control heatmap features')
    parser.add_argument('--checkpoint', help='Resume from checkpoint')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Output
    parser.add_argument('--output', default='nexus.nnue', help='Output file')
    parser.add_argument('--checkpoints-dir', default='checkpoints',
                        help='Checkpoint directory')
    
    # System
    parser.add_argument('--device', default='cpu', 
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device (default: cpu for GitHub Actions)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0 for CPU micro-training)')
    
    args = parser.parse_args()
    
    # Load data
    print("=" * 60)
    print("Nexus Infinite NNUE Trainer")
    print("=" * 60)
    
    if not args.no_space_control:
        print(f"Space Control: ENABLED (128 extra features)")
        print(f"  - 64 squares attacked by side-to-move")
        print(f"  - 64 squares attacked by opponent")
    else:
        print("Space Control: DISABLED (--no-space-control)")
    
    full_dataset = NexusDataset(args.data, args.max_positions)
    
    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # CPU-optimized DataLoader (no pin_memory, no workers for GitHub Actions)
    pin = (args.device == 'cuda') and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    ) if val_size > 0 else None
    
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    
    # Create model
    model = NexusNNUE(hidden_size=args.hidden_size)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train
    trainer = Trainer(model, device=args.device)
    
    # Determine resume checkpoint
    resume_ckpt = args.checkpoint
    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location='cpu')
        start_epoch = ckpt.get('epoch', 0) + 1
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoints_dir,
        resume_checkpoint=resume_ckpt,
        start_epoch=start_epoch,
    )
    
    # Export
    trainer.export_for_engine(args.output)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
