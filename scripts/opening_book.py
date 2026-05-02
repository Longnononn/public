#!/usr/bin/env python3
"""
Binary Opening Book Generator for Nexus Infinite.

Creates compact binary opening books from PGN/game data.
Supports:
  - Variable-depth entries (common moves = higher depth)
  - Weighted by engine performance (win rate)
  - Diversity enforcement (prevent overfitting one line)
  - ECO-coded entries for curriculum learning

Binary format:
  Header (32 bytes):
    - Magic: "NEXUSOBK" (8 bytes)
    - Version: uint32
    - Entry count: uint32
    - Flags: uint32
    - Reserved: 12 bytes
  Entries (variable):
    - Key: uint64 (Zobrist hash of position)
    - Move: uint16 (from_sq 6 bits | to_sq 6 bits | promo 4 bits)
    - Weight: uint16 (0-65535, based on performance)
    - Depth: uint8 (how deep in opening tree)
    - ECO category: uint8 (tactical/positional/endgame/etc)
    - Win rate: uint8 (0-100, white win %)
    - Draw rate: uint8 (0-100)
    - Game count: uint32
    - Next positions: uint32[] (linked positions for tree structure)
"""

import argparse
import json
import math
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Zobrist keys for opening book hashing
ZOBRIST_PIECE = [
    [[0] * 64 for _ in range(6)]  # 6 piece types
    for _ in range(2)  # 2 colors
]
ZOBRIST_SIDE = [0] * 64
ZOBRIST_CASTLING = [0] * 16
ZOBRIST_ENPASSANT = [0] * 8


def _init_zobrist():
    """Initialize Zobrist keys."""
    import random
    rng = random.Random(0x5D1B1C85)  # Fixed seed for reproducibility
    
    for color in range(2):
        for piece in range(6):
            for sq in range(64):
                ZOBRIST_PIECE[color][piece][sq] = rng.getrandbits(64)
    
    for sq in range(64):
        ZOBRIST_SIDE[sq] = rng.getrandbits(64)
    
    for i in range(16):
        ZOBRIST_CASTLING[i] = rng.getrandbits(64)
    
    for i in range(8):
        ZOBRIST_ENPASSANT[i] = rng.getrandbits(64)


_init_zobrist()


# Piece encoding for book
PIECES = 'PNBRQKpnbrqk'
PIECE_TO_IDX = {p: i for i, p in enumerate(PIECES)}


@dataclass
class BookEntry:
    """Single opening book entry."""
    key: int            # Zobrist hash
    move: int           # Encoded move
    weight: int = 0     # 0-65535
    depth: int = 0      # Opening depth (ply)
    eco_category: int = 0  # 0=general, 1=tactical, 2=positional, 3=endgame, 4=imbalanced
    win_rate: int = 0   # 0-100 (from White perspective)
    draw_rate: int = 0  # 0-100
    game_count: int = 0
    
    # Derived
    score: float = 0.0  # Computed selection score
    diversity_penalty: float = 0.0  # Penalty for over-representation


class PositionHasher:
    """Compute Zobrist hash from FEN."""
    
    @staticmethod
    def hash_fen(fen: str) -> int:
        """Compute Zobrist hash for a FEN position."""
        parts = fen.split()
        board = parts[0]
        side = parts[1] if len(parts) > 1 else 'w'
        castling = parts[2] if len(parts) > 2 else '-'
        enpassant = parts[3] if len(parts) > 3 else '-'
        
        key = 0
        
        # Hash pieces
        sq = 56  # a8
        for char in board:
            if char == '/':
                sq -= 16
            elif char.isdigit():
                sq += int(char)
            else:
                color = 0 if char.isupper() else 1
                piece = PIECE_TO_IDX.get(char, -1)
                if piece >= 0:
                    key ^= ZOBRIST_PIECE[color][piece % 6][sq]
                sq += 1
        
        # Hash side to move
        if side == 'b':
            key ^= ZOBRIST_SIDE[0]  # Fixed side key
        
        # Hash castling rights
        castling_bits = 0
        if 'K' in castling: castling_bits |= 1
        if 'Q' in castling: castling_bits |= 2
        if 'k' in castling: castling_bits |= 4
        if 'q' in castling: castling_bits |= 8
        key ^= ZOBRIST_CASTLING[castling_bits]
        
        # Hash en passant
        if enpassant != '-' and len(enpassant) == 2:
            file_idx = ord(enpassant[0]) - ord('a')
            if 0 <= file_idx < 8:
                key ^= ZOBRIST_ENPASSANT[file_idx]
        
        return key & 0xFFFFFFFFFFFFFFFF
    
    @staticmethod
    def encode_move(from_sq: int, to_sq: int, promotion: int = 0) -> int:
        """Encode move as 16-bit value."""
        return (from_sq & 0x3F) | ((to_sq & 0x3F) << 6) | ((promotion & 0xF) << 12)
    
    @staticmethod
    def decode_move(move: int) -> Tuple[int, int, int]:
        """Decode move from 16-bit value."""
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        promotion = (move >> 12) & 0xF
        return from_sq, to_sq, promotion


class OpeningBookBuilder:
    """Builds binary opening book from game data."""
    
    MAGIC = b"NEXUSOBK"
    VERSION = 1
    
    # Flags
    FLAG_SORTED = 1       # Entries sorted by key for binary search
    FLAG_WEIGHTED = 2     # Entries weighted by performance
    FLAG_ECO = 4          # Contains ECO category data
    FLAG_TREE = 8         # Contains tree structure (next positions)
    
    def __init__(self):
        self.entries: Dict[int, List[BookEntry]] = defaultdict(list)
        self.position_counts: Dict[int, int] = defaultdict(int)
        self.total_games = 0
    
    def add_game(self, moves: List[Tuple[str, str, Optional[str]]],
                 result: str = '*',  # '1-0', '0-1', '1/2-1/2'
                 eco: str = '',
                 max_depth: int = 20):
        """
        Add game to book.
        
        moves: [(fen, uci_move, eco_code), ...]
        result: game result from White perspective
        """
        self.total_games += 1
        
        for ply, (fen, move_uci, move_eco) in enumerate(moves):
            if ply >= max_depth:
                break
            
            key = PositionHasher.hash_fen(fen)
            self.position_counts[key] += 1
            
            # Parse UCI move
            from_sq = self._uci_to_sq(move_uci[:2])
            to_sq = self._uci_to_sq(move_uci[2:4])
            promotion = 0
            if len(move_uci) == 5:
                promo_map = {'n': 1, 'b': 2, 'r': 3, 'q': 4,
                           'N': 1, 'B': 2, 'R': 3, 'Q': 4}
                promotion = promo_map.get(move_uci[4], 0)
            
            encoded_move = PositionHasher.encode_move(from_sq, to_sq, promotion)
            
            # Determine ECO category
            eco_cat = self._eco_category(move_eco or eco)
            
            # Check if entry exists
            existing = next((e for e in self.entries[key] 
                           if e.move == encoded_move), None)
            
            if existing:
                existing.game_count += 1
                
                # Update win/draw rates
                if result == '1-0':
                    # Existing win rate is white win %
                    existing.win_rate = (
                        existing.win_rate * (existing.game_count - 1) + 100
                    ) // existing.game_count
                elif result == '0-1':
                    existing.win_rate = (
                        existing.win_rate * (existing.game_count - 1)
                    ) // existing.game_count
                elif result == '1/2-1/2':
                    existing.draw_rate = (
                        existing.draw_rate * (existing.game_count - 1) + 100
                    ) // existing.game_count
            else:
                # New entry
                entry = BookEntry(
                    key=key,
                    move=encoded_move,
                    depth=ply,
                    eco_category=eco_cat,
                    game_count=1,
                    win_rate=100 if result == '1-0' else (50 if result == '1/2-1/2' else 0),
                    draw_rate=100 if result == '1/2-1/2' else 0,
                )
                self.entries[key].append(entry)
    
    def _uci_to_sq(self, uci_sq: str) -> int:
        """Convert 'e2' to square index."""
        if len(uci_sq) != 2:
            return 0
        file = ord(uci_sq[0]) - ord('a')
        rank = int(uci_sq[1]) - 1
        return rank * 8 + file
    
    def _eco_category(self, eco: str) -> int:
        """Determine category from ECO code."""
        if not eco:
            return 0
        
        # A = Flank openings (positional/closed)
        # B = Semi-open (tactical/sharp)
        # C = Open games (tactical)
        # D = Closed games (positional)
        # E = Indian/Reti (positional/complex)
        first = eco[0].upper() if eco else ''
        
        if first == 'C':
            return 1  # tactical
        elif first == 'B':
            return 1  # tactical/semi-open
        elif first == 'A':
            return 2  # positional/flank
        elif first == 'D':
            return 2  # positional/closed
        elif first == 'E':
            return 3  # complex/endgame-oriented
        
        return 0  # general
    
    def compute_weights(self, 
                       diversity_target: int = 3,
                       performance_weight: float = 0.4,
                       diversity_weight: float = 0.3,
                       depth_weight: float = 0.3):
        """Compute selection weights for all entries.
        
        Formula:
          score = performance * perf_weight + 
                  depth_bonus * depth_weight - 
                  diversity_penalty * diversity_weight
        
        Ensures no single line dominates (>diversity_target games).
        """
        for key, entries in self.entries.items():
            pos_count = self.position_counts[key]
            
            for entry in entries:
                # Performance: prefer moves with good results
                # For White: high win rate
                # For balanced: high draw rate + decent win rate
                performance = (entry.win_rate + entry.draw_rate * 0.5) / 150.0
                performance = min(1.0, max(0.0, performance))
                
                # Depth bonus: early moves = more important
                depth_bonus = max(0, 1.0 - entry.depth / 20.0)
                
                # Diversity penalty: if this move is over-represented
                move_ratio = entry.game_count / max(1, pos_count)
                if pos_count >= diversity_target:
                    if move_ratio > 0.6:  # More than 60% of games
                        entry.diversity_penalty = (move_ratio - 0.6) * 2.5
                
                entry.score = (performance * performance_weight +
                              depth_bonus * depth_weight -
                              entry.diversity_penalty * diversity_weight)
                
                # Convert to weight (0-65535)
                entry.weight = int(entry.score * 65535)
    
    def write_binary(self, filename: str, 
                    flags: int = FLAG_SORTED | FLAG_WEIGHTED | FLAG_ECO):
        """Write book to binary file."""
        
        # Flatten entries
        all_entries = []
        for key, entries in self.entries.items():
            all_entries.extend(entries)
        
        # Sort by key for binary search
        if flags & self.FLAG_SORTED:
            all_entries.sort(key=lambda e: e.key)
        
        with open(filename, 'wb') as f:
            # Header
            f.write(self.MAGIC)                          # 8 bytes
            f.write(struct.pack('<I', self.VERSION))     # 4 bytes
            f.write(struct.pack('<I', len(all_entries))) # 4 bytes
            f.write(struct.pack('<I', flags))            # 4 bytes
            f.write(b'\x00' * 12)                         # Reserved: 12 bytes
            # Total: 32 bytes
            
            # Entries
            for entry in all_entries:
                f.write(struct.pack('<Q', entry.key))        # 8 bytes
                f.write(struct.pack('<H', entry.move))         # 2 bytes
                f.write(struct.pack('<H', entry.weight))     # 2 bytes
                f.write(struct.pack('<B', entry.depth))      # 1 byte
                f.write(struct.pack('<B', entry.eco_category)) # 1 byte
                f.write(struct.pack('<B', entry.win_rate))     # 1 byte
                f.write(struct.pack('<B', entry.draw_rate))    # 1 byte
                f.write(struct.pack('<I', entry.game_count))  # 4 bytes
                # Total: 20 bytes per entry
            
            # Index (optional, for faster lookup)
            if flags & self.FLAG_SORTED:
                # Write index: key -> file offset mapping
                # Simplified: just write key boundaries
                pass
        
        # Stats
        file_size = os.path.getsize(filename)
        print(f"Written {len(all_entries)} entries ({file_size} bytes)")
        print(f"Average entry size: {file_size / max(1, len(all_entries)):.1f} bytes")
        print(f"Positions covered: {len(self.entries)}")
    
    def write_text(self, filename: str):
        """Write human-readable text version."""
        with open(filename, 'w') as f:
            f.write(f"# Nexus Infinite Opening Book\n")
            f.write(f"# Version: {self.VERSION}\n")
            f.write(f"# Total entries: {sum(len(e) for e in self.entries.values())}\n")
            f.write(f"# Positions: {len(self.entries)}\n\n")
            
            for key in sorted(self.entries.keys()):
                entries = self.entries[key]
                f.write(f"\nPosition {key:016x} ({self.position_counts[key]} games)\n")
                
                # Sort by weight
                entries.sort(key=lambda e: -e.weight)
                
                for entry in entries:
                    from_sq, to_sq, promo = PositionHasher.decode_move(entry.move)
                    from_uci = f"{chr(ord('a') + from_sq % 8)}{from_sq // 8 + 1}"
                    to_uci = f"{chr(ord('a') + to_sq % 8)}{to_sq // 8 + 1}"
                    promo_str = ''
                    if promo > 0:
                        promo_map = {1: 'n', 2: 'b', 3: 'r', 4: 'q'}
                        promo_str = promo_map.get(promo, '')
                    
                    cat_names = {0: 'general', 1: 'tactical', 2: 'positional',
                                3: 'endgame', 4: 'imbalanced'}
                    
                    f.write(f"  {from_uci}{to_uci}{promo_str}: "
                           f"weight={entry.weight} "
                           f"depth={entry.depth} "
                           f"cat={cat_names.get(entry.eco_category, '?')} "
                           f"W/D/L={entry.win_rate}/{entry.draw_rate}/"
                           f"{100-entry.win_rate-entry.draw_rate} "
                           f"n={entry.game_count}\n")
    
    def generate_stats(self) -> Dict:
        """Generate book statistics."""
        stats = {
            'total_positions': len(self.entries),
            'total_entries': sum(len(e) for e in self.entries.values()),
            'total_games': self.total_games,
            'avg_moves_per_position': 0.0,
            'category_distribution': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            'depth_distribution': defaultdict(int),
            'top_positions': [],
        }
        
        total_moves = 0
        for key, entries in self.entries.items():
            total_moves += len(entries)
            
            for entry in entries:
                stats['category_distribution'][entry.eco_category] += 1
                stats['depth_distribution'][entry.depth] += 1
        
        stats['avg_moves_per_position'] = total_moves / max(1, len(self.entries))
        
        # Top 10 positions by game count
        top = sorted(self.position_counts.items(), key=lambda x: -x[1])[:10]
        stats['top_positions'] = [
            {'key': hex(k), 'games': v, 'moves': len(self.entries.get(k, []))}
            for k, v in top
        ]
        
        return stats


def create_from_pgn(pgn_file: str, output_file: str, 
                    max_depth: int = 20,
                    min_games: int = 2,
                    text_output: Optional[str] = None):
    """Create opening book from PGN file."""
    
    builder = OpeningBookBuilder()
    
    # Parse PGN (simplified - real impl would use proper PGN parser)
    print(f"Reading {pgn_file}...")
    
    # For now, read simple format: one game per line
    # Format: FEN moves | result | ECO
    # Example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5 | 1-0 | C20"
    
    games_parsed = 0
    
    try:
        with open(pgn_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                
                start_fen = parts[0].strip()
                moves_str = parts[1].strip()
                result = parts[2].strip() if len(parts) > 2 else '*'
                eco = parts[3].strip() if len(parts) > 3 else ''
                
                moves = moves_str.split()
                
                # Build position-move sequence
                # Would need actual move application to get FENs
                # For now, use placeholder
                move_seq = []
                for i, move in enumerate(moves[:max_depth]):
                    move_seq.append((start_fen, move, eco))
                    # Would update FEN after each move
                
                if move_seq:
                    builder.add_game(move_seq, result, eco, max_depth)
                    games_parsed += 1
                
                if games_parsed % 10000 == 0:
                    print(f"  Parsed {games_parsed} games...")
    
    except FileNotFoundError:
        print(f"PGN file not found: {pgn_file}")
        print("Creating empty book...")
    
    # Compute weights
    print("Computing weights...")
    builder.compute_weights()
    
    # Write binary
    print(f"Writing binary book to {output_file}...")
    builder.write_binary(output_file)
    
    # Write text version if requested
    if text_output:
        print(f"Writing text book to {text_output}...")
        builder.write_text(text_output)
    
    # Stats
    stats = builder.generate_stats()
    print(f"\nBook statistics:")
    print(f"  Positions: {stats['total_positions']}")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Games: {stats['total_games']}")
    print(f"  Avg moves/position: {stats['avg_moves_per_position']:.1f}")
    
    return builder


def main():
    parser = argparse.ArgumentParser(description='Binary Opening Book Builder')
    parser.add_argument('--input', help='Input PGN or game file')
    parser.add_argument('--output', required=True, help='Output .obk binary file')
    parser.add_argument('--text', help='Output human-readable .txt file')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Maximum opening depth in plies')
    parser.add_argument('--min-games', type=int, default=2,
                        help='Minimum games per position')
    parser.add_argument('--stats', action='store_true',
                        help='Only show stats of existing book')
    
    args = parser.parse_args()
    
    if args.stats:
        # Read and display stats
        if not os.path.exists(args.output):
            print(f"Book not found: {args.output}")
            sys.exit(1)
        
        with open(args.output, 'rb') as f:
            magic = f.read(8)
            version = struct.unpack('<I', f.read(4))[0]
            entries = struct.unpack('<I', f.read(4))[0]
            flags = struct.unpack('<I', f.read(4))[0]
        
        print(f"Book: {args.output}")
        print(f"Magic: {magic}")
        print(f"Version: {version}")
        print(f"Entries: {entries}")
        print(f"Flags: {flags:08x}")
        print(f"Size: {os.path.getsize(args.output)} bytes")
        return
    
    # Create book
    if not args.input:
        # Create from curriculum positions instead
        print("No input file, creating from curriculum positions...")
        builder = OpeningBookBuilder()
        
        # Add standard starting position
        # Standard opening moves with high weights
        start_moves = [
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 
             'e2e4', 'C20'),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
             'd2d4', 'D00'),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
             'c2c4', 'A10'),
            ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
             'g1f3', 'A04'),
        ]
        
        for fen, move, eco in start_moves:
            builder.add_game([(fen, move, eco)], '*', eco, 1)
        
        builder.compute_weights()
        builder.write_binary(args.output)
    else:
        create_from_pgn(args.input, args.output, 
                       args.max_depth, args.min_games, args.text)


if __name__ == '__main__':
    main()
