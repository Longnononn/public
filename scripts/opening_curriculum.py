#!/usr/bin/env python3
"""
Dynamic Opening Curriculum Manager for Nexus Infinite.

Instead of fixed random openings, rotates through categories to prevent
overfitting and maximize diversity. Ensures engine sees all types of positions.

Categories (rotated per training cycle):
  1. TACTICAL: Gambits, sharp Sicilians, attacking openings
  2. POSITIONAL: Closed games, d4 openings, fianchetto systems
  3. ENDGAME: Simplified early, technical endgames from opening
  4. IMBALANCED: Material imbalance, exchange sacrifices
  5. DEFENSIVE: Underdog positions, cramped counterplay
  6. COMPUTER: Known engine vs engine lines, anti-computer prep
  7. RANDOM: Pure random from legal positions
  8. FASHION: Recent human GM trends, popular lines

Each category has a difficulty progression (easy -> hard).
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


@dataclass
class OpeningPosition:
    """Single opening position with metadata."""
    fen: str
    eco_code: str = ""           # Encyclopedia of Chess Openings
    name: str = ""
    category: str = "general"
    difficulty: int = 1            # 1-10 scale
    seen_count: int = 0          # How many times used in training
    last_seen: str = ""          # ISO date
    elo_range: Tuple[int, int] = (0, 3000)
    engine_score: int = 0        # Known engine eval at depth 20
    result_stats: Dict[str, int] = field(default_factory=lambda: {
        'white_wins': 0, 'draws': 0, 'black_wins': 0, 'total': 0
    })
    average_result: float = 0.0    # +1 white win, 0 draw, -1 black win
    value_score: float = 0.0     # Computed training value


class DynamicCurriculum:
    """Manages opening curriculum with diversity scheduling."""
    
    CATEGORIES = [
        'tactical', 'positional', 'endgame', 'imbalanced',
        'defensive', 'computer', 'random', 'fashion'
    ]
    
    # Rotation: each cycle focuses on 2-3 categories, then shifts
    ROTATION_SCHEDULE = [
        ['tactical', 'positional'],      # Cycle 1: Build fundamentals
        ['tactical', 'imbalanced'],     # Cycle 2: Sharp positions
        ['positional', 'endgame'],      # Cycle 3: Technique
        ['imbalanced', 'defensive'],    # Cycle 4: Resourcefulness
        ['computer', 'fashion'],        # Cycle 5: Modern trends
        ['random', 'tactical'],         # Cycle 6: Adaptability
        ['endgame', 'defensive'],       # Cycle 7: Survival
        ['positional', 'fashion'],      # Cycle 8: Mastery
    ]
    
    def __init__(self, curriculum_file: str = "opening_curriculum.json",
                 stats_file: str = "opening_stats.json"):
        self.curriculum_file = curriculum_file
        self.stats_file = stats_file
        self.positions: List[OpeningPosition] = []
        self.cycle_number: int = 0
        self.load()
    
    def load(self):
        """Load curriculum from JSON."""
        if os.path.exists(self.curriculum_file):
            with open(self.curriculum_file, 'r') as f:
                data = json.load(f)
            
            for item in data.get('positions', []):
                pos = OpeningPosition(**item)
                # Fix dict conversion
                if isinstance(pos.result_stats, list):
                    pos.result_stats = dict(pos.result_stats)
                self.positions.append(pos)
            
            self.cycle_number = data.get('cycle_number', 0)
            print(f"Loaded {len(self.positions)} positions, cycle {self.cycle_number}")
    
    def save(self):
        """Save curriculum to JSON."""
        data = {
            'positions': [asdict(p) for p in self.positions],
            'cycle_number': self.cycle_number,
            'last_updated': datetime.now().isoformat(),
            'categories': self.CATEGORIES,
        }
        
        with open(self.curriculum_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_position(self, fen: str, category: str = "general",
                     name: str = "", eco: str = "",
                     difficulty: int = 1, engine_score: int = 0):
        """Add new opening position to curriculum."""
        pos = OpeningPosition(
            fen=fen,
            category=category,
            name=name,
            eco_code=eco,
            difficulty=difficulty,
            engine_score=engine_score
        )
        self.positions.append(pos)
    
    def add_tactical_set(self):
        """Add known sharp/tactical positions."""
        tactical = [
            # King's Gambit Accepted
            ("rnbqkb1r/pppp1ppp/5n2/4p3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3",
             "C30", "King's Gambit Accepted", 3),
            # Sicilian Najdorf
            ("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N1B3/PPP2PPP/R1BQK2R b KQkq - 0 7",
             "B90", "Sicilian Najdorf", 8),
            # Dragon Yugoslav Attack
            ("r1bq1rk1/pp1n1pp1/2pb1n1p/3pp3/3PP3/2PB1N1P/PPQ2PP1/R1B2RK1 w - - 0 11",
             "B76", "Dragon Yugoslav", 7),
            # Evans Gambit
            ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
             "C51", "Evans Gambit", 4),
            # Danish Gambit
            ("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 3",
             "C21", "Danish Gambit", 3),
            # Budapest Gambit
            ("rnbqkb1r/pppp1ppp/5n2/4p3/2P1P3/8/PP1P1PPP/RNBQKBNR w KQkq - 1 4",
             "A52", "Budapest Gambit", 4),
            # Benko Gambit
            ("rnbqkb1r/p2ppppp/1p3n2/2p5/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 4",
             "A57", "Benko Gambit", 5),
            # Two Knights Defense, Fried Liver
            ("r1bq1b1r/ppp2kpp/2n5/3np3/2B5/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 6",
             "C57", "Fried Liver Attack", 6),
            # Marshall Attack
            ("r1bq1rk1/ppp2ppp/2n2n2/3pp3/2PP4/2PBPN2/P4PPP/R1BQ1RK1 b - - 0 8",
             "C89", "Marshall Attack", 8),
            # Traxler Counter-Attack
            ("r1bqkb1r/pppp1ppp/5n2/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
             "C57", "Traxler Counter-Attack", 7),
        ]
        
        for fen, eco, name, diff in tactical:
            self.add_position(fen, 'tactical', name, eco, diff)
    
    def add_positional_set(self):
        """Add closed/positional positions."""
        positional = [
            # Queen's Gambit Declined
            ("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/4PN2/PP3PPP/RNBQKB1R b KQkq - 0 4",
             "D35", "QGD Exchange", 3),
            # English Opening
            ("rnbqkb1r/pppp1ppp/5n2/4p3/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 0 3",
             "A15", "English Opening", 2),
            # Catalan
            ("rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3",
             "E06", "Catalan", 3),
            # Reti
            ("rnbqkb1r/pppppppp/5n2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 0 2",
             "A04", "Reti Opening", 2),
            # King's Indian Defense
            ("rnbqk2r/ppppppbp/5np1/8/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq - 0 5",
             "E60", "King's Indian", 4),
            # Grunfeld
            ("rnbqkb1r/pp1ppppp/5n2/2p5/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 3",
             "D85", "Grunfeld", 5),
            # Nimzo-Indian
            ("rnbqk2r/ppp1ppbp/5np1/3p4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 0 5",
             "E32", "Nimzo-Indian", 5),
            # Slav Defense
            ("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
             "D10", "Slav Defense", 3),
            # London System
            ("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 0 3",
             "D02", "London System", 2),
            # Colle System
            ("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/3BPN2/PPP2PPP/RN1QKB1R b KQkq - 0 4",
             "D04", "Colle System", 3),
        ]
        
        for fen, eco, name, diff in positional:
            self.add_position(fen, 'positional', name, eco, diff)
    
    def add_endgame_set(self):
        """Add simplified/endgame-oriented positions."""
        endgame = [
            # Berlin Defense (Ruy Lopez, known endgame)
            ("r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
             "C67", "Berlin Defense", 4),
            # Queenless middlegame
            ("r1b2rk1/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQR1K1 w - - 0 8",
             "C67", "Berlin Endgame", 5),
            # Exchange Ruy Lopez
            ("r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
             "C68", "Exchange Ruy Lopez", 3),
            # Caro-Kann Endgame
            ("rnbqkb1r/ppp1pppp/2p5/8/3Pp3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",
             "B13", "Caro-Kann Exchange", 3),
            # French Exchange
            ("rnbqkb1r/ppp2ppp/4pn2/8/3Pp3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",
             "C01", "French Exchange", 2),
            # Italian Game, Giuoco Pianissimo
            ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
             "C50", "Giuoco Pianissimo", 2),
            # Petrov's Defense
            ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
             "C42", "Petrov Defense", 3),
            # Four Knights
            ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",
             "C47", "Four Knights", 2),
        ]
        
        for fen, eco, name, diff in endgame:
            self.add_position(fen, 'endgame', name, eco, diff)
    
    def add_imbalanced_set(self):
        """Add material imbalance positions."""
        imbalanced = [
            # Exchange sacrifice positions
            ("r1bqk2r/ppp1bppp/2n5/3pp3/1nB1P3/5N2/PPPP1PPP/RNBQR1K1 w kq - 0 7",
             "C78", "Exchange Sacrifice", 6),
            # Minor piece sacrifice for initiative
            ("r1bqk2r/ppp2ppp/2n2n2/1B1pp3/1b2P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 6",
             "C57", "Two Knights Sacrifice", 5),
            # Rook vs minor piece
            ("r1bq1rk1/ppp2ppp/2n5/1B1pp3/1b2P3/5N2/PPPP1PPP/RNBQR1K1 w - - 0 7",
             "C78", "Rook for Bishop", 7),
            # Quality sacrifice
            ("r1bq1rk1/pppp1ppp/2n2n2/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQR1K1 w - - 0 6",
             "C78", "Quality Sacrifice", 6),
            # Doubled pawns compensation
            ("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/4PN2/PP3PPP/RNBQKB1R w KQkq - 0 4",
             "D30", "QGD, Isolated Queen Pawn", 5),
        ]
        
        for fen, eco, name, diff in imbalanced:
            self.add_position(fen, 'imbalanced', name, eco, diff)
    
    def get_current_categories(self) -> List[str]:
        """Get categories for current cycle."""
        idx = self.cycle_number % len(self.ROTATION_SCHEDULE)
        return self.ROTATION_SCHEDULE[idx]
    
    def select_openings(self, count: int = 10, 
                       difficulty_range: Tuple[int, int] = (1, 10),
                       avoid_recent: bool = True) -> List[OpeningPosition]:
        """Select diverse openings for current cycle."""
        categories = self.get_current_categories()
        
        # Filter by category and difficulty
        candidates = [
            p for p in self.positions
            if p.category in categories
            and difficulty_range[0] <= p.difficulty <= difficulty_range[1]
        ]
        
        # Compute selection score (diversity + freshness)
        def score(pos: OpeningPosition) -> float:
            # Prefer unseen positions
            freshness = 1.0 / (1 + pos.seen_count * 0.5)
            
            # Prefer positions with balanced results (not too drawish)
            if pos.result_stats['total'] > 10:
                balance = 1.0 - abs(pos.average_result)
            else:
                balance = 0.5
            
            # Prefer higher training value
            value = pos.value_score
            
            return freshness * 0.4 + balance * 0.3 + value * 0.3
        
        # Sort by score, take top candidates
        candidates.sort(key=score, reverse=True)
        
        # Ensure diversity within selection (don't pick too similar)
        selected = []
        for pos in candidates:
            if len(selected) >= count:
                break
            
            # Check if too similar to already selected
            similar = False
            for s in selected:
                if self._similar_positions(pos, s):
                    similar = True
                    break
            
            if not similar:
                selected.append(pos)
        
        # If not enough, fill with random from remaining
        if len(selected) < count:
            remaining = [p for p in candidates if p not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:count - len(selected)])
        
        # Update seen count
        for pos in selected:
            pos.seen_count += 1
            pos.last_seen = datetime.now().isoformat()
        
        self.save()
        
        return selected
    
    def _similar_positions(self, a: OpeningPosition, 
                          b: OpeningPosition) -> bool:
        """Check if two positions are too similar (same opening family)."""
        # Same ECO family (first letter)
        if a.eco_code and b.eco_code:
            if a.eco_code[0] == b.eco_code[0]:
                # Very similar if same first 2 chars
                if len(a.eco_code) >= 2 and len(b.eco_code) >= 2:
                    if a.eco_code[:2] == b.eco_code[:2]:
                        return True
        
        # Same category is OK, but same exact name = same position
        if a.name == b.name and a.name:
            return True
        
        return False
    
    def advance_cycle(self):
        """Move to next training cycle."""
        self.cycle_number += 1
        self.save()
        
        categories = self.get_current_categories()
        print(f"Advanced to cycle {self.cycle_number}")
        print(f"Focus categories: {', '.join(categories)}")
    
    def update_results(self, fen: str, result: str):
        """Update result statistics for an opening.
        
        result: '1-0', '0-1', '1/2-1/2', '*'
        """
        for pos in self.positions:
            if pos.fen == fen:
                pos.result_stats['total'] += 1
                
                if result == '1-0':
                    pos.result_stats['white_wins'] += 1
                    pos.average_result = ((pos.average_result * 
                                          (pos.result_stats['total'] - 1) + 1) /
                                          pos.result_stats['total'])
                elif result == '0-1':
                    pos.result_stats['black_wins'] += 1
                    pos.average_result = ((pos.average_result * 
                                          (pos.result_stats['total'] - 1) - 1) /
                                          pos.result_stats['total'])
                elif result == '1/2-1/2':
                    pos.result_stats['draws'] += 1
                    # average stays same for draws
                
                self.save()
                break
    
    def export_pgn_openings(self, count: int = 20, 
                           output_file: str = "openings.pgn"):
        """Export selected openings as PGN file for book."""
        openings = self.select_openings(count)
        
        with open(output_file, 'w') as f:
            for i, pos in enumerate(openings, 1):
                f.write(f'[Event "Nexus Training Opening {i}"]\n')
                f.write(f'[ECO "{pos.eco_code}"]\n')
                f.write(f'[Opening "{pos.name}"]\n')
                f.write(f'[Setup "1"]\n')
                f.write(f'[FEN "{pos.fen}"]\n')
                f.write(f'[Difficulty "{pos.difficulty}"]\n')
                f.write('\n*\n\n')
        
        print(f"Exported {len(openings)} openings to {output_file}")
    
    def get_statistics(self) -> Dict:
        """Get curriculum statistics."""
        stats = {
            'total_positions': len(self.positions),
            'by_category': {},
            'by_difficulty': {},
            'total_seen': sum(p.seen_count for p in self.positions),
            'cycle_number': self.cycle_number,
            'current_focus': self.get_current_categories(),
            'average_result_balance': 0.0,
        }
        
        for pos in self.positions:
            cat = pos.category
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            diff = pos.difficulty
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1
        
        total_games = sum(p.result_stats['total'] for p in self.positions)
        if total_games > 0:
            avg = sum(p.average_result * p.result_stats['total'] 
                     for p in self.positions) / total_games
            stats['average_result_balance'] = avg
        
        return stats


def initialize_curriculum(output_file: str = "opening_curriculum.json"):
    """Create initial curriculum with all position sets."""
    curriculum = DynamicCurriculum(output_file)
    
    print("Initializing opening curriculum...")
    curriculum.add_tactical_set()
    curriculum.add_positional_set()
    curriculum.add_endgame_set()
    curriculum.add_imbalanced_set()
    
    curriculum.save()
    
    stats = curriculum.get_statistics()
    print(f"\nInitialized {stats['total_positions']} positions:")
    for cat, count in stats['by_category'].items():
        print(f"  {cat}: {count}")
    
    return curriculum


def main():
    parser = argparse.ArgumentParser(description='Dynamic Opening Curriculum')
    parser.add_argument('--init', action='store_true', 
                        help='Initialize curriculum database')
    parser.add_argument('--curriculum', default='opening_curriculum.json')
    parser.add_argument('--select', type=int, help='Select N openings')
    parser.add_argument('--export-pgn', help='Export as PGN file')
    parser.add_argument('--advance-cycle', action='store_true',
                        help='Advance to next cycle')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics')
    parser.add_argument('--difficulty-min', type=int, default=1)
    parser.add_argument('--difficulty-max', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.init:
        initialize_curriculum(args.curriculum)
        return
    
    curriculum = DynamicCurriculum(args.curriculum)
    
    if args.stats:
        stats = curriculum.get_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    if args.advance_cycle:
        curriculum.advance_cycle()
        return
    
    if args.select:
        openings = curriculum.select_openings(
            args.select,
            (args.difficulty_min, args.difficulty_max)
        )
        
        print(f"\nSelected {len(openings)} openings (Cycle {curriculum.cycle_number}):")
        print(f"Focus: {', '.join(curriculum.get_current_categories())}")
        print()
        
        for i, pos in enumerate(openings, 1):
            print(f"{i}. [{pos.eco_code}] {pos.name} "
                  f"(diff={pos.difficulty}, seen={pos.seen_count}x)")
            print(f"   FEN: {pos.fen}")
        
        if args.export_pgn:
            curriculum.export_pgn_openings(args.select, args.export_pgn)
    
    curriculum.save()


if __name__ == '__main__':
    main()
