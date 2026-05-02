#pragma once

#include "types.h"
#include "board.h"
#include "bitboard.h"

namespace Nexus {

// SEE (Static Exchange Evaluation) - fast capture sequence evaluation
// Used for move ordering and pruning decisions

class SEE {
public:
    // Simple piece values for SEE (centipawns)
    static constexpr int PieceValues[PIECE_TYPE_NB] = {
        0, 100, 450, 450, 650, 1300, 0, 0  // Pawn, Knight, Bishop, Rook, Queen
    };
    
    // Evaluate a single capture using SEE
    // Returns true if capture is winning or equal (SEE >= 0)
    static bool is_capture_winning(const BoardState& pos, Move capture);
    
    // Full SEE value for a capture (positive = winning for side to move)
    static int see_value(const BoardState& pos, Move capture);
    
    // Quick threshold test: returns true if SEE >= threshold
    static bool see_ge(const BoardState& pos, Move capture, int threshold);
    
private:
    // Find least valuable attacker of a given color attacking a square
    static PieceType get_lva(const BoardState& pos, Square sq, Color c, u64& occupied);
    
    // Get attacks to a square considering occupied squares
    static u64 attacks_to(const BoardState& pos, Square sq, u64 occupied);
};

// Inline helper for quick access
inline int see_piece_value(PieceType pt) {
    return SEE::PieceValues[pt];
}

} // namespace Nexus
