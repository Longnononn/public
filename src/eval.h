#pragma once

#include "types.h"
#include "board.h"

namespace Nexus {

class Eval {
public:
    static Value evaluate(const BoardState& pos);
    static Value eval_after_move(const BoardState& pos, Move m, Value currentEval);
    
    // Material weights (centipawns)
    static constexpr int PieceValue[PIECE_TYPE_NB] = {
        0, 126, 781, 825, 1276, 2538, 0, 0
    };
    
private:
    // Piece-Square Tables (simplified, from PeSTO / Stockfish classical)
    static const int Pst[PIECE_TYPE_NB][SQUARE_NB];
    
    static Value evaluate_pawns(const BoardState& pos, Color us);
    static Value evaluate_knights(const BoardState& pos, Color us);
    static Value evaluate_bishops(const BoardState& pos, Color us);
    static Value evaluate_rooks(const BoardState& pos, Color us);
    static Value evaluate_queens(const BoardState& pos, Color us);
    static Value evaluate_king(const BoardState& pos, Color us);
    static Value evaluate_mobility(const BoardState& pos, Color us);
    
    static int game_phase(const BoardState& pos);
    
    static constexpr int phase_values[PIECE_TYPE_NB] = {
        0, 0, 1, 1, 2, 4, 0, 0
    };
    static constexpr int total_phase = 24; // 4*1 + 4*1 + 4*2 + 2*4 = 24
};

} // namespace Nexus
