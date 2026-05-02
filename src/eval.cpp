#include "eval.h"
#include <algorithm>

namespace Nexus {

// PeSTO-style piece-square tables (tapered eval: mg + (eg - mg) * phase / 24)
// Tables are from White's perspective (need to flip for Black)

const int Eval::Pst[PIECE_TYPE_NB][SQUARE_NB] = {
    // Empty
    { 0 },
    
    // Pawn (middle game)
    {
         0,   0,   0,   0,   0,   0,   0,   0,
        98, 134,  61,  95,  68, 126,  34, -11,
        -6,   7,  26,  31,  65,  56,  25, -20,
       -14,  13,   6,  21,  23,  12,  17, -23,
       -27,  -2,  -5,  12,  17,   6,  10, -25,
       -26,  -9, -10,  10,  13,   0,   4, -23,
       -35,  -1, -20, -23, -15,  24,  38, -22,
         0,   0,   0,   0,   0,   0,   0,   0,
    },
    
    // Knight
    {
       -167, -89, -34, -49,  61, -97, -15, -107,
        -73, -41,  72,  36,  23,  62,   7,  -17,
        -47,  60,  37,  65,  84, 129,  73,   44,
         -9,  17,  19,  53,  37,  69,  18,   22,
        -13,   4,  16,  13,  28,  19,  21,   -8,
        -23,  -9,  12,  10,  19,  17,  25,  -16,
        -29, -53, -12,  -3,  -1,  18, -14,  -19,
       -105, -21, -58, -33, -17, -28, -19,  -23,
    },
    
    // Bishop
    {
       -29,   4, -82, -37, -25, -42,   7,  -8,
       -26,  16, -18, -13,  30,  59,  18, -47,
       -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
         0,  15,  15,  15,  14,  27,  18,  10,
         4,  15,  16,   0,   7,  21,  33,   1,
       -33,  -3, -14, -21, -13, -12, -10, -21,
    },
    
    // Rook
    {
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  53,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
       -24, -11,   7,  26, 24, 35,  -8, -20,
       -36, -26, -12,  -1,  9, -7,   6, -23,
       -45, -25, -16, -17,  3,  0,  -5, -33,
       -44, -16, -20,  -9, -1, 11,  -6, -71,
       -19, -13,   1,  17, 16,  7, -37, -26,
    },
    
    // Queen
    {
       -28,  30,   3,  65,  62,  55,  56,  34,
       -27, -11,  14,  58,  55,  55,  57,  11,
       -16, -20,   6,  24,  37,  37,  19,  15,
       -18, -18, -11,  27,  32,  24,  14,   8,
       -22, -20,  -6,  17,  16,   3,  -1, -15,
       -26, -22, -14,   1,   2,  -5, -16, -22,
       -30, -25, -18, -16, -12, -21, -24, -33,
       -33, -28, -22, -43,  -5, -32, -20, -41,
    },
    
    // King (middle game)
    {
       -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
       -17, -20, -12, -27, -30, -25, -14, -36,
       -49,  -1, -27, -39, -46, -44, -33, -51,
       -14, -14, -22, -46, -44, -30, -15, -27,
         1,   7,  -8, -64, -43, -16,   9,   8,
       -15,  36,  12, -54,   8, -28,  24,  14,
    },
    
    // All pieces (placeholder)
    { 0 }
};

// Endgame PST (simplified — in production, use separate tables)
// For now, we use the same tables with a simple scaling

Value Eval::evaluate(const BoardState& pos) {
    Color us = pos.sideToMove;
    Color them = ~us;
    
    Value score = VALUE_ZERO;
    int phase = game_phase(pos);
    
    // Material + PST
    for (Square s = SQ_A1; s <= SQ_H8; s = Square(s + 1)) {
        Piece p = pos.piece_on(s);
        if (p == NO_PIECE) continue;
        
        PieceType pt = type_of(p);
        Color c = color_of(p);
        int sq = (c == WHITE) ? s : (s ^ 56); // Flip for black
        
        int val = PieceValue[pt] + Pst[pt][sq];
        if (c == us) score += Value(val);
        else score -= Value(val);
    }
    
    // Tempo bonus
    score += Value(28); // ~28 cp tempo for side to move
    
    // Simple tapering (phase = 24 full material, 0 = endgame)
    // For now, just return middle game score
    (void)phase; // Will be used when tapered eval is added
    
    return score;
}

Value Eval::eval_after_move(const BoardState& pos, Move m, Value currentEval) {
    // Incremental eval update (simplified)
    // In full engine, track incremental PST scores
    (void)pos;
    (void)m;
    return currentEval; // Placeholder: full recompute for now
}

Value Eval::evaluate_pawns(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO; // Detailed pawn structure eval deferred
}

Value Eval::evaluate_knights(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO;
}

Value Eval::evaluate_bishops(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO;
}

Value Eval::evaluate_rooks(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO;
}

Value Eval::evaluate_queens(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO;
}

Value Eval::evaluate_king(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO;
}

Value Eval::evaluate_mobility(const BoardState& pos, Color us) {
    (void)pos;
    (void)us;
    return VALUE_ZERO;
}

int Eval::game_phase(const BoardState& pos) {
    int phase = 0;
    for (int pt = KNIGHT; pt <= QUEEN; ++pt)
        phase += phase_values[pt] * count_bits(pos.pieces(PieceType(pt)));
    return phase;
}

} // namespace Nexus
