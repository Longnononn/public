#pragma once

#include "types.h"
#include "move.h"
#include <cstring>

namespace Nexus {

// Butterfly history: main heuristic for quiet move ordering
// Indexed by [color][from][to]
class ButterflyHistory {
public:
    ButterflyHistory() { clear(); }
    
    void clear() {
        memset(table, 0, sizeof(table));
    }
    
    // Get history score for a move
    int get(Color c, Move m) const {
        return table[c][m.from_to()];
    }
    
    // Update history with bonus/malus
    void update(Color c, Move m, int bonus) {
        int16_t& entry = table[c][m.from_to()];
        entry += bonus - entry * std::abs(bonus) / 16384;
        entry = std::clamp(entry, (int16_t)-8192, (int16_t)8192);
    }
    
private:
    int16_t table[COLOR_NB][64 * 64];
};

// Capture history: for capture move ordering
// Indexed by [color][captured_piece_type][to_square][attacking_piece_type]
class CaptureHistory {
public:
    CaptureHistory() { clear(); }
    
    void clear() {
        memset(table, 0, sizeof(table));
    }
    
    int get(Color c, PieceType captured, Square to, PieceType attacker) const {
        return table[c][captured][to][attacker];
    }
    
    void update(Color c, PieceType captured, Square to, PieceType attacker, int bonus) {
        int16_t& entry = table[c][captured][to][attacker];
        entry += bonus - entry * std::abs(bonus) / 16384;
        entry = std::clamp(entry, (int16_t)-8192, (int16_t)8192);
    }
    
private:
    int16_t table[COLOR_NB][PIECE_TYPE_NB][SQUARE_NB][PIECE_TYPE_NB];
};

// Continuation history: 2-move sequence context
// Tracks success of moves based on previous moves
// Indexed by [previous_piece][previous_to][current_piece][current_to]
class ContinuationHistory {
public:
    ContinuationHistory() { clear(); }
    
    void clear() {
        memset(table, 0, sizeof(table));
    }
    
    // Get pointer to history table for a given piece/square pair
    // This is used to access the history for current move based on previous
    int16_t* get(Piece prevPiece, Square prevTo) {
        return table[prevPiece][prevTo];
    }
    
    // Get score for a move continuation
    int get(Piece prevPiece, Square prevTo, Piece currPiece, Square currTo) const {
        return table[prevPiece][prevTo][currPiece * 64 + currTo];
    }
    
    // Update continuation history
    void update(Piece prevPiece, Square prevTo, Piece currPiece, Square currTo, int bonus) {
        int16_t& entry = table[prevPiece][prevTo][currPiece * 64 + currTo];
        entry += bonus - entry * std::abs(bonus) / 16384;
        entry = std::clamp(entry, (int16_t)-8192, (int16_t)8192);
    }
    
private:
    // [previous_piece][previous_to][current_piece][current_to]
    // Piece is 0-15, Square is 0-63
    int16_t table[PIECE_NB][SQUARE_NB][PIECE_NB * SQUARE_NB];
};

// Counter-move history: what move to play in response to opponent's move
// Indexed by [piece][to] -> best response move
class CounterMoveHistory {
public:
    CounterMoveHistory() { clear(); }
    
    void clear() {
        memset(table, 0, sizeof(table));
    }
    
    Move get(Piece p, Square to) const {
        return table[p][to];
    }
    
    void set(Piece p, Square to, Move m) {
        table[p][to] = m;
    }
    
private:
    Move table[PIECE_NB][SQUARE_NB];
};

// Combined history manager
class HistoryManager {
public:
    ButterflyHistory butterfly;
    CaptureHistory capture;
    ContinuationHistory continuation[2][2];  // [inCheck][capture]
    CounterMoveHistory counterMove;
    
    void clear() {
        butterfly.clear();
        capture.clear();
        continuation[0][0].clear();
        continuation[0][1].clear();
        continuation[1][0].clear();
        continuation[1][1].clear();
        counterMove.clear();
    }
    
    // Get combined history score for a quiet move
    int get_quiet_score(Color c, Move m, 
                        const ContinuationHistory* cmh,
                        const ContinuationHistory* fmh) const;
    
    // Update all histories after a best move is found
    void update_quiet(Color c, Move bestMove, 
                      const Move* quiets, int quietCount,
                      Depth depth, 
                      Piece prevPiece, Square prevTo,
                      const ContinuationHistory* cmh);
    
    // Update capture history
    void update_capture(Color c, Move m, PieceType captured, Depth depth);
};

// Stat bonus calculation (Stockfish-style)
inline int stat_bonus(Depth d) {
    return std::min(268 * d - 346, 1660);
}

inline int stat_malus(Depth d) {
    return -std::min(201 * d - 298, 1152);
}

} // namespace Nexus
