#pragma once

#include "types.h"
#include "board.h"
#include "move.h"
#include <string>

namespace Syzygy {

// WDL probing results
enum WDLResult {
    WDL_LOSS = -2,
    WDL_BLESSED_LOSS = -1,
    WDL_DRAW = 0,
    WDL_CURSED_WIN = 1,
    WDL_WIN = 2,
    WDL_FAIL = 100  // Probe failed
};

// DTZ probing results
enum DTZResult {
    DTZ_ZERO = 0,   // Win or draw by 50-move rule
    DTZ_DRAW = 1,   // Draw
    DTZ_MATE = 32767  // Mate
};

// Syzygy tablebase manager
class Tablebases {
public:
    Tablebases();
    ~Tablebases();
    
    // Initialize with path to .rtbw and .rtbz files
    bool init(const std::string& path);
    void close();
    
    // Check if tablebases are available
    bool is_initialized() const { return initialized; }
    int max_pieces() const { return maxPieces; }
    
    // Probe WDL (Win/Draw/Loss)
    // Returns WDL value without 50-move rule consideration
    WDLResult probe_wdl(const BoardState& pos) const;
    
    // Probe DTZ (Distance to Zeroing)
    // Returns distance to 50-move rule reset or mate
    int probe_dtz(const BoardState& pos) const;
    
    // Root probe - find best move from root position
    // Returns true if position found in TB, fills bestMove with optimal
    bool root_probe(const BoardState& pos, Move& bestMove, Value& bestScore) const;
    
    // Root probe WDL - returns true if all moves are in TB
    bool root_probe_wdl(const BoardState& pos, Move& bestMove, Value& bestScore) const;
    
    // Check if position is in tablebase territory
    bool is_tb_position(const BoardState& pos) const;
    
    // Filter moves by tablebase result
    void filter_root_moves(const BoardState& pos, std::vector<Move>& moves) const;
    
private:
    bool initialized = false;
    int maxPieces = 0;
    std::string tbPath;
    
    // Internal probing functions
    int probe_table(const BoardState& pos, bool wdl) const;
    int map_score(WDLResult wdl, int dtz) const;
};

// Global instance
extern Tablebases TB;

// Helper functions
inline bool can_probe_wdl(const BoardState& pos) {
    // Only probe if no pawns on 7th/2nd rank (STB limitation)
    return true;  // Simplified - full check in implementation
}

// Score mapping for search integration
inline Value wdl_to_value(WDLResult wdl) {
    switch (wdl) {
        case WDL_WIN: return VALUE_MATE - 2;
        case WDL_CURSED_WIN: return VALUE_DRAW + 200;
        case WDL_DRAW: return VALUE_DRAW;
        case WDL_BLESSED_LOSS: return VALUE_DRAW - 200;
        case WDL_LOSS: return -VALUE_MATE + 2;
        default: return VALUE_NONE;
    }
}

inline bool is_dtz_zero(int dtz) {
    return dtz == 0;
}

} // namespace Syzygy
