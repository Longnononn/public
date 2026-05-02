#pragma once

#include "types.h"
#include "board.h"
#include "move.h"
#include "timeman.h"
#include <atomic>
#include <vector>

namespace Nexus {

struct SearchStack {
    Move* pv;
    int ply;
    Move currentMove;
    Move excludedMove;
    Value staticEval;
    int doubleExtensions;
    int cutOffCnt;
    int16_t* continuationHistory;
};

struct RootMove {
    Move pv[64];
    int pvSize;
    Value score;
    Value previousScore;
    int selDepth;
    u64 nodes;
    int tbRank;
    Value tbScore;
    
    RootMove(Move m) : pvSize(0), score(-VALUE_INFINITE), previousScore(-VALUE_INFINITE), 
                       selDepth(0), nodes(0), tbRank(0), tbScore(VALUE_NONE) {
        pv[0] = m;
    }
    bool operator<(const RootMove& o) const { return score > o.score; }
};

struct SearchLimits {
    int depth;
    u64 nodes;
    int mate;
    int movetime;
    bool infinite;
    int multiPV;  // Number of PV lines to output (1 = single best line)
    std::vector<Move> searchmoves;
    
    SearchLimits() : depth(128), nodes(0), mate(0), movetime(0), 
                     infinite(false), multiPV(1) {}
};

class Search {
public:
    Search();
    
    void clear();
    void start(BoardState& pos, const SearchLimits& limits, bool ponder);
    void stop();
    bool is_running() const { return running.load(); }
    
    Value best_score() const { return rootMoves.empty() ? VALUE_NONE : rootMoves[0].score; }
    Move best_move() const { return rootMoves.empty() ? Move::none() : rootMoves[0].pv[0]; }
    const std::vector<RootMove>& get_root_moves() const { return rootMoves; }
    u64 nodes_searched() const { return nodesSearched; }
    
    // Bench command for testing
    static u64 bench(int depth = 14);
    
    // MultiPV support
    void set_multi_pv(int n) { multiPV = std::max(1, n); }
    int get_multi_pv() const { return multiPV; }
    
private:
    void iterative_deepening(BoardState& pos);
    
    template<bool PvNode>
    Value negamax(BoardState& pos, SearchStack* ss, Value alpha, Value beta, Depth depth, bool cutNode);
    
    template<bool PvNode>
    Value qsearch(BoardState& pos, SearchStack* ss, Value alpha, Value beta);
    
    Value aspiration_window(BoardState& pos, SearchStack* ss, Value prevScore, Depth depth);
    
    void update_quiet_stats(BoardState& pos, SearchStack* ss, Move bestMove, 
                            Move* quiets, int quietCount, int bonus);
    void update_capture_stats(Move m, Move* captures, int captureCount, int bonus);
    
    int reduction(bool i, Depth d, int mn, Value delta, Value rootDelta);
    int stat_bonus(Depth d) { return std::min(268 * d - 346, 1660); }
    int stat_malus(Depth d)  { return -std::min(201 * d - 298, 1152); }
    
    // History tables
    int16_t mainHistory[COLOR_NB][64 * 64];
    int16_t captureHistory[2][6 * 6][64][6];
    int16_t* continuationHistory[2][2];  // [inCheck][capture]
    int16_t contHistoryBuf[2][2][6 * 64 * 6 * 64];  // Flattened
    
    // Search state
    std::vector<RootMove> rootMoves;
    std::atomic<bool> running;
    bool stopOnPonderhit;
    u64 nodesSearched;
    int selDepth;
    Depth completedDepth;
    Value previousScore;
    int multiPV;
    
    // Constants
    static constexpr int QSEARCH_MAX_DEPTH = 12;
    static constexpr int MAX_MULTI_PV = 64;
};

extern Search Searcher;

} // namespace Nexus
