#pragma once

#include "types.h"

namespace Nexus {

// Advanced Late Move Reduction (LMR) tables
// These tables provide reduction amounts based on depth and move number

class Reductions {
public:
    // Initialize reduction tables
    static void init();
    
    // Get reduction amount for a given depth and move count
    // depth: current search depth
    // moveCount: number of moves already searched at this node
    // improving: true if static eval improved from previous ply
    // cutNode: true if we're expected to fail low (cut node)
    // returns: number of plies to reduce
    static Depth get_reduction(Depth depth, int moveCount, bool improving, bool cutNode);
    
    // Get SEE-based reduction adjustment
    // Returns additional reduction for bad captures
    static int see_reduction(int seeValue);
    
    // Get history-based reduction adjustment
    // historyScore: from history heuristic (-8192 to 8192)
    // returns: reduction adjustment (can be negative)
    static int history_reduction(int historyScore);
    
    // Constants for reduction calculation
    static constexpr int LmrDepthBits = 10;
    static constexpr int LmrMoveBits = 10;
    static constexpr double LmrScale = 2.0;
    static constexpr double LmrDivisor = 2.5;
    
private:
    // Precomputed reduction tables
    static int ReductionTable[2][2][64][64]; // [pv][improving][depth][moveCount]
    
    // Helper to compute base reduction
    static int base_reduction(Depth d, int m);
};

// ProbCut and MultiCut pruning thresholds
struct PruningThresholds {
    // ProbCut: beta cutoff prediction
    static constexpr int ProbCutMargin = 176;
    static constexpr Depth ProbCutDepth = Depth(5);
    
    // MultiCut: multiple fail-high threshold
    static constexpr int MultiCutCount = 3;
    static constexpr Value MultiCutMargin = Value(200);
    
    // Internal Iterative Deepening (IID) depth reduction
    static constexpr Depth IidDepthReduction = Depth(4);
    
    // Futility pruning margins indexed by depth
    static constexpr int FutilityMargins[8] = {
        0, 100, 200, 400, 600, 800, 1000, 1200
    };
    
    // Razoring margins indexed by depth
    static constexpr int RazoringMargins[5] = {
        0, 200, 400, 600, 800
    };
};

// Search constants for modern chess engines
struct SearchParams {
    // Aspiration window initial delta and growth
    static constexpr Value AspirationDelta = Value(17);
    static constexpr double AspirationGrowth = 1.5;
    
    // Singular extension thresholds
    static constexpr Value SingularMargin = Value(2);
    static constexpr Depth SingularDepth = Depth(7);
    static constexpr Depth SingularDepthMin = Depth(4);
    
    // Double extension limit to avoid search explosion
    static constexpr int DoubleExtensionMax = 6;
    
    // Null move pruning constants
    static constexpr Depth NullMoveMinDepth = Depth(3);
    static constexpr int NullMoveBaseR = 4;
    static constexpr int NullMoveDepthDivisor = 3;
    static constexpr int NullMoveEvalDivisor = 200;
    
    // Reverse futility pruning
    static constexpr int RfpMaxDepth = 8;
    static constexpr int RfpMargin = 168;
    
    // Late move pruning thresholds (LMP)
    static int lmp_threshold(Depth d, bool improving);
};

// Inline helper for LMP threshold calculation
inline int SearchParams::lmp_threshold(Depth d, bool improving) {
    // Base threshold: 3 + depth^2 for improving, (3 + depth^2) / 2 otherwise
    int base = 3 + d * d;
    return improving ? base : base / 2;
}

} // namespace Nexus
