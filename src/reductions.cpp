#include "reductions.h"
#include <cmath>

namespace Nexus {

int Reductions::ReductionTable[2][2][64][64];

void Reductions::init() {
    for (int pv = 0; pv < 2; ++pv) {
        for (int imp = 0; imp < 2; ++imp) {
            for (int d = 1; d < 64; ++d) {
                for (int m = 1; m < 64; ++m) {
                    double r = std::log(d) * std::log(m) / LmrDivisor;
                    
                    // PV nodes get less reduction
                    if (pv) r -= 1.0;
                    
                    // Improving positions get less reduction
                    if (!imp) r += 1.0;
                    
                    // Scale and clamp
                    int reduction = static_cast<int>(std::round(r * LmrScale));
                    ReductionTable[pv][imp][d][m] = std::max(0, std::min(reduction, d));
                }
            }
        }
    }
}

Depth Reductions::get_reduction(Depth depth, int moveCount, bool improving, bool cutNode) {
    // Clamp to table bounds
    int d = std::min(static_cast<int>(depth), 63);
    int m = std::min(moveCount, 63);
    
    // 0 = non-PV, 1 = PV (simplified - caller knows if it's PV)
    int pvIndex = 1; // Assume PV for safety
    int impIndex = improving ? 1 : 0;
    
    int reduction = ReductionTable[pvIndex][impIndex][d][m];
    
    // Cut nodes get more reduction
    if (cutNode) reduction += 1;
    
    // Ensure we don't reduce below 0
    return Depth(std::max(0, std::min(reduction, d - 1)));
}

int Reductions::see_reduction(int seeValue) {
    // If SEE is very negative, reduce more
    if (seeValue < -300) return 2;
    if (seeValue < -100) return 1;
    if (seeValue >= 0) return -1; // Good captures: reduce less
    return 0;
}

int Reductions::history_reduction(int historyScore) {
    // History score ranges from -8192 to 8192
    // High history = good move, reduce less (negative adjustment)
    // Low history = bad move, reduce more
    
    if (historyScore > 6000) return -2;
    if (historyScore > 3000) return -1;
    if (historyScore < -6000) return 2;
    if (historyScore < -3000) return 1;
    return 0;
}

} // namespace Nexus
