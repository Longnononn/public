#include "timeman.h"
#include <algorithm>
#include <cmath>

namespace Nexus {

TimeManager Time;

void TimeManager::init(const TimeControl& tc, Color us, int ply) {
    startTime = std::chrono::steady_clock::now();
    nodes = 0;
    
    if (tc.movetime > 0) {
        optimumTime = maximumTime = tc.movetime - 10;
        return;
    }
    
    if (tc.infinite || tc.ponder) {
        optimumTime = maximumTime = INT_MAX;
        return;
    }
    
    int timeLeft = std::max(tc.time[us], 1);
    int inc = tc.increment[us];
    int mtg = (tc.movesToGo > 0) ? tc.movesToGo : MOVE_HORIZON;
    mtg = std::min(mtg, MOVE_HORIZON);
    
    // Base time allocation
    double timeRatio = (inc < timeLeft * 0.4) ? 
        1.0 + inc / timeLeft : 
        1.0 + std::min(double(inc) / timeLeft, 0.4);
    
    optimumTime = int(timeLeft * std::min(0.028 + 0.4 * std::pow(ply / 200.0, 4), timeRatio) / mtg);
    maximumTime = int(timeLeft * std::min(0.76 * timeRatio, MAX_RATIO) / mtg);
    
    // Clamp
    optimumTime = std::min(optimumTime, timeLeft - 100);
    maximumTime = std::min(maximumTime, timeLeft - 100);
    optimumTime = std::max(optimumTime, 1);
    maximumTime = std::max(maximumTime, 1);
}

void TimeManager::update_nodes(u64 n) {
    nodes = n;
}

bool TimeManager::should_stop() const {
    if (maximumTime <= 0) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();
    return elapsed >= maximumTime;
}

bool TimeManager::can_stop_on_fail_low() const {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - startTime).count();
    return elapsed >= optimumTime;
}

} // namespace Nexus
