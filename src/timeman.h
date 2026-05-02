#pragma once

#include "types.h"
#include <chrono>

namespace Nexus {

struct TimeControl {
    int time[COLOR_NB];       // ms remaining
    int increment[COLOR_NB]; // ms increment
    int movesToGo;
    int depth;
    int nodes;
    int mate;
    int movetime;
    bool infinite;
    bool ponder;
};

class TimeManager {
public:
    void init(const TimeControl& tc, Color us, int ply);
    void update_nodes(u64 nodes);
    bool should_stop() const;
    bool can_stop_on_fail_low() const;
    
    int time_for_move() const { return optimumTime; }
    int max_time() const { return maximumTime; }
    
private:
    int optimumTime;
    int maximumTime;
    std::chrono::steady_clock::time_point startTime;
    u64 nodes;
    
    static constexpr int MOVE_HORIZON = 50;
    static constexpr double MAX_RATIO = 7.09;
    static constexpr double STEAL_RATIO = 0.34;
};

extern TimeManager Time;

} // namespace Nexus
