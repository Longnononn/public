#pragma once

#include "types.h"
#include "move.h"
#include <atomic>
#include <cstring>

namespace Nexus {

// Compact 10-byte TT entry (with 6 bytes padding to 16 for alignment)
struct TTEntry {
    u16 key16;       // Upper 16 bits of Zobrist key
    Move move;       // Best move from this node
    i16 value;       // Search result value
    i16 eval;        // Static evaluation cached
    u8 depth;        // Depth searched
    u8 bound;        // BOUND_UPPER, BOUND_LOWER, BOUND_EXACT
    u8 generation;     // Age counter for replacement
    u8 padding[5];   // Pad to 16 bytes
    
    bool is_empty() const { return key16 == 0; }
};

static_assert(sizeof(TTEntry) == 16, "TTEntry must be 16 bytes");

struct TTCluster {
    static constexpr int SIZE = 3;
    TTEntry entries[SIZE];
};

static_assert(sizeof(TTCluster) == 48, "TTCluster must be 48 bytes");

class TranspositionTable {
public:
    TranspositionTable();
    ~TranspositionTable();
    
    void resize(size_t mbSize);
    void clear();
    void new_search();
    
    // Probe — returns pointer to entry or nullptr
    TTEntry* probe(Key key, bool& found) const;
    
    // Store
    void store(Key key, Value v, bool pv, Bound b, Depth d, Move m, Value ev);
    
    // Hashfull (permill)
    int hashfull() const;
    
    // Prefetch
    void prefetch(Key key) const;
    
private:
    size_t clusterCount;
    TTCluster* table;
    u8 generation;
    
    size_t index(Key key) const { return (size_t)key & (clusterCount - 1); }
};

extern TranspositionTable TT;

} // namespace Nexus
