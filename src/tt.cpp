#include "tt.h"
#include "simd.h"  // For PREFETCH macro
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace Nexus {

TranspositionTable::TranspositionTable() : clusterCount(0), table(nullptr), generation(0) {}

TranspositionTable::~TranspositionTable() {
    if (table) {
        free(table);
    }
}

void TranspositionTable::resize(size_t mbSize) {
    if (table) free(table);
    
    // Cluster count must be power of 2
    size_t clusters = 1;
    while (clusters * sizeof(TTCluster) < mbSize * 1024 * 1024)
        clusters <<= 1;
    
    clusterCount = clusters;
    table = (TTCluster*)calloc(clusterCount, sizeof(TTCluster));
    if (!table) {
        clusterCount = 0;
        return;
    }
    
    clear();
}

void TranspositionTable::clear() {
    if (table)
        std::memset(table, 0, clusterCount * sizeof(TTCluster));
    generation = 0;
}

void TranspositionTable::new_search() {
    generation++;
}

TTEntry* TranspositionTable::probe(Key key, bool& found) const {
    if (!table) {
        found = false;
        return nullptr;
    }
    
    TTCluster* cluster = &table[index(key)];
    u16 key16 = (u16)((u64)key >> 48);
    
    for (int i = 0; i < TTCluster::SIZE; ++i) {
        if (cluster->entries[i].key16 == key16) {
            found = true;
            return &cluster->entries[i];
        }
    }
    
    found = false;
    return &cluster->entries[0];
}

void TranspositionTable::store(Key key, Value v, bool pv, Bound b, Depth d, Move m, Value ev) {
    size_t idx = index(key);
    TTCluster* cluster = &table[idx];
    
    u16 key16 = (u16)((u64)key >> 48);
    TTEntry* replace = nullptr;
    
    // Find best entry to replace using depth-preferred aging strategy
    for (int i = 0; i < TTCluster::SIZE; ++i) {
        TTEntry* entry = &cluster->entries[i];
        
        // Always replace if exact key match
        if (entry->key16 == key16) {
            replace = entry;
            break;
        }
        
        // Prefer empty entries
        if (entry->is_empty()) {
            replace = entry;
            break;
        }
        
        // Calculate replacement score (lower = better to replace)
        // Age difference: older entries are better to replace
        int ageDiff = (256 + generation - entry->generation) & 0xFF;
        // Depth preference: shallow entries are better to replace
        int depthDiff = 4 * (256 + d - entry->depth);
        
        int replaceScore = ageDiff - depthDiff;
        
        if (!replace) {
            replace = entry;
        } else {
            int bestScore = ((256 + generation - replace->generation) & 0xFF) 
                          - 4 * (256 + d - replace->depth);
            if (replaceScore < bestScore) {
                replace = entry;
            }
        }
    }
    
    // Preserve existing move if we don't have a better one
    if (!m.is_ok() && replace->key16 == key16 && Move(replace->move).is_ok()) {
        m = Move(replace->move);
    }
    
    // Don't overwrite deeper entries with same generation unless exact match
    if (replace->key16 != key16 && replace->depth > d && 
        replace->generation == generation && !pv) {
        return;
    }
    
    // Store entry
    replace->key16 = key16;
    replace->move = m;
    replace->value = (i16)v;
    replace->eval = (i16)ev;
    replace->depth = (u8)d;
    replace->bound = (u8)b;
    replace->generation = generation;
}

int TranspositionTable::hashfull() const {
    if (!table) return 0;
    int used = 0;
    for (size_t i = 0; i < 1000 && i < clusterCount; ++i)
        for (int j = 0; j < TTCluster::SIZE; ++j)
            if (table[i].entries[j].generation == generation)
                used++;
    return used / TTCluster::SIZE;
}

void TranspositionTable::prefetch(Key key) const {
    size_t idx = index(key);
    PREFETCH(&table[idx]);
}

TranspositionTable TT;

} // namespace Nexus
