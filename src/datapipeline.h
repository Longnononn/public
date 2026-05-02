#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>

namespace DataPipeline {

// Filter types for training data
enum FilterType {
    FILTER_NONE          = 0,
    FILTER_DUPLICATE     = 1 << 0,  // Remove duplicate positions
    FILTER_FORCED_MATE   = 1 << 1,  // Remove forced mates
    FILTER_QUIET_ONLY    = 1 << 2,  // Keep only quiet positions
    FILTER_CHECK         = 1 << 3,  // Remove positions in check
    FILTER_ENDGAME       = 1 << 4,  // Remove pure endgame
    FILTER_OPENING       = 1 << 5,  // Remove early opening
    FILTER_LOW_ENTROPY   = 1 << 6,  // Remove low-entropy (trivial) positions
    FILTER_HIGH_SCORE    = 1 << 7,  // Remove positions with |score| > threshold
};

// Position with metadata for filtering
struct FilteredPosition {
    std::string fen;
    Value score;
    int result;
    int gamePhase;
    int ply;
    int move50;
    bool inCheck;
    std::string gameId;
    
    // Derived metrics
    double entropy;           // Position complexity
    bool isQuiet;             // No tactical threats
    bool isSharp;             // High tactical content
    int materialImbalance;      // Material difference
    int numLegalMoves;        // Branching factor
};

// Hash for deduplication
struct PositionHash {
    uint64_t hash;
    
    bool operator==(const PositionHash& other) const {
        return hash == other.hash;
    }
};

struct PositionHashHasher {
    size_t operator()(const PositionHash& h) const {
        return std::hash<uint64_t>{}(h.hash);
    }
};

// Data quality metrics
struct QualityMetrics {
    u64 totalPositions;
    u64 afterFiltering;
    u64 duplicatesRemoved;
    u64 forcedMatesRemoved;
    u64 quietPositions;
    u64 tacticalPositions;
    u64 endgamePositions;
    u64 openingPositions;
    u64 lowEntropyRemoved;
    
    double avgEntropy;
    double scoreVariance;
    double resultBalance;  // 0.33 each = perfect
    
    void print() const;
};

// Main pipeline class
class DataPipeline {
public:
    DataPipeline(uint32_t filters = FILTER_DUPLICATE | FILTER_FORCED_MATE | FILTER_LOW_ENTROPY);
    
    // Configuration
    void set_filters(uint32_t f) { activeFilters = f; }
    void set_entropy_threshold(double t) { entropyThreshold = t; }
    void set_score_threshold(Value v) { scoreThreshold = v; }
    void set_min_ply(int p) { minPly = p; }
    void set_max_ply(int p) { maxPly = p; }
    void set_opening_ply(int p) { openingPly = p; }
    
    // Processing pipeline
    std::vector<FilteredPosition> process(
        const std::vector<Training::TrainingPosition>& raw);
    
    // Individual filters
    std::vector<FilteredPosition> remove_duplicates(
        std::vector<FilteredPosition>& data);
    std::vector<FilteredPosition> remove_forced_mates(
        std::vector<FilteredPosition>& data);
    std::vector<FilteredPosition> filter_by_entropy(
        std::vector<FilteredPosition>& data);
    std::vector<FilteredPosition> filter_by_phase(
        std::vector<FilteredPosition>& data);
    std::vector<FilteredPosition> balance_results(
        std::vector<FilteredPosition>& data);
    
    // Advanced filtering
    std::vector<FilteredPosition> filter_sharp_positions(
        std::vector<FilteredPosition>& data, double percentile);
    std::vector<FilteredPosition> filter_quiet_positions(
        std::vector<FilteredPosition>& data);
    
    // Hard example mining
    std::vector<FilteredPosition> mine_hard_positions(
        const std::vector<FilteredPosition>& enginePositions,
        const std::vector<FilteredPosition>& referencePositions);
    
    // Quality analysis
    QualityMetrics analyze(const std::vector<FilteredPosition>& data) const;
    
    // Export
    void export_plain(const std::string& filename,
                      const std::vector<FilteredPosition>& data);
    void export_binpack(const std::string& filename,
                        const std::vector<FilteredPosition>& data);
    void export_for_nnue(const std::string& filename,
                         const std::vector<FilteredPosition>& data);
    
private:
    uint32_t activeFilters;
    double entropyThreshold;
    Value scoreThreshold;
    int minPly;
    int maxPly;
    int openingPly;
    
    // Deduplication hash set
    std::unordered_set<PositionHash, PositionHashHasher> seenPositions;
    
    // Helper functions
    double compute_entropy(const FilteredPosition& pos) const;
    bool is_forced_mate(const FilteredPosition& pos) const;
    PositionHash compute_hash(const std::string& fen) const;
    int count_material(const std::string& fen) const;
};

// Stream processing for large datasets
class StreamingPipeline {
public:
    StreamingPipeline(const std::string& inputFile,
                      const std::string& outputFile,
                      uint32_t filters);
    
    void process_stream();
    u64 get_processed_count() const { return processedCount; }
    u64 get_written_count() const { return writtenCount; }
    
private:
    std::string inputPath;
    std::string outputPath;
    uint32_t filters;
    u64 processedCount;
    u64 writtenCount;
    
    std::unordered_set<PositionHash, PositionHashHasher> seenCache;
    static constexpr size_t CACHE_SIZE = 1000000;  // 1M positions in memory
};

// Position sampling for diversity
class DiversitySampler {
public:
    // Reservoir sampling with diversity constraints
    std::vector<FilteredPosition> sample_diverse(
        const std::vector<FilteredPosition>& data,
        size_t targetSize);
    
    // Stratified sampling by game phase
    std::vector<FilteredPosition> sample_stratified(
        const std::vector<FilteredPosition>& data,
        size_t perCategory);
    
    // K-means-like clustering for position types
    std::vector<std::vector<FilteredPosition>> cluster_positions(
        const std::vector<FilteredPosition>& data,
        int numClusters);
};

} // namespace DataPipeline
