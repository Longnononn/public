#pragma once

#include "types.h"
#include "board.h"
#include "move.h"
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <mutex>

namespace SearchTelemetry {

// ======== Telemetry Event Types ========

enum class EventType {
    SEARCH_START,
    SEARCH_END,
    NODE_EXPANDED,
    PRUNING_DECISION,
    CUTOFF,
    TT_HIT,
    NULL_MOVE,
    FUTILITY,
    RAZORING,
    PROBCUT,
    LMR,
    EXTENSION,
    ASYNC_EVAL_SUBMIT,
    ASYNC_EVAL_COMPLETE,
    NNUE_EVAL,
    SYZYGY_PROBE,
    HARD_POSITION,
    ERROR
};

// ======== Telemetry Entry ========
// Single log entry for search analysis

struct TelemetryEntry {
    EventType type;
    int thread_id;
    Depth depth;
    int ply;
    
    // Position info
    Key position_key;
    std::string fen;
    
    // Decision info
    Value score;
    Value alpha;
    Value beta;
    Move move;
    
    // Timing
    uint64_t timestamp_us;
    uint64_t duration_us;
    
    // Additional context (JSON-like string)
    std::string context;
    
    // Stack trace for debugging
    std::vector<std::string> call_stack;
    
    TelemetryEntry()
        : type(EventType::ERROR)
        , thread_id(0)
        , depth(0)
        , ply(0)
        , position_key(0)
        , score(VALUE_NONE)
        , alpha(VALUE_NONE)
        , beta(VALUE_NONE)
        , timestamp_us(0)
        , duration_us(0) {}
};

// ======== Pruning Decision Record ========
// Records WHY a decision was made (for ML explainability)

struct PruningDecision {
    enum class Reason {
        NULL_MOVE_VERIFIED,      // Null move searched, verified
        FUTILITY_MARGIN,          // Score below futility margin
        RAZORING_MARGIN,         // Shallow depth, score below razor margin
        PROBCUT_FAILED,          // ProbCut search failed
        LMR_REDUCTION,           // Late Move Reduction applied
        LMR_REJECTED,            // LMR rejected (searched full depth)
        EXTENSION_TACTICAL,      // Tactical extension
        EXTENSION_CHECK,         // Check extension
        EXTENSION_PAWN_PUSH,     // Pawn push extension
        EXTENSION_RECAPTURE,     // Recapture extension
        HISTORY_PRUNE,           // History-based pruning
        COUNTERMOVE_PRUNE,       // Countermove-based pruning
        SEE_PRUNE,               // Static Exchange Evaluation prune
        TT_CUTOFF,               // Transposition table cutoff
        KILLER_MOVE_PRUNE,       // Killer move pruning
        NNUE_EVAL_LOW,           // NNUE evaluation low
        NNUE_EVAL_HIGH,          // NNUE evaluation high
        CUSTOM                   // Custom ML-based pruning
    };
    
    Reason reason;
    Move move;
    Value score_before;
    Value score_after;
    Depth depth;
    bool accepted;  // Was the pruning accepted?
    
    // ML model weights (if ML-based pruning)
    float model_confidence;
    std::vector<float> feature_vector;
    
    std::string explain() const;
};

// ======== ML Pruning Explainability ========
// Records feature importance for ML-based pruning decisions

struct PruningFeatures {
    // Position features
    float material_diff;      // Material difference
    float king_safety;        // King safety score
    float pawn_structure;     // Pawn structure score
    float piece_activity;     // Piece activity score
    float space_control;      // Space control
    
    // Search context
    float depth_remaining;
    float alpha_beta_distance;
    float node_count;
    float tt_hit_rate;
    
    // Move-specific
    float move_history_score;
    float move_counter_score;
    float move_see_score;
    
    // NNUE prediction
    float nnue_eval;
    float nnue_confidence;
    
    // Model output
    float prune_probability;
    float actual_decision;    // 0 or 1 (prune or keep)
    
    std::string to_json() const;
};

// ======== Telemetry Logger ========
// Central logging system for search telemetry

class TelemetryLogger {
public:
    static TelemetryLogger& instance();
    
    // Enable/disable logging
    void enable(bool enabled = true) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    // Log an event
    void log(const TelemetryEntry& entry);
    
    // Convenience methods for common events
    void log_search_start(const BoardState& pos, Depth depth);
    void log_search_end(int nodes, Value score, uint64_t duration_us);
    void log_node_expanded(const BoardState& pos, Depth depth, int ply);
    void log_pruning(const PruningDecision& decision, const BoardState& pos);
    void log_cutoff(const BoardState& pos, Move move, Value score, Depth depth);
    void log_tt_hit(const BoardState& pos, Value score, Depth depth);
    void log_nnue_eval(const BoardState& pos, Value score, uint64_t latency_us);
    
    // ML-specific logging
    void log_ml_pruning(const PruningDecision& decision, 
                         const PruningFeatures& features);
    
    // Export logs
    void export_to_file(const std::string& filename) const;
    void export_pruning_decisions(const std::string& filename) const;
    
    // Statistics
    size_t total_entries() const;
    size_t entries_by_type(EventType type) const;
    
    // Clear logs
    void clear();
    
private:
    TelemetryLogger() : enabled_(false) {}
    ~TelemetryLogger();
    
    bool enabled_;
    std::vector<TelemetryEntry> entries_;
    std::vector<PruningDecision> pruning_decisions_;
    
    mutable std::mutex mutex_;
    
    uint64_t get_timestamp_us() const;
};

// ======== Search Visualizer ========
// Generates visual representation of search tree for debugging

class SearchVisualizer {
public:
    struct Node {
        Key key;
        Move move;
        Value score;
        int depth;
        int visits;
        std::vector<Node> children;
    };
    
    // Build tree from telemetry
    Node build_tree(const std::vector<TelemetryEntry>& entries);
    
    // Export tree to DOT format (GraphViz)
    void export_dot(const Node& root, const std::string& filename);
    
    // Export tree to JSON
    void export_json(const Node& root, const std::string& filename);
    
    // Generate ASCII tree visualization
    std::string to_ascii(const Node& root, int max_depth = 6);
    
private:
    std::string node_to_string(const Node& node) const;
};

// ======== Pruning Analyzer ========
// Analyzes pruning decisions to find regressions

class PruningAnalyzer {
public:
    // Analyze pruning patterns
    struct Analysis {
        int total_prunes;
        int total_searches;
        float prune_rate;
        
        // Breakdown by reason
        std::map<PruningDecision::Reason, int> reason_counts;
        
        // Accuracy: was pruning correct?
        int correct_prunes;      // Pruning led to same result
        int incorrect_prunes;    // Pruning missed better move
        float accuracy;
        
        // Regression detection
        float recent_accuracy;
        float accuracy_trend;   // Positive = improving, negative = regressing
        
        std::string summary() const;
    };
    
    // Analyze recent pruning decisions
    Analysis analyze_recent(int window = 1000);
    
    // Compare two versions
    void compare_versions(const std::string& old_logs, 
                         const std::string& new_logs);
    
    // Detect suspicious patterns (potential bugs)
    std::vector<std::string> detect_anomalies();
    
    // Generate recommendations
    std::vector<std::string> generate_recommendations();
};

// ======== Macro for easy logging ========

#define TELEMETRY_LOG(type, ...) \
    do { \
        if (SearchTelemetry::TelemetryLogger::instance().is_enabled()) { \
            SearchTelemetry::TelemetryEntry entry; \
            entry.type = SearchTelemetry::type; \
            entry.thread_id = 0; \
            entry.timestamp_us = SearchTelemetry::TelemetryLogger::instance().get_timestamp_us(); \
            __VA_ARGS__; \
            SearchTelemetry::TelemetryLogger::instance().log(entry); \
        } \
    } while(0)

} // namespace SearchTelemetry
