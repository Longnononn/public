#include "search_telemetry.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace SearchTelemetry {

// ============== TelemetryLogger ==============

TelemetryLogger& TelemetryLogger::instance() {
    static TelemetryLogger logger;
    return logger;
}

TelemetryLogger::~TelemetryLogger() {
    // Auto-export on destruction
    if (enabled_ && !entries_.empty()) {
        export_to_file("telemetry_last_run.log");
        export_pruning_decisions("pruning_decisions_last_run.json");
    }
}

void TelemetryLogger::log(const TelemetryEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(entry);
    
    // Keep memory bounded
    if (entries_.size() > 1000000) {
        entries_.erase(entries_.begin(), entries_.begin() + 100000);
    }
}

void TelemetryLogger::log_search_start(const BoardState& pos, Depth depth) {
    TelemetryEntry entry;
    entry.type = EventType::SEARCH_START;
    entry.depth = depth;
    entry.ply = 0;
    entry.position_key = pos.get_key();
    entry.fen = pos.get_fen();
    entry.timestamp_us = get_timestamp_us();
    
    log(entry);
}

void TelemetryLogger::log_search_end(int nodes, Value score, uint64_t duration_us) {
    TelemetryEntry entry;
    entry.type = EventType::SEARCH_END;
    entry.score = score;
    entry.duration_us = duration_us;
    entry.timestamp_us = get_timestamp_us();
    
    std::ostringstream oss;
    oss << "{\"nodes\":" << nodes << "}";
    entry.context = oss.str();
    
    log(entry);
}

void TelemetryLogger::log_node_expanded(const BoardState& pos, Depth depth, int ply) {
    TelemetryEntry entry;
    entry.type = EventType::NODE_EXPANDED;
    entry.depth = depth;
    entry.ply = ply;
    entry.position_key = pos.get_key();
    entry.timestamp_us = get_timestamp_us();
    
    log(entry);
}

void TelemetryLogger::log_pruning(const PruningDecision& decision, const BoardState& pos) {
    std::lock_guard<std::mutex> lock(mutex_);
    pruning_decisions_.push_back(decision);
    
    TelemetryEntry entry;
    entry.type = EventType::PRUNING_DECISION;
    entry.move = decision.move;
    entry.score = decision.score_before;
    entry.depth = decision.depth;
    entry.position_key = pos.get_key();
    entry.fen = pos.get_fen();
    entry.timestamp_us = get_timestamp_us();
    
    std::ostringstream oss;
    oss << "{\"reason\":\"" << decision.explain() << "\",\"accepted\":" 
        << (decision.accepted ? "true" : "false") << "}";
    entry.context = oss.str();
    
    entries_.push_back(entry);
    
    if (entries_.size() > 1000000) {
        entries_.erase(entries_.begin(), entries_.begin() + 100000);
    }
}

void TelemetryLogger::log_cutoff(const BoardState& pos, Move move, Value score, Depth depth) {
    TelemetryEntry entry;
    entry.type = EventType::CUTOFF;
    entry.move = move;
    entry.score = score;
    entry.depth = depth;
    entry.position_key = pos.get_key();
    entry.timestamp_us = get_timestamp_us();
    
    log(entry);
}

void TelemetryLogger::log_tt_hit(const BoardState& pos, Value score, Depth depth) {
    TelemetryEntry entry;
    entry.type = EventType::TT_HIT;
    entry.score = score;
    entry.depth = depth;
    entry.position_key = pos.get_key();
    entry.timestamp_us = get_timestamp_us();
    
    log(entry);
}

void TelemetryLogger::log_nnue_eval(const BoardState& pos, Value score, uint64_t latency_us) {
    TelemetryEntry entry;
    entry.type = EventType::NNUE_EVAL;
    entry.score = score;
    entry.position_key = pos.get_key();
    entry.duration_us = latency_us;
    entry.timestamp_us = get_timestamp_us();
    
    std::ostringstream oss;
    oss << "{\"latency_us\":" << latency_us << "}";
    entry.context = oss.str();
    
    log(entry);
}

void TelemetryLogger::log_ml_pruning(const PruningDecision& decision,
                                       const PruningFeatures& features) {
    std::lock_guard<std::mutex> lock(mutex_);
    pruning_decisions_.push_back(decision);
    
    TelemetryEntry entry;
    entry.type = EventType::PRUNING_DECISION;
    entry.move = decision.move;
    entry.score = decision.score_before;
    entry.depth = decision.depth;
    entry.timestamp_us = get_timestamp_us();
    
    entry.context = features.to_json();
    
    entries_.push_back(entry);
}

void TelemetryLogger::export_to_file(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "# Nexus Infinite Search Telemetry\n";
    file << "# Entries: " << entries_.size() << "\n\n";
    
    for (const auto& entry : entries_) {
        file << "[" << static_cast<int>(entry.type) << "] "
             << "thread=" << entry.thread_id
             << " depth=" << entry.depth
             << " ply=" << entry.ply
             << " score=" << entry.score
             << " move=" << entry.move.to_string()
             << " key=" << entry.position_key
             << " time=" << entry.timestamp_us
             << " dur=" << entry.duration_us;
        
        if (!entry.context.empty()) {
            file << " " << entry.context;
        }
        
        if (!entry.fen.empty()) {
            file << " fen=" << entry.fen.substr(0, 40);
        }
        
        file << "\n";
    }
}

void TelemetryLogger::export_pruning_decisions(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "[\n";
    
    for (size_t i = 0; i < pruning_decisions_.size(); ++i) {
        const auto& decision = pruning_decisions_[i];
        
        file << "  {\n";
        file << "    \"reason\": \"" << decision.explain() << "\",\n";
        file << "    \"move\": \"" << decision.move.to_string() << "\",\n";
        file << "    \"score_before\": " << decision.score_before << ",\n";
        file << "    \"score_after\": " << decision.score_after << ",\n";
        file << "    \"depth\": " << decision.depth << ",\n";
        file << "    \"accepted\": " << (decision.accepted ? "true" : "false") << ",\n";
        file << "    \"confidence\": " << decision.model_confidence << "\n";
        file << "  }" << (i < pruning_decisions_.size() - 1 ? "," : "") << "\n";
    }
    
    file << "]\n";
}

size_t TelemetryLogger::total_entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

size_t TelemetryLogger::entries_by_type(EventType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::count_if(entries_.begin(), entries_.end(),
        [type](const TelemetryEntry& e) { return e.type == type; });
}

void TelemetryLogger::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    pruning_decisions_.clear();
}

uint64_t TelemetryLogger::get_timestamp_us() const {
    auto now = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    return static_cast<uint64_t>(us);
}

// ============== PruningDecision ==============

std::string PruningDecision::explain() const {
    switch (reason) {
        case Reason::NULL_MOVE_VERIFIED: return "null_move_verified";
        case Reason::FUTILITY_MARGIN: return "futility_margin";
        case Reason::RAZORING_MARGIN: return "razoring_margin";
        case Reason::PROBCUT_FAILED: return "probcut_failed";
        case Reason::LMR_REDUCTION: return "lmr_reduction";
        case Reason::LMR_REJECTED: return "lmr_rejected";
        case Reason::EXTENSION_TACTICAL: return "extension_tactical";
        case Reason::EXTENSION_CHECK: return "extension_check";
        case Reason::EXTENSION_PAWN_PUSH: return "extension_pawn_push";
        case Reason::EXTENSION_RECAPTURE: return "extension_recapture";
        case Reason::HISTORY_PRUNE: return "history_prune";
        case Reason::COUNTERMOVE_PRUNE: return "countermove_prune";
        case Reason::SEE_PRUNE: return "see_prune";
        case Reason::TT_CUTOFF: return "tt_cutoff";
        case Reason::KILLER_MOVE_PRUNE: return "killer_move_prune";
        case Reason::NNUE_EVAL_LOW: return "nnue_eval_low";
        case Reason::NNUE_EVAL_HIGH: return "nnue_eval_high";
        case Reason::CUSTOM: return "custom_ml";
        default: return "unknown";
    }
}

// ============== PruningFeatures ==============

std::string PruningFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{"
         << "\"material_diff\":" << material_diff << ","
         << "\"king_safety\":" << king_safety << ","
         << "\"pawn_structure\":" << pawn_structure << ","
         << "\"piece_activity\":" << piece_activity << ","
         << "\"space_control\":" << space_control << ","
         << "\"depth_remaining\":" << depth_remaining << ","
         << "\"alpha_beta_distance\":" << alpha_beta_distance << ","
         << "\"node_count\":" << node_count << ","
         << "\"tt_hit_rate\":" << tt_hit_rate << ","
         << "\"move_history_score\":" << move_history_score << ","
         << "\"move_counter_score\":" << move_counter_score << ","
         << "\"move_see_score\":" << move_see_score << ","
         << "\"nnue_eval\":" << nnue_eval << ","
         << "\"nnue_confidence\":" << nnue_confidence << ","
         << "\"prune_probability\":" << prune_probability << ","
         << "\"actual_decision\":" << actual_decision
         << "}";
    return oss.str();
}

// ============== SearchVisualizer ==============

SearchVisualizer::Node SearchVisualizer::build_tree(
    const std::vector<TelemetryEntry>& entries) {
    
    Node root;
    // Would build tree from telemetry entries
    // Simplified: just return empty root
    return root;
}

void SearchVisualizer::export_dot(const Node& root, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "digraph SearchTree {\n";
    // Would export tree to DOT format
    file << "}\n";
}

void SearchVisualizer::export_json(const Node& root, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "{\n  \"nodes\": [\n";
    // Would export tree to JSON
    file << "  ]\n}\n";
}

std::string SearchVisualizer::to_ascii(const Node& root, int max_depth) {
    std::ostringstream oss;
    oss << "Search Tree (depth " << max_depth << ")\n";
    // Would generate ASCII tree
    return oss.str();
}

std::string SearchVisualizer::node_to_string(const Node& node) const {
    std::ostringstream oss;
    oss << node.move.to_string() << " (" << node.score << ")";
    return oss.str();
}

// ============== PruningAnalyzer ==============

PruningAnalyzer::Analysis PruningAnalyzer::analyze_recent(int window) {
    Analysis analysis;
    
    auto& logger = TelemetryLogger::instance();
    // Would analyze recent pruning decisions from logger
    
    analysis.total_prunes = 0;
    analysis.total_searches = 0;
    analysis.prune_rate = 0.0f;
    analysis.correct_prunes = 0;
    analysis.incorrect_prunes = 0;
    analysis.accuracy = 0.0f;
    analysis.recent_accuracy = 0.0f;
    analysis.accuracy_trend = 0.0f;
    
    return analysis;
}

void PruningAnalyzer::compare_versions(const std::string& old_logs,
                                       const std::string& new_logs) {
    // Would compare pruning patterns between two versions
}

std::vector<std::string> PruningAnalyzer::detect_anomalies() {
    std::vector<std::string> anomalies;
    // Would detect suspicious patterns
    return anomalies;
}

std::vector<std::string> PruningAnalyzer::generate_recommendations() {
    std::vector<std::string> recommendations;
    // Would generate tuning recommendations
    return recommendations;
}

std::string PruningAnalyzer::Analysis::summary() const {
    std::ostringstream oss;
    oss << "Pruning Analysis:\n"
         << "  Total: " << total_prunes << "/" << total_searches
         << " (" << (prune_rate * 100) << "%)\n"
         << "  Accuracy: " << (accuracy * 100) << "%\n"
         << "  Trend: " << (accuracy_trend * 100) << "%\n";
    return oss.str();
}

} // namespace SearchTelemetry
