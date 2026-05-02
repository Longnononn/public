#include "multi_network.h"
#include "eval.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace MultiNetwork {

// ============== PhaseDetector ==============

GamePhase PhaseDetector::detect(const BoardState& pos) {
    auto info = analyze(pos);
    return info.phase;
}

PhaseDetector::PhaseInfo PhaseDetector::analyze(const BoardState& pos) {
    PhaseInfo info{};
    
    // Count pieces
    int total_pieces = 0;
    int queens = 0;
    int rooks = 0;
    int minors = 0;
    int pawns = 0;
    
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Piece p = pos.piece_on(sq);
        if (p == NO_PIECE) continue;
        
        PieceType pt = type_of(p);
        total_pieces++;
        
        switch (pt) {
            case QUEEN: queens++; break;
            case ROOK: rooks++; break;
            case BISHOP:
            case KNIGHT: minors++; break;
            case PAWN: pawns++; break;
            default: break;
        }
    }
    
    info.piece_count = total_pieces;
    
    // Determine phase based on material and move count approximation
    // (move count is hard to know, use material as proxy)
    int non_pawn_pieces = total_pieces - pawns;
    
    if (non_pawn_pieces <= 4 || total_pieces <= 7) {
        info.phase = GamePhase::ENDGAME;
        info.endgame_weight = 1.0f;
    } else if (non_pawn_pieces <= 8 && queens <= 1) {
        info.phase = GamePhase::LATE_MIDDLEGAME;
        info.endgame_weight = 0.6f;
        info.opening_weight = 0.0f;
    } else if (non_pawn_pieces <= 14 || pawns <= 12) {
        info.phase = GamePhase::MIDDLEGAME;
        info.endgame_weight = 0.2f;
        info.opening_weight = 0.1f;
    } else if (non_pawn_pieces <= 22) {
        info.phase = GamePhase::EARLY_MIDDLEGAME;
        info.endgame_weight = 0.05f;
        info.opening_weight = 0.4f;
    } else {
        info.phase = GamePhase::OPENING;
        info.opening_weight = 0.8f;
    }
    
    // Tactical assessment: lots of pieces near kings, many captures available
    // Simplified: more material = more tactical possibilities
    info.tactical_weight = std::min(1.0f, (float)(queens + rooks) / 4.0f);
    
    // Material imbalance
    // Count white vs black material (excluding kings and pawns for simplicity)
    info.material_imbalance = 0.0f;  // Would need detailed count
    
    return info;
}

// ============== MultiNetworkManager ==============

MultiNetworkManager::MultiNetworkManager() {
    std::fill(loaded_.begin(), loaded_.end(), false);
    
    // Initialize cache
    for (auto& entry : cache_) {
        entry.key = 0;
        entry.value = VALUE_NONE;
        entry.phase = GamePhase::OPENING;
        entry.timestamp = std::chrono::steady_clock::now();
    }
}

MultiNetworkManager::~MultiNetworkManager() = default;

bool MultiNetworkManager::load_all(const char* directory) {
    // Try to load all known network files
    const char* filenames[MAX_NETWORKS] = {
        "opening.nnue",
        "middlegame.nnue",
        "endgame.nnue",
        "tactical.nnue",
        "defensive.nnue"
    };
    
    bool all_success = true;
    
    for (size_t i = 0; i < MAX_NETWORKS; ++i) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", directory, filenames[i]);
        
        // For now, networks_ is abstract - would need concrete implementations
        // loaded_[i] = networks_[i]->load(path);
        // Placeholder
        loaded_[i] = false;
        all_success = false;
    }
    
    return all_success;
}

bool MultiNetworkManager::load_network(NetworkRole role, const char* filepath) {
    size_t idx = static_cast<size_t>(role);
    if (idx >= MAX_NETWORKS) return false;
    
    // Would create appropriate network implementation
    // networks_[idx] = create_network(role);
    // loaded_[idx] = networks_[idx]->load(filepath);
    
    // Placeholder
    loaded_[idx] = false;
    return false;
}

Value MultiNetworkManager::evaluate(const BoardState& pos) {
    stats_.total_calls++;
    
    // Check cache
    Key key = pos.get_key();
    size_t cidx = cache_index(key);
    
    auto now = std::chrono::steady_clock::now();
    auto& entry = cache_[cidx];
    
    if (entry.key == key) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - entry.timestamp).count();
        
        if (age < 60) {  // 60 second TTL
            stats_.cache_hits++;
            return entry.value;
        }
    }
    
    // Get network blending weights
    auto weights = get_weights(pos);
    
    // Evaluate with weighted ensemble
    Value final_value = VALUE_NONE;
    float total_weight = 0.0f;
    float weighted_sum = 0.0f;
    
    for (size_t i = 0; i < MAX_NETWORKS; ++i) {
        if (!loaded_[i] || weights[i] <= 0.0f) continue;
        
        auto role = static_cast<NetworkRole>(i);
        Value v = evaluate_with_role(pos, role);
        
        if (v != VALUE_NONE) {
            weighted_sum += (float)v * weights[i];
            total_weight += weights[i];
            stats_.calls_by_role[i]++;
        }
    }
    
    if (total_weight > 0.0f) {
        final_value = static_cast<Value>(weighted_sum / total_weight);
    } else {
        // Fallback to classical evaluation
        final_value = Eval::evaluate(pos);
    }
    
    // Update cache
    entry.key = key;
    entry.value = final_value;
    auto phase_info = PhaseDetector::analyze(pos);
    entry.phase = phase_info.phase;
    entry.timestamp = now;
    
    return final_value;
}

std::array<Value, MultiNetworkManager::MAX_NETWORKS> 
MultiNetworkManager::evaluate_all(const BoardState& pos) {
    std::array<Value, MAX_NETWORKS> results;
    results.fill(VALUE_NONE);
    
    for (size_t i = 0; i < MAX_NETWORKS; ++i) {
        if (loaded_[i]) {
            auto role = static_cast<NetworkRole>(i);
            results[i] = evaluate_with_role(pos, role);
        }
    }
    
    return results;
}

std::array<float, MultiNetworkManager::MAX_NETWORKS>
MultiNetworkManager::get_weights(const BoardState& pos) const {
    auto phase_info = PhaseDetector::analyze(pos);
    
    std::array<float, MAX_NETWORKS> weights;
    
    size_t phase_idx = static_cast<size_t>(phase_info.phase);
    if (phase_idx >= static_cast<size_t>(GamePhase::NUM_PHASES)) {
        phase_idx = static_cast<size_t>(GamePhase::MIDDLEGAME);
    }
    
    // Copy weights from phase matrix
    for (size_t i = 0; i < MAX_NETWORKS; ++i) {
        weights[i] = PHASE_WEIGHTS[phase_idx][i];
        
        // Boost tactical network in sharp positions
        if (static_cast<NetworkRole>(i) == NetworkRole::TACTICAL &&
            phase_info.tactical_weight > 0.6f) {
            weights[i] *= 1.5f;
        }
        
        // Only include loaded networks
        if (!loaded_[i]) {
            weights[i] = 0.0f;
        }
    }
    
    // Normalize
    float sum = 0.0f;
    for (auto w : weights) sum += w;
    
    if (sum > 0.0f) {
        for (auto& w : weights) w /= sum;
    }
    
    return weights;
}

Value MultiNetworkManager::evaluate_with_role(const BoardState& pos, 
                                               NetworkRole role) {
    size_t idx = static_cast<size_t>(role);
    if (idx >= MAX_NETWORKS || !loaded_[idx]) {
        return VALUE_NONE;
    }
    
    // Would call network evaluation
    // Placeholder: use classical with phase bias
    Value base = Eval::evaluate(pos);
    
    // Apply role-specific bias (would be replaced by actual network)
    switch (role) {
        case NetworkRole::OPENING:
            // Slight development bonus
            return base;
        case NetworkRole::MIDDLEGAME:
            return base;
        case NetworkRole::ENDGAME:
            // Could apply endgame-specific scaling
            return base;
        case NetworkRole::TACTICAL:
            // Could boost sharp positions
            return base;
        case NetworkRole::DEFENSIVE:
            // Could adjust king safety
            return base;
        default:
            return base;
    }
}

MultiNetworkManager::Stats MultiNetworkManager::get_stats() const {
    return stats_;
}

void MultiNetworkManager::reset_stats() {
    stats_ = {};
}

bool MultiNetworkManager::all_loaded() const {
    for (size_t i = 0; i < MAX_NETWORKS; ++i) {
        if (!loaded_[i]) return false;
    }
    return true;
}

const char* MultiNetworkManager::network_name(NetworkRole role) const {
    switch (role) {
        case NetworkRole::OPENING: return "opening";
        case NetworkRole::MIDDLEGAME: return "middlegame";
        case NetworkRole::ENDGAME: return "endgame";
        case NetworkRole::TACTICAL: return "tactical";
        case NetworkRole::DEFENSIVE: return "defensive";
        default: return "unknown";
    }
}

bool MultiNetworkManager::network_loaded(NetworkRole role) const {
    size_t idx = static_cast<size_t>(role);
    return idx < MAX_NETWORKS && loaded_[idx];
}

// ============== AdaptiveTrainer ==============

void AdaptiveTrainer::train_specialized_networks(
    const char* /*input_dataset*/,
    const char* /*output_directory*/,
    const TrainingConfig& /*config*/) {
    // Would:
    // 1. Parse dataset and classify positions by phase
    // 2. Train separate networks for each phase
    // 3. Save to output_directory/
    
    // Placeholder implementation
}

void AdaptiveTrainer::mine_tactical_positions(
    const char* /*telemetry_file*/,
    const char* /*output_file*/,
    int /*min_depth*/,
    float /*instability_threshold*/) {
    // Would:
    // 1. Read search telemetry
    // 2. Find positions with high instability
    // 3. Filter by depth and threshold
    // 4. Export to output_file for training
    
    // Placeholder implementation
}

float AdaptiveTrainer::validate_ensemble(
    const char* /*test_positions*/,
    const char* /*network_directory*/) {
    // Would:
    // 1. Load all networks from directory
    // 2. Evaluate on test positions
    // 3. Compare single vs ensemble accuracy
    
    return 0.0f;
}

bool AdaptiveTrainer::is_tactical_position(
    const BoardState& /*pos*/,
    Value /*eval_before*/,
    Value /*eval_after*/) {
    // Would detect if position is sharp/tactical
    return false;
}

bool AdaptiveTrainer::is_defensive_position(
    const BoardState& /*pos*/,
    Value /*threat_level*/) {
    // Would detect if king is under attack
    return false;
}

// ============== Global Instance ==============

MultiNetworkManager& get_multi_network() {
    static MultiNetworkManager instance;
    return instance;
}

} // namespace MultiNetwork
