#pragma once

#include "types.h"
#include "board.h"
#include <string>
#include <memory>
#include <array>

namespace MultiNetwork {

// ======== Network Roles ========
// Different neural networks for different game phases

enum class NetworkRole {
    OPENING,      // 0-12 moves: focus on development, king safety, center control
    MIDDLEGAME,   // 13-40 moves: tactics, complex evaluation
    ENDGAME,      // 40+ moves or material low: endgame technique, passed pawns
    TACTICAL,     // Sharp positions: captures, checks, promotions
    DEFENSIVE,    // King under attack: defensive resources
    NUM_ROLES
};

// Phase detection
enum class GamePhase {
    OPENING,      // < 12 full moves, all pieces on board
    EARLY_MIDDLEGAME,  // 12-25 moves
    MIDDLEGAME,   // 25-40 moves
    LATE_MIDDLEGAME,   // 40-55 moves
    ENDGAME,      // < 8 pieces or pawn-only positions
    NUM_PHASES
};

// ======== Phase Detector ========
// Determines which network to use

class PhaseDetector {
public:
    static GamePhase detect(const BoardState& pos);
    
    // Detailed analysis
    struct PhaseInfo {
        GamePhase phase;
        float opening_weight;      // 0.0-1.0, how "opening-like"
        float endgame_weight;      // 0.0-1.0, how "endgame-like"
        float tactical_weight;     // 0.0-1.0, position sharpness
        float material_imbalance;  // 0.0-1.0, asymmetry
        int move_count;            // Approximate move number
        int piece_count;           // Total pieces on board
    };
    
    static PhaseInfo analyze(const BoardState& pos);
    
private:
    static constexpr int OPENING_THRESHOLD = 12;
    static constexpr int MIDDLEGAME_THRESHOLD = 25;
    static constexpr int LATE_MIDDLEGAME_THRESHOLD = 40;
    static constexpr int ENDGAME_PIECE_THRESHOLD = 8;
};

// ======== Network Interface ========
// Abstract interface for any NNUE network

class EvalNetwork {
public:
    virtual ~EvalNetwork() = default;
    
    virtual Value evaluate(const BoardState& pos) = 0;
    virtual bool load(const char* filepath) = 0;
    virtual bool save(const char* filepath) = 0;
    virtual const char* name() const = 0;
    virtual size_t size_bytes() const = 0;
};

// ======== Multi-Network Manager ========
// Manages multiple specialized networks

class MultiNetworkManager {
public:
    static constexpr size_t MAX_NETWORKS = 
        static_cast<size_t>(NetworkRole::NUM_ROLES);
    
    MultiNetworkManager();
    ~MultiNetworkManager();
    
    // Load networks for all roles
    bool load_all(const char* directory);
    
    // Load single network
    bool load_network(NetworkRole role, const char* filepath);
    
    // Evaluate with appropriate network(s)
    Value evaluate(const BoardState& pos);
    
    // Evaluate with all networks (for ensemble)
    std::array<Value, MAX_NETWORKS> evaluate_all(const BoardState& pos);
    
    // Get network blending weights
    std::array<float, MAX_NETWORKS> get_weights(const BoardState& pos) const;
    
    // Statistics
    struct Stats {
        size_t calls_by_role[MAX_NETWORKS];
        size_t total_calls;
        double avg_eval_time_us;
        size_t cache_hits;
    };
    
    Stats get_stats() const;
    void reset_stats();
    
    // Check if all networks loaded
    bool all_loaded() const;
    
    // Network info
    const char* network_name(NetworkRole role) const;
    bool network_loaded(NetworkRole role) const;
    
private:
    std::array<std::unique_ptr<EvalNetwork>, MAX_NETWORKS> networks_;
    std::array<bool, MAX_NETWORKS> loaded_;
    
    // Phase-based weights
    // [GamePhase][NetworkRole] = weight
    static constexpr float PHASE_WEIGHTS[static_cast<size_t>(GamePhase::NUM_PHASES)]
                                         [MAX_NETWORKS] = {
        // OPENING: mostly opening, some middlegame
        {0.70f, 0.25f, 0.05f, 0.00f, 0.00f},
        // EARLY_MIDDLEGAME
        {0.30f, 0.55f, 0.10f, 0.05f, 0.00f},
        // MIDDLEGAME
        {0.10f, 0.60f, 0.15f, 0.10f, 0.05f},
        // LATE_MIDDLEGAME
        {0.05f, 0.40f, 0.40f, 0.10f, 0.05f},
        // ENDGAME
        {0.00f, 0.15f, 0.75f, 0.05f, 0.05f},
    };
    
    // Cache for repeated positions
    struct CacheEntry {
        Key key;
        Value value;
        GamePhase phase;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    static constexpr size_t CACHE_SIZE = 65536;
    std::array<CacheEntry, CACHE_SIZE> cache_;
    
    Stats stats_{};
    
    Value evaluate_with_role(const BoardState& pos, NetworkRole role);
    size_t cache_index(Key key) const { return key % CACHE_SIZE; }
};

// ======== Adaptive Network Trainer ========
// Trains specialized networks for different phases

class AdaptiveTrainer {
public:
    struct TrainingConfig {
        int opening_epochs = 50;
        int middlegame_epochs = 100;
        int endgame_epochs = 50;
        int tactical_epochs = 30;
        
        size_t opening_positions = 500000;
        size_t middlegame_positions = 1000000;
        size_t endgame_positions = 300000;
        size_t tactical_positions = 200000;
        
        int hidden_size = 128;
        int batch_size = 4096;
    };
    
    // Split dataset by phase and train separate networks
    void train_specialized_networks(
        const char* input_dataset,
        const char* output_directory,
        const TrainingConfig& config = TrainingConfig{}
    );
    
    // Mine tactical positions from search instability
    void mine_tactical_positions(
        const char* telemetry_file,
        const char* output_file,
        int min_depth = 10,
        float instability_threshold = 2.0f
    );
    
    // Validate ensemble performance
    float validate_ensemble(
        const char* test_positions,
        const char* network_directory
    );
    
private:
    bool is_tactical_position(const BoardState& pos, Value eval_before, Value eval_after);
    bool is_defensive_position(const BoardState& pos, Value threat_level);
};

// ======== Global Access ========

MultiNetworkManager& get_multi_network();

} // namespace MultiNetwork
