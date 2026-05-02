#pragma once

#include "types.h"
#include "board.h"
#include <cstdint>
#include <vector>

namespace NeuralSyzygy {

// ======== Neural Endgame Solver ========
// Replaces traditional 6-piece EGTB with a tiny neural network
// Trained on endgame positions to predict exact WDL/DTZ
// Runs in RAM/cache, no disk I/O, sub-microsecond latency

enum class EndgameCategory {
    KPK,           // King + Pawn vs King
    KRKP,          // King + Rook vs King + Pawn
    KQKP,          // King + Queen vs King + Pawn
    KRPKR,         // King + Rook + Pawn vs King + Rook
    KQPKR,         // King + Queen + Pawn vs King + Rook
    KBNK,          // King + Bishop + Knight vs King
    KBBK,          // King + Bishop + Bishop vs King
    KPKP,          // King + Pawn vs King + Pawn
    KQKQ,          // King + Queen vs King + Queen
    KRKR,          // King + Rook vs King + Rook
    KNNK,          // King + Knight + Knight vs King
    GENERAL,       // Generic endgame (6-7 pieces)
};

// ======== Endgame Prediction ========

struct EndgamePrediction {
    // Win-Draw-Loss probability
    float win_prob;      // 0.0 to 1.0
    float draw_prob;     // 0.0 to 1.0
    float loss_prob;     // 0.0 to 1.0
    
    // Distance to zero (DTZ) estimate
    int dtz;             // Moves to zero (can be negative for losing side)
    
    // Confidence
    float confidence;    // 0.0 to 1.0 (how certain the prediction is)
    
    // Category
    EndgameCategory category;
    
    // Is this a known tablebase position?
    bool is_exact;       // True if we have exact TB data
    
    // Result for search
    Value value() const {
        if (is_exact) {
            if (dtz == 0) return VALUE_DRAW;
            return dtz > 0 ? VALUE_MATE - dtz : -VALUE_MATE + dtz;
        }
        
        // Probabilistic value
        float expected = win_prob - loss_prob;
        return Value(expected * 1000.0);  // Scale to centipawns
    }
};

// ======== Piece Counter ========
// Counts pieces by type for endgame classification

struct PieceCount {
    int total;
    int kings;
    int queens;
    int rooks;
    int bishops;
    int knights;
    int pawns;
    
    // By color
    int white_kings, white_queens, white_rooks, white_bishops, white_knights, white_pawns;
    int black_kings, black_queens, black_rooks, black_bishops, black_knights, black_pawns;
    
    PieceCount() { reset(); }
    
    void reset() {
        total = kings = queens = rooks = bishops = knights = pawns = 0;
        white_kings = white_queens = white_rooks = white_bishops = white_knights = white_pawns = 0;
        black_kings = black_queens = black_rooks = black_bishops = black_knights = black_pawns = 0;
    }
    
    void from_board(const BoardState& pos);
    
    bool is_endgame() const {
        return total <= 7 || (total <= 10 && queens == 0);
    }
    
    EndgameCategory classify() const;
};

// ======== Neural Endgame Network ========
// Tiny network for endgame evaluation
// Architecture: Input(256) -> Hidden(64) -> Output(3) [W/D/L] + DTZ regression

class NeuralEndgame {
public:
    NeuralEndgame();
    ~NeuralEndgame();
    
    // Load trained weights
    bool load(const char* filename);
    
    // Save weights
    bool save(const char* filename) const;
    
    // Predict endgame result
    EndgamePrediction predict(const BoardState& pos) const;
    
    // Batch prediction for multiple positions
    std::vector<EndgamePrediction> predict_batch(
        const std::vector<BoardState>& positions) const;
    
    // Check if position is in training domain
    bool in_domain(const PieceCount& pc) const;
    
    // Statistics
    int cache_hits() const { return cacheHits_; }
    int cache_misses() const { return cacheMisses_; }
    void clear_cache() { cache_.clear(); }
    
private:
    // Network weights (quantized INT8 for cache efficiency)
    struct Weights {
        // Input layer: 256 -> 64
        int8_t w1[256 * 64];
        int8_t b1[64];
        
        // Hidden layer: 64 -> 32
        int8_t w2[64 * 32];
        int8_t b2[32];
        
        // Output layer: 32 -> 4 (W, D, L, DTZ)
        int8_t w3[32 * 4];
        int8_t b3[4];
        
        // Scales for dequantization
        float scale1, scale2, scale3;
    };
    
    Weights weights_;
    
    // Cache for repeated positions
    struct CacheEntry {
        Key key;
        EndgamePrediction pred;
    };
    std::vector<CacheEntry> cache_;
    static constexpr size_t CACHE_SIZE = 4096;
    mutable int cacheHits_ = 0;
    mutable int cacheMisses_ = 0;
    
    // Feature extraction
    void extract_features(const BoardState& pos, const PieceCount& pc,
                          int8_t* features) const;
    
    // Forward pass
    void forward(const int8_t* features, float* output) const;
    
    // Clipped ReLU
    float clipped_relu(float x) const {
        return std::max(0.0f, std::min(1.0f, x));
    }
};

// ======== Syzygy Interface ========
// Fallback to traditional Syzygy when neural prediction confidence is low

class SyzygyInterface {
public:
    SyzygyInterface();
    
    // Load traditional tablebases (optional, for high-confidence cases)
    bool load_tablebases(const char* path, int max_pieces = 6);
    
    // Probe: try neural first, fallback to TB if available
    EndgamePrediction probe(const BoardState& pos);
    
    // Get best move for endgame
    Move get_best_move(const BoardState& pos);
    
    // Statistics
    int neural_probes() const { return neuralProbes_; }
    int tb_probes() const { return tbProbes_; }
    int tb_hits() const { return tbHits_; }
    
private:
    NeuralEndgame neural_;
    void* tbHandle_;  // Syzygy tablebase handle (opaque pointer)
    
    int neuralProbes_ = 0;
    int tbProbes_ = 0;
    int tbHits_ = 0;
    
    bool has_tb(Key key) const;
};

// ======== Endgame Trainer ========
// Generates training data for neural endgame network

class EndgameTrainer {
public:
    // Generate endgame positions from tablebases
    void generate_positions(const char* tb_path, const char* output_file,
                           int max_positions = 1000000);
    
    // Train neural network on generated data
    void train(const char* data_file, const char* output_model, int epochs = 100);
    
    // Validate against tablebase
    float validate(const char* data_file, const char* tb_path);
    
private:
    // Monte Carlo sampling of endgame positions
    void sample_endgame_positions(const PieceCount& pc,
                                   std::vector<BoardState>& positions,
                                   int count);
};

// ======== Global Instance ========
// Singleton for endgame probing

SyzygyInterface& get_syzygy();

inline EndgamePrediction probe_endgame(const BoardState& pos) {
    return get_syzygy().probe(pos);
}

} // namespace NeuralSyzygy
