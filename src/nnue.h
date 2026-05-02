#pragma once

#include "types.h"
#include "board.h"
#include <string>
#include <cstdint>

namespace NNUE {

using namespace Nexus;

// NNUE Architecture Constants
constexpr int INPUT_FEATURES = 41024;  // HalfKP: 64 king squares × 640 piece positions
constexpr int L1_SIZE = 256;           // Hidden layer size
constexpr int OUTPUT_BUCKETS = 8;      // Phase-based output buckets
constexpr int MAX_PLY = 128;

// Quantization constants
constexpr int WEIGHT_SCALE_BITS = 6;
constexpr int WEIGHT_SCALE = 1 << WEIGHT_SCALE_BITS;  // 64
constexpr int EVAL_SCALE = 16;

// Feature index: perspective × king_square × piece_square × piece_type
inline int make_index(Color perspective, Square kingSq, Square pieceSq, 
                      PieceType pt, Color pieceColor) {
    // Piece encoding: 0-5 for our pieces, 6-11 for opponent pieces (from perspective)
    int pieceEncode = (pieceColor == perspective) ? (pt) : (pt + 6);
    // Offset by king square
    int kingOffset = static_cast<int>(kingSq) * 640;  // 64 squares × 10 piece types
    // Piece position
    int pieceOffset = static_cast<int>(pieceSq) * 10 + pieceEncode;
    
    return kingOffset + pieceOffset;
}

// Network weights structure
struct alignas(64) Network {
    // Layer 1: Feature -> Accumulator
    int16_t featureWeights[INPUT_FEATURES][L1_SIZE];
    int16_t featureBiases[L1_SIZE];
    
    // Layer 2: Accumulator -> Output (per bucket)
    int16_t outputWeights[OUTPUT_BUCKETS][2][L1_SIZE];  // [bucket][perspective][neuron]
    int32_t outputBiases[OUTPUT_BUCKETS];
};

// Accumulator for incremental updates
class Accumulator {
public:
    Accumulator() = default;
    
    void reset();
    void refresh(const BoardState& pos, const Network& net);
    
    // Incremental updates
    void update_add(int feature, const Network& net);
    void update_remove(int feature, const Network& net);
    void update_move(int removed, int added, const Network& net);
    
    // Evaluation
    int32_t evaluate(Color sideToMove, int bucket, const Network& net) const;
    
    // State management for stack
    void save_state(int ply);
    void restore_state(int ply);
    void clear_stack();
    
private:
    // Accumulator values for both perspectives
    alignas(64) int16_t values[2][L1_SIZE];  // [WHITE_PERSP][neurons]
    
    // Stack for undoing during search
    alignas(64) int16_t stackValues[MAX_PLY][2][L1_SIZE];
    int stackSize = 0;
};

// Main evaluator
class Evaluator {
public:
    Evaluator();
    
    // Initialization
    bool load_network(const std::string& path);
    void init_default();
    bool is_loaded() const { return networkLoaded; }
    
    // Accumulator management
    void reset_accumulator(const BoardState& pos);
    void update(Move m, const BoardState& pos);  // After do_move
    void undo();  // After undo_move
    
    // Evaluation
    Value evaluate(const BoardState& pos);
    
    // Direct evaluation (no accumulator)
    Value evaluate_direct(const BoardState& pos);
    
private:
    Network net;
    Accumulator acc;
    bool networkLoaded = false;
    
    int get_bucket(const BoardState& pos) const;
    int32_t transform(int32_t x) const;  // Clipped ReLU
};

// Global evaluator instance
extern Evaluator g_nnue;

// Helper functions
bool verify_network(const std::string& path);
void init_nnue();
Value evaluate_nnue(const BoardState& pos);

} // namespace NNUE
