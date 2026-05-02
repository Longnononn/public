#include "nnue.h"
#include "simd.h"
#include <cstring>
#include <fstream>
#include <iostream>

namespace NNUE {

// Global NNUE evaluator
Evaluator g_nnue;

// ================ Accumulator Implementation ================

void Accumulator::reset() {
    std::memset(values, 0, sizeof(values));
}

void Accumulator::refresh(const BoardState& pos, const Network& net) {
    // Full recompute from scratch
    std::memset(values, 0, sizeof(values));
    
    // For both perspectives
    for (int p = 0; p < 2; ++p) {
        Color perspective = Color(p);
        Square kingSq = pos.king_square(perspective);
        
        // Add all non-king pieces
        for (int sq_i = SQ_A1; sq_i <= SQ_H8; ++sq_i) {
            Square sq = Square(sq_i);
            Piece pc = pos.piece_on(sq);
            if (pc != NO_PIECE && type_of(pc) != KING) {
                int idx = make_index(perspective, kingSq, sq, 
                                    type_of(pc), color_of(pc));
                
                // Add feature weights
                #ifdef USE_AVX2
                // SIMD version
                for (int i = 0; i < L1_SIZE; i += 16) {
                    __m256i acc = _mm256_load_si256((__m256i*)&values[p][i]);
                    __m256i w = _mm256_load_si256((__m256i*)&net.featureWeights[idx][i]);
                    acc = _mm256_add_epi16(acc, w);
                    _mm256_store_si256((__m256i*)&values[p][i], acc);
                }
                #else
                // Scalar version
                for (int i = 0; i < L1_SIZE; ++i) {
                    values[p][i] += net.featureWeights[idx][i];
                }
                #endif
            }
        }
        
        // Add bias
        for (int i = 0; i < L1_SIZE; ++i) {
            values[p][i] += net.featureBiases[i];
        }
    }
}

void Accumulator::update_add(int feature, const Network& net) {
    #ifdef USE_AVX2
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_load_si256((__m256i*)&values[p][i]);
            __m256i w = _mm256_load_si256((__m256i*)&net.featureWeights[feature][i]);
            acc = _mm256_add_epi16(acc, w);
            _mm256_store_si256((__m256i*)&values[p][i], acc);
        }
    }
    #else
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; ++i) {
            values[p][i] += net.featureWeights[feature][i];
        }
    }
    #endif
}

void Accumulator::update_remove(int feature, const Network& net) {
    #ifdef USE_AVX2
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_load_si256((__m256i*)&values[p][i]);
            __m256i w = _mm256_load_si256((__m256i*)&net.featureWeights[feature][i]);
            acc = _mm256_sub_epi16(acc, w);
            _mm256_store_si256((__m256i*)&values[p][i], acc);
        }
    }
    #else
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; ++i) {
            values[p][i] -= net.featureWeights[feature][i];
        }
    }
    #endif
}

void Accumulator::update_move(int removed, int added, const Network& net) {
    // Remove old feature and add new feature
    #ifdef USE_AVX2
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i acc = _mm256_load_si256((__m256i*)&values[p][i]);
            
            __m256i wRemoved = _mm256_load_si256((__m256i*)&net.featureWeights[removed][i]);
            __m256i wAdded = _mm256_load_si256((__m256i*)&net.featureWeights[added][i]);
            
            acc = _mm256_sub_epi16(acc, wRemoved);
            acc = _mm256_add_epi16(acc, wAdded);
            
            _mm256_store_si256((__m256i*)&values[p][i], acc);
        }
    }
    #else
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; ++i) {
            values[p][i] -= net.featureWeights[removed][i];
            values[p][i] += net.featureWeights[added][i];
        }
    }
    #endif
}

int32_t Accumulator::evaluate(Color sideToMove, int bucket, const Network& net) const {
    // Clipped ReLU activation
    alignas(64) int16_t activated[2][L1_SIZE];
    
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; ++i) {
            // Clip to [0, 127] range
            int16_t val = values[p][i] >> WEIGHT_SCALE_BITS;
            activated[p][i] = std::max<int16_t>(0, std::min<int16_t>(127, val));
        }
    }
    
    // Output layer: dot product with weights
    int32_t sum = net.outputBiases[bucket];
    
    #ifdef USE_AVX2
    // Process both perspectives
    for (int p = 0; p < 2; ++p) {
        __m256i sumVec = _mm256_setzero_si256();
        for (int i = 0; i < L1_SIZE; i += 16) {
            __m256i act = _mm256_load_si256((__m256i*)&activated[p][i]);
            __m256i w = _mm256_load_si256((__m256i*)&net.outputWeights[bucket][p][i]);
            
            // Multiply and add (16-bit × 16-bit -> 32-bit)
            __m256i prod = _mm256_madd_epi16(act, w);
            sumVec = _mm256_add_epi32(sumVec, prod);
        }
        
        // Horizontal sum
        sumVec = _mm256_hadd_epi32(sumVec, sumVec);
        sumVec = _mm256_hadd_epi32(sumVec, sumVec);
        sum += _mm256_extract_epi32(sumVec, 0) + _mm256_extract_epi32(sumVec, 4);
    }
    #else
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < L1_SIZE; ++i) {
            sum += activated[p][i] * net.outputWeights[bucket][p][i];
        }
    }
    #endif
    
    return sum;
}

void Accumulator::save_state(int ply) {
    if (ply < MAX_PLY) {
        std::memcpy(stackValues[ply], values, sizeof(values));
    }
}

void Accumulator::restore_state(int ply) {
    if (ply < MAX_PLY) {
        std::memcpy(values, stackValues[ply], sizeof(values));
    }
}

void Accumulator::clear_stack() {
    stackSize = 0;
}

// ================ Evaluator Implementation ================

Evaluator::Evaluator() {
    // Initialize with zeros
    std::memset(&net, 0, sizeof(net));
}

bool Evaluator::load_network(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "NNUE: Failed to open " << path << std::endl;
        return false;
    }
    
    // Read header
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "NNUE", 4) != 0) {
        std::cerr << "NNUE: Invalid magic" << std::endl;
        return false;
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "NNUE: Unsupported version " << version << std::endl;
        return false;
    }
    
    uint32_t hash;
    file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
    
    // Read architecture
    uint16_t inputFeatures, hiddenSize, outputBuckets;
    file.read(reinterpret_cast<char*>(&inputFeatures), sizeof(inputFeatures));
    file.read(reinterpret_cast<char*>(&hiddenSize), sizeof(hiddenSize));
    file.read(reinterpret_cast<char*>(&outputBuckets), sizeof(outputBuckets));
    
    if (inputFeatures != INPUT_FEATURES || hiddenSize != L1_SIZE || 
        outputBuckets != OUTPUT_BUCKETS) {
        std::cerr << "NNUE: Architecture mismatch" << std::endl;
        return false;
    }
    
    // Read weights
    file.read(reinterpret_cast<char*>(&net.featureWeights), sizeof(net.featureWeights));
    file.read(reinterpret_cast<char*>(&net.featureBiases), sizeof(net.featureBiases));
    file.read(reinterpret_cast<char*>(&net.outputWeights), sizeof(net.outputWeights));
    file.read(reinterpret_cast<char*>(&net.outputBiases), sizeof(net.outputBiases));
    
    networkLoaded = true;
    std::cout << "NNUE: Loaded network from " << path << std::endl;
    return true;
}

void Evaluator::init_default() {
    // Initialize with random small weights for testing
    // In production, load from trained file
    for (int i = 0; i < INPUT_FEATURES; ++i) {
        for (int j = 0; j < L1_SIZE; ++j) {
            net.featureWeights[i][j] = (i + j) % 7 - 3;  // Small random values
        }
    }
    for (int i = 0; i < L1_SIZE; ++i) {
        net.featureBiases[i] = 0;
    }
    for (int b = 0; b < OUTPUT_BUCKETS; ++b) {
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < L1_SIZE; ++i) {
                net.outputWeights[b][p][i] = 1;
            }
        }
        net.outputBiases[b] = 0;
    }
    networkLoaded = true;
}

void Evaluator::reset_accumulator(const BoardState& pos) {
    acc.refresh(pos, net);
    acc.clear_stack();
}

void Evaluator::update(Move m, const BoardState& pos) {
    if (!m.is_ok() || !networkLoaded) return;
    
    // Save current state for undo
    // Note: This should be managed by ply in search
    
    Square from = m.from_sq();
    Square to = m.to_sq();
    Color side = pos.sideToMove;  // Before move was made
    
    Piece movingPiece = pos.piece_on(from);
    if (movingPiece == NO_PIECE) return;
    
    PieceType pt = type_of(movingPiece);
    
    // Handle for both perspectives
    for (int p = 0; p < 2; ++p) {
        Color perspective = Color(p);
        Square kingSq = pos.king_square(perspective);
        
        // Remove piece from 'from' square
        int idxFrom = make_index(perspective, kingSq, from, pt, side);
        
        // Add piece to 'to' square
        int idxTo = make_index(perspective, kingSq, to, pt, side);
        
        // Handle promotion
        if (m.is_promotion()) {
            idxTo = make_index(perspective, kingSq, to, m.promotion_type(), side);
        }
        
        // Handle capture
        Square capSq = to;
        if (m.is_enpassant()) {
            capSq = Square(to + (side == WHITE ? -8 : 8));
        }
        Piece captured = pos.piece_on(capSq);
        if (captured != NO_PIECE) {
            PieceType capPt = type_of(captured);
            Color capColor = color_of(captured);
            int idxCap = make_index(perspective, kingSq, capSq, capPt, capColor);
            // Remove captured piece
            acc.update_remove(idxCap, net);
        }
        
        // Update moving piece
        if (m.is_promotion()) {
            acc.update_remove(idxFrom, net);
            acc.update_add(idxTo, net);
        } else {
            acc.update_move(idxFrom, idxTo, net);
        }
    }
}

void Evaluator::undo() {
    // State restored by search stack
}

int Evaluator::get_bucket(const BoardState& pos) const {
    // Bucket based on material count (simple phase calculation)
    int material = 0;
    for (int sq_i = SQ_A1; sq_i <= SQ_H8; ++sq_i) {
        Square sq = Square(sq_i);
        Piece pc = pos.piece_on(sq);
        if (pc != NO_PIECE) {
            PieceType pt = type_of(pc);
            if (pt == QUEEN) material += 9;
            else if (pt == ROOK) material += 5;
            else if (pt == BISHOP || pt == KNIGHT) material += 3;
        }
    }
    
    // Map material to bucket
    if (material >= 56) return 0;
    if (material >= 48) return 1;
    if (material >= 40) return 2;
    if (material >= 32) return 3;
    if (material >= 24) return 4;
    if (material >= 16) return 5;
    if (material >= 8) return 6;
    return 7;
}

Value Evaluator::evaluate(const BoardState& pos) {
    if (!networkLoaded) {
        return VALUE_NONE;
    }
    
    int bucket = get_bucket(pos);
    int32_t nnueValue = acc.evaluate(pos.sideToMove, bucket, net);
    
    // Scale to centipawns
    int32_t eval = nnueValue / (EVAL_SCALE * WEIGHT_SCALE);
    
    // Clamp to valid range
    return Value(std::max<int>(-VALUE_MATE + 1, std::min<int>(VALUE_MATE - 1, eval)));
}

Value Evaluator::evaluate_direct(const BoardState& pos) {
    // Full evaluation without using accumulator
    Accumulator tempAcc;
    tempAcc.refresh(pos, net);
    
    int bucket = get_bucket(pos);
    int32_t nnueValue = tempAcc.evaluate(pos.sideToMove, bucket, net);
    int32_t eval = nnueValue / (EVAL_SCALE * WEIGHT_SCALE);
    
    return Value(std::max<int>(-VALUE_MATE + 1, std::min<int>(VALUE_MATE - 1, eval)));
}

// ================ Helper Functions ================

bool verify_network(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Expected size: header + weights
    size_t expectedSize = 4 + 4 + 4 + 6 + sizeof(Network);  // magic + version + hash + arch + weights
    
    return size >= expectedSize;
}

void init_nnue() {
    // Try to load network file
    if (!g_nnue.load_network("nexus.nnue")) {
        std::cout << "NNUE: No network found, using default initialization" << std::endl;
        g_nnue.init_default();
    }
}

Value evaluate_nnue(const BoardState& pos) {
    return g_nnue.evaluate(pos);
}

} // namespace NNUE
