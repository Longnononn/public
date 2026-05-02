#include "neural_syzygy.h"
#include "bitboard.h"
#include <cmath>
#include <fstream>
#include <algorithm>

namespace NeuralSyzygy {

// ============== PieceCount ==============

void PieceCount::from_board(const BoardState& pos) {
    reset();
    
    // Count pieces from bitboards
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Piece p = pos.piece_on(sq);
        if (p == NO_PIECE) continue;
        
        Color c = color_of(p);
        PieceType pt = type_of(p);
        
        total++;
        
        if (c == WHITE) {
            if (pt == KING) white_kings++;
            else if (pt == QUEEN) white_queens++;
            else if (pt == ROOK) white_rooks++;
            else if (pt == BISHOP) white_bishops++;
            else if (pt == KNIGHT) white_knights++;
            else if (pt == PAWN) white_pawns++;
        } else {
            if (pt == KING) black_kings++;
            else if (pt == QUEEN) black_queens++;
            else if (pt == ROOK) black_rooks++;
            else if (pt == BISHOP) black_bishops++;
            else if (pt == KNIGHT) black_knights++;
            else if (pt == PAWN) black_pawns++;
        }
    }
    
    kings = white_kings + black_kings;
    queens = white_queens + black_queens;
    rooks = white_rooks + black_rooks;
    bishops = white_bishops + black_bishops;
    knights = white_knights + black_knights;
    pawns = white_pawns + black_pawns;
}

EndgameCategory PieceCount::classify() const {
    if (total <= 5) {
        if (kings == 2 && pawns == 1) return EndgameCategory::KPK;
        if (kings == 2 && rooks == 1 && pawns == 1) {
            if (white_rooks == 1) return EndgameCategory::KRPKR;
        }
        if (kings == 2 && queens == 1 && pawns == 1) {
            if (white_queens == 1) return EndgameCategory::KQKP;
        }
        if (kings == 2 && bishops == 1 && knights == 1) {
            if (white_bishops == 1 && white_knights == 1) return EndgameCategory::KBNK;
        }
        if (kings == 2 && bishops == 2) return EndgameCategory::KBBK;
        if (kings == 2 && knights == 2) return EndgameCategory::KNNK;
    }
    
    if (kings == 2 && rooks == 2) return EndgameCategory::KRKR;
    if (kings == 2 && queens == 2) return EndgameCategory::KQKQ;
    if (kings == 2 && queens == 1 && rooks == 1 && pawns == 1) return EndgameCategory::KQPKR;
    if (kings == 2 && pawns == 2) return EndgameCategory::KPKP;
    
    return EndgameCategory::GENERAL;
}

// ============== NeuralEndgame ==============

NeuralEndgame::NeuralEndgame() {
    // Initialize with random weights or load from file
    std::fill(std::begin(weights_.w1), std::end(weights_.w1), 0);
    std::fill(std::begin(weights_.w2), std::end(weights_.w2), 0);
    std::fill(std::begin(weights_.w3), std::end(weights_.w3), 0);
    
    weights_.scale1 = 0.01f;
    weights_.scale2 = 0.01f;
    weights_.scale3 = 0.01f;
    
    cache_.resize(CACHE_SIZE);
}

NeuralEndgame::~NeuralEndgame() = default;

bool NeuralEndgame::load(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&weights_), sizeof(weights_));
    return true;
}

bool NeuralEndgame::save(const char* filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(&weights_), sizeof(weights_));
    return true;
}

EndgamePrediction NeuralEndgame::predict(const BoardState& pos) const {
    EndgamePrediction pred;
    pred.is_exact = false;
    pred.confidence = 0.5f;
    
    // Check cache
    Key key = pos.get_key();
    size_t cache_idx = key % CACHE_SIZE;
    
    if (cache_[cache_idx].key == key) {
        cacheHits_++;
        return cache_[cache_idx].pred;
    }
    cacheMisses_++;
    
    // Count pieces
    PieceCount pc;
    pc.from_board(pos);
    
    // Check if in domain
    if (!in_domain(pc)) {
        // Not in training domain, return default
        pred.win_prob = 0.33f;
        pred.draw_prob = 0.34f;
        pred.loss_prob = 0.33f;
        pred.dtz = 0;
        pred.category = pc.classify();
        return pred;
    }
    
    // Extract features
    alignas(16) int8_t features[256];
    extract_features(pos, pc, features);
    
    // Forward pass
    alignas(16) float output[4];
    forward(features, output);
    
    // Parse output
    float sum = output[0] + output[1] + output[2];
    if (sum > 0) {
        pred.win_prob = output[0] / sum;
        pred.draw_prob = output[1] / sum;
        pred.loss_prob = output[2] / sum;
    } else {
        pred.win_prob = 0.33f;
        pred.draw_prob = 0.34f;
        pred.loss_prob = 0.33f;
    }
    
    pred.dtz = static_cast<int>(output[3]);
    pred.category = pc.classify();
    
    // Confidence based on max probability
    pred.confidence = std::max({pred.win_prob, pred.draw_prob, pred.loss_prob});
    
    // Cache result
    cache_[cache_idx].key = key;
    cache_[cache_idx].pred = pred;
    
    return pred;
}

void NeuralEndgame::extract_features(const BoardState& pos, const PieceCount& pc,
                                      int8_t* features) const {
    // 256-dimensional feature vector
    // Encodes: piece placement, king positions, pawn structure
    
    std::fill(features, features + 256, 0);
    
    // Piece counts (32 features)
    features[0] = pc.white_kings;
    features[1] = pc.white_queens;
    features[2] = pc.white_rooks;
    features[3] = pc.white_bishops;
    features[4] = pc.white_knights;
    features[5] = pc.white_pawns;
    features[6] = pc.black_kings;
    features[7] = pc.black_queens;
    features[8] = pc.black_rooks;
    features[9] = pc.black_bishops;
    features[10] = pc.black_knights;
    features[11] = pc.black_pawns;
    
    // Material difference (4 features)
    features[12] = static_cast<int8_t>(pc.queens);
    features[13] = static_cast<int8_t>(pc.rooks);
    features[14] = static_cast<int8_t>(pc.bishops - pc.knights);
    features[15] = static_cast<int8_t>(pc.pawns);
    
    // King positions (64 features - one-hot for each king square)
    Square wk = pos.king_square(WHITE);
    Square bk = pos.king_square(BLACK);
    
    if (wk != SQ_NONE) features[16 + wk] = 1;
    if (bk != SQ_NONE) features[80 + bk] = 1;
    
    // King distance (1 feature)
    if (wk != SQ_NONE && bk != SQ_NONE) {
        int fileDist = abs(file_of(wk) - file_of(bk));
        int rankDist = abs(rank_of(wk) - rank_of(bk));
        features[144] = static_cast<int8_t>(fileDist + rankDist);
    }
    
    // Passed pawns (64 features)
    // Simplified: mark squares with passed pawns
    // Full implementation would check pawn structure
    
    // Space control (64 features)
    // Which squares are attacked by which side
    // Simplified: mark central squares
    
    features[200] = 1;  // e4
    features[201] = 1;  // d4
    features[202] = 1;  // e5
    features[203] = 1;  // d5
    
    // Remaining features reserved for future expansion
}

void NeuralEndgame::forward(const int8_t* features, float* output) const {
    // Layer 1: 256 -> 64
    alignas(16) float h1[64];
    
    for (int i = 0; i < 64; ++i) {
        int32_t sum = weights_.b1[i];
        for (int j = 0; j < 256; ++j) {
            sum += static_cast<int32_t>(features[j]) * 
                   static_cast<int32_t>(weights_.w1[i * 256 + j]);
        }
        h1[i] = clipped_relu(sum * weights_.scale1);
    }
    
    // Layer 2: 64 -> 32
    alignas(16) float h2[32];
    
    for (int i = 0; i < 32; ++i) {
        int32_t sum = weights_.b2[i];
        for (int j = 0; j < 64; ++j) {
            sum += static_cast<int32_t>(static_cast<int8_t>(h1[j] * 127)) * 
                   static_cast<int32_t>(weights_.w2[i * 64 + j]);
        }
        h2[i] = clipped_relu(sum * weights_.scale2);
    }
    
    // Layer 3: 32 -> 4 (W, D, L, DTZ)
    for (int i = 0; i < 4; ++i) {
        int32_t sum = weights_.b3[i];
        for (int j = 0; j < 32; ++j) {
            sum += static_cast<int32_t>(static_cast<int8_t>(h2[j] * 127)) * 
                   static_cast<int32_t>(weights_.w3[i * 32 + j]);
        }
        output[i] = sum * weights_.scale3;
    }
    
    // Apply softmax to first 3 outputs (W, D, L)
    float max_out = std::max({output[0], output[1], output[2]});
    float exp_sum = std::exp(output[0] - max_out) + 
                    std::exp(output[1] - max_out) + 
                    std::exp(output[2] - max_out);
    
    if (exp_sum > 0) {
        output[0] = std::exp(output[0] - max_out) / exp_sum;
        output[1] = std::exp(output[1] - max_out) / exp_sum;
        output[2] = std::exp(output[2] - max_out) / exp_sum;
    }
}

bool NeuralEndgame::in_domain(const PieceCount& pc) const {
    // Training domain: 2-7 pieces, specific categories
    return pc.total >= 2 && pc.total <= 7 && pc.kings == 2;
}

// ============== SyzygyInterface ==============

SyzygyInterface::SyzygyInterface() : tbHandle_(nullptr) {
}

bool SyzygyInterface::load_tablebases(const char* /*path*/, int /*max_pieces*/) {
    // Would call actual Syzygy library
    // For now, return false (no TB loaded)
    return false;
}

EndgamePrediction SyzygyInterface::probe(const BoardState& pos) {
    neuralProbes_++;
    
    // Try neural prediction first
    EndgamePrediction pred = neural_.predict(pos);
    
    // If confidence is low and TB is available, fall back
    if (pred.confidence < 0.7f && tbHandle_ != nullptr) {
        tbProbes_++;
        // Would probe actual TB
        // tbHits_++;
        // pred.is_exact = true;
    }
    
    return pred;
}

Move SyzygyInterface::get_best_move(const BoardState& pos) {
    // Get prediction first
    EndgamePrediction pred = probe(pos);
    
    // If exact TB available, use it
    if (pred.is_exact && tbHandle_ != nullptr) {
        // Would query TB for best move
        // For now, return none
    }
    
    // Otherwise, use neural prediction to guide search
    // Return none to let search handle it
    return Move::none();
}

bool SyzygyInterface::has_tb(Key /*key*/) const {
    return tbHandle_ != nullptr;
}

// ============== Global Instance ==============

SyzygyInterface& get_syzygy() {
    static SyzygyInterface instance;
    return instance;
}

// ============== EndgameTrainer ==============

void EndgameTrainer::generate_positions(const char* /*tb_path*/,
                                         const char* /*output_file*/,
                                         int /*max_positions*/) {
    // Would generate positions from tablebases
    // For each endgame category:
    // 1. Enumerate all legal positions
    // 2. Query TB for WDL/DTZ
    // 3. Save to training file
}

void EndgameTrainer::train(const char* /*data_file*/,
                            const char* /*output_model*/,
                            int /*epochs*/) {
    // Would train neural network using PyTorch/TF
    // Similar to train_nnue.py but specialized for endgames
}

float EndgameTrainer::validate(const char* /*data_file*/,
                                 const char* /*tb_path*/) {
    // Would validate neural predictions against exact TB
    // Return accuracy
    return 0.0f;
}

} // namespace NeuralSyzygy
