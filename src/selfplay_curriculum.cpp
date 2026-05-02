#include "selfplay_curriculum.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>

namespace SelfPlay {

// ============== Dirichlet Noise ==============

DirichletNoise::DirichletNoise(double alpha, unsigned seed)
    : alpha_(alpha)
    , rng_(seed ? seed : std::random_device{}())
{
}

std::vector<double> DirichletNoise::generate(int n) const {
    std::vector<double> noise;
    noise.reserve(n);
    
    // Sample from Gamma(alpha, 1) distribution
    for (int i = 0; i < n; ++i) {
        noise.push_back(sample_gamma(alpha_));
    }
    
    // Normalize to sum to 1
    double sum = std::accumulate(noise.begin(), noise.end(), 0.0);
    if (sum > 0.0) {
        for (auto& v : noise) {
            v /= sum;
        }
    }
    
    return noise;
}

void DirichletNoise::apply(std::vector<double>& probabilities, double epsilon) const {
    if (probabilities.empty()) return;
    
    auto noise = generate(static_cast<int>(probabilities.size()));
    
    for (size_t i = 0; i < probabilities.size(); ++i) {
        probabilities[i] = (1.0 - epsilon) * probabilities[i] + epsilon * noise[i];
    }
}

double DirichletNoise::sample_gamma(double shape) const {
    // Marsaglia-Tsang method for Gamma sampling
    if (shape < 1.0) {
        // Use Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        return sample_gamma(shape + 1.0) * std::pow(
            std::uniform_real_distribution<double>(0.0, 1.0)(rng_), 1.0 / shape);
    }
    
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    
    while (true) {
        double x = std::normal_distribution<double>(0.0, 1.0)(rng_);
        double v = 1.0 + c * x;
        
        if (v <= 0.0) continue;
        
        v = v * v * v;
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
        
        if (u < 1.0 - 0.0331 * x * x * x * x) {
            return d * v;
        }
        
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
            return d * v;
        }
    }
}

// ============== Curriculum Manager ==============

CurriculumManager::CurriculumManager() : rng_(std::random_device{}()) {
    init_default_curriculum();
}

void CurriculumManager::init_default_curriculum() {
    add_gambit_openings();
    add_sharp_sicilians();
    add_endgame_studies();
    add_imbalanced_material();
    add_space_control_positions();
    add_king_safety_positions();
    add_tactical_storms();
}

void CurriculumManager::add_gambit_openings() {
    // King's Gambit - highly unbalanced, forces engine to attack/defend
    positions_.push_back({
        "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
        "King's Gambit Accepted",
        1800, true, true, 15
    });
    
    // Evans Gambit
    positions_.push_back({
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
        "Evans Gambit",
        1900, true, true, 15
    });
    
    // Danish Gambit
    positions_.push_back({
        "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 3",
        "Danish Gambit",
        1850, true, true, 15
    });
    
    // Budapest Gambit
    positions_.push_back({
        "rnbqkb1r/pppp1ppp/5n2/4p3/2P1P3/8/PP1P1PPP/RNBQKBNR w KQkq - 1 4",
        "Budapest Gambit",
        1750, true, true, 12
    });
}

void CurriculumManager::add_sharp_sicilians() {
    // Najdorf Poisoned Pawn
    positions_.push_back({
        "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq - 1 7",
        "Najdorf Sicilian",
        2200, true, true, 20
    });
    
    // Dragon Yugoslav Attack
    positions_.push_back({
        "r1bq1rk1/pp1n1pp1/2pb1n1p/3pp3/3PP3/2PB1N1P/PPQ2PP1/R1B2RK1 w - - 0 11",
        "Dragon Yugoslav",
        2100, true, true, 20
    });
    
    // Sveshnikov
    positions_.push_back({
        "r1bqkb1r/pp1p1ppp/2p2n2/4p3/3Pn3/3BPN2/PPP2PPP/RNBQK2R w KQkq - 2 7",
        "Sveshnikov Sicilian",
        2150, true, true, 18
    });
}

void CurriculumManager::add_endgame_studies() {
    // Lucena position - rook endgame
    positions_.push_back({
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "Lucena Position",
        2000, true, false, 30
    });
    
    // Philidor position - rook vs pawn
    positions_.push_back({
        "6k1/6p1/7K/8/8/8/8/1r6 b - - 0 1",
        "Philidor Position",
        1900, true, false, 30
    });
    
    // Opposition
    positions_.push_back({
        "8/8/8/3k4/8/3K4/8/8 w - - 0 1",
        "Opposition",
        1500, false, false, 25
    });
}

void CurriculumManager::add_imbalanced_material() {
    // Rook vs minor piece + pawns
    positions_.push_back({
        "r7/2k5/8/3p4/3P4/2N1K3/8/8 w - - 0 1",
        "Rook vs Knight+Pawns",
        1800, true, false, 25
    });
    
    // Two minors vs rook
    positions_.push_back({
        "8/4k3/8/3r4/8/2B1N3/4K3/8 w - - 0 1",
        "Two Minors vs Rook",
        1850, true, false, 25
    });
    
    // Queen vs rook+minor
    positions_.push_back({
        "4k3/8/8/3q4/8/3r1b2/8/4K3 w - - 0 1",
        "Queen vs Rook+Bishop",
        2000, true, true, 25
    });
}

void CurriculumManager::add_space_control_positions() {
    // White dominates center
    positions_.push_back({
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2BPP3/2P2N2/PP3PPP/RNBQK2R b KQkq - 0 5",
        "Center Domination",
        1700, false, false, 18
    });
    
    // Black cramped position - must find counterplay
    positions_.push_back({
        "r1bqk2r/ppp1ppbp/2np1np1/1N6/2PP4/4PN2/PP3PPP/R1BQKB1R b KQkq - 3 7",
        "Cramped Position",
        1900, false, true, 20
    });
}

void CurriculumManager::add_king_safety_positions() {
    // Castled kings with opposite side attacks
    positions_.push_back({
        "r4rk1/1pp1qppp/p1np1n2/1Bb1p3/4P3/2NP1N2/1PPQ1PPP/R4RK1 w - - 0 11",
        "Opposite Castling",
        2000, true, true, 18
    });
    
    // King in center - must castle or defend
    positions_.push_back({
        "rnb1kb1r/ppp2ppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        "King in Center",
        1700, true, true, 15
    });
}

void CurriculumManager::add_tactical_storms() {
    // Greek gift
    positions_.push_back({
        "r2q1rk1/ppp1bppp/2np1n2/4p3/2B1P3/3P1N2/PPP2PPP/R1BQ1RK1 w - - 0 9",
        "Greek Gift Setup",
        1850, true, true, 18
    });
    
    // Double rook sacrifice pattern
    positions_.push_back({
        "r3r1k1/ppp2ppp/2np1n2/1Bb1p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 0 9",
        "Sacrificial Pattern",
        1900, true, true, 18
    });
}

void CurriculumManager::add_positions(
    const std::vector<CurriculumPosition>& positions,
    PositionCategory /*category*/) {
    positions_.insert(positions_.end(), positions.begin(), positions.end());
}

std::string CurriculumManager::get_opening(double difficulty) const {
    if (positions_.empty()) {
        return "";  // Use standard starting position
    }
    
    // Filter positions by difficulty range
    std::vector<const CurriculumPosition*> candidates;
    
    double minElo = 1500 + difficulty * 500;  // 1500 to 2000+
    double maxElo = minElo + 400;
    
    for (const auto& pos : positions_) {
        if (pos.elo >= minElo && pos.elo <= maxElo) {
            candidates.push_back(&pos);
        }
    }
    
    // If no candidates in range, use all
    if (candidates.empty()) {
        for (const auto& pos : positions_) {
            candidates.push_back(&pos);
        }
    }
    
    // Weighted random: prefer unbalanced and tactical positions at higher difficulty
    std::vector<double> weights;
    for (const auto* pos : candidates) {
        double w = 1.0;
        if (difficulty > 0.5) {
            if (pos->unbalanced) w *= 2.0;
            if (pos->tactical) w *= 1.5;
        }
        weights.push_back(w);
    }
    
    // Weighted random selection
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    size_t idx = dist(rng_);
    
    return candidates[idx]->fen;
}

void CurriculumManager::update_difficulty(double current_elo_estimate) {
    // Map Elo to difficulty: 2500 -> 0.0, 3000 -> 1.0
    currentDifficulty_ = std::clamp((current_elo_estimate - 2500.0) / 500.0, 0.0, 1.0);
    
    // Adjust based on games played
    double games_factor = std::min(gamesPlayed_ / 100000.0, 1.0);
    currentDifficulty_ = std::max(currentDifficulty_, games_factor);
}

double CurriculumManager::get_temperature() const {
    // Higher temperature = more exploration
    // Decrease as difficulty increases
    return 1.5 - currentDifficulty_ * 0.8;
}

size_t CurriculumManager::total_positions() const {
    return positions_.size();
}

size_t CurriculumManager::positions_by_category(PositionCategory /*cat*/) const {
    // Would need to store category info per position
    return positions_.size() / 7;  // Approximate
}

void CurriculumManager::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "# Nexus Infinite Curriculum\n";
    file << "# difficulty=" << currentDifficulty_ << "\n";
    file << "# games=" << gamesPlayed_ << "\n\n";
    
    for (const auto& pos : positions_) {
        file << pos.fen << " | " << pos.name << " | "
             << pos.elo << " | " << (pos.unbalanced ? "U" : "")
             << (pos.tactical ? "T" : "") << "\n";
    }
}

void CurriculumManager::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    
    positions_.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        // Parse: FEN | name | elo | flags
        std::stringstream ss(line);
        std::string fen, name, elo_str, flags;
        
        std::getline(ss, fen, '|');
        std::getline(ss, name, '|');
        std::getline(ss, elo_str, '|');
        std::getline(ss, flags);
        
        // Trim
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(fen); trim(name); trim(elo_str); trim(flags);
        
        CurriculumPosition pos;
        pos.fen = fen;
        pos.name = name;
        pos.elo = std::stoi(elo_str);
        pos.unbalanced = flags.find('U') != std::string::npos;
        pos.tactical = flags.find('T') != std::string::npos;
        pos.minDepth = 15;
        
        positions_.push_back(pos);
    }
}

// ============== Self-Play Engine ==============

SelfPlayEngine::SelfPlayEngine(const SelfPlayConfig& config)
    : config_(config)
    , noise_(config.dirichletAlpha)
    , rng_(std::random_device{}())
{
}

void SelfPlayEngine::prepare_game(int game_number) {
    gamesPlayed_ = game_number;
    
    // Update curriculum difficulty
    curriculum_.update_difficulty(2500.0 + game_number / 100.0);
    
    // Select opening
    if (config_.useCurriculum) {
        currentOpening_ = curriculum_.get_opening(curriculum_.get_current_difficulty());
    }
}

std::string SelfPlayEngine::get_opening_fen() const {
    return currentOpening_;
}

void SelfPlayEngine::apply_root_noise(
    std::vector<std::pair<Move, double>>& move_scores) const {
    
    if (!config_.useDirichletNoise || move_scores.empty()) {
        return;
    }
    
    // Extract probabilities
    std::vector<double> probs;
    probs.reserve(move_scores.size());
    
    // Convert scores to probabilities using softmax
    double max_score = move_scores[0].second;
    for (const auto& p : move_scores) {
        max_score = std::max(max_score, p.second);
    }
    
    double sum = 0.0;
    for (const auto& p : move_scores) {
        double prob = std::exp((p.second - max_score) / 100.0);  // Temperature scaling
        probs.push_back(prob);
        sum += prob;
    }
    
    if (sum > 0.0) {
        for (auto& p : probs) {
            p /= sum;
        }
    }
    
    // Apply Dirichlet noise
    noise_.apply(probs, config_.dirichletEpsilon);
    
    // Convert back to scores
    for (size_t i = 0; i < move_scores.size(); ++i) {
        // Higher probability = higher score, but with noise
        move_scores[i].second = std::log(probs[i]) * 100.0 + max_score;
    }
}

double SelfPlayEngine::get_temperature(int ply) const {
    double temp = config_.temperatureInitial;
    
    // Decay temperature per ply
    for (int i = 0; i < ply; ++i) {
        temp *= config_.temperatureDecay;
    }
    
    return std::max(temp, config_.temperatureFloor);
}

bool SelfPlayEngine::should_adjudicate(
    const BoardState& /*pos*/, int ply, Value score) const {
    
    if (!config_.adjudicateEarly) {
        return false;
    }
    
    // Only adjudicate after minimum ply
    if (ply < config_.adjudicateMinPly) {
        return false;
    }
    
    // Check score threshold
    int abs_score = std::abs(score);
    if (abs_score >= config_.adjudicateScoreThreshold) {
        // Would need consecutive moves tracking
        // Simplified: just check score
        adjudications_++;
        return true;
    }
    
    return false;
}

bool SelfPlayEngine::should_avoid_repetition(int repetition_count) const {
    if (!config_.preventRepetition) {
        return false;
    }
    
    return repetition_count >= config_.maxRepetitions;
}

} // namespace SelfPlay
