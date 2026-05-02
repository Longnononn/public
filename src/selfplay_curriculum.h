#pragma once

#include "types.h"
#include "board.h"
#include <string>
#include <vector>
#include <random>
#include <functional>

namespace SelfPlay {

// ======== Opening Position Database ========
// Unbalanced / sharp / "weird" positions to force engine out of draw-death

struct CurriculumPosition {
    std::string fen;
    std::string name;
    int elo;           // Estimated difficulty
    bool unbalanced;   // Material or space imbalance
    bool tactical;     // Known sharp lines
    int minDepth;      // Minimum search depth before adjudication
};

// Categories of curriculum positions
enum class PositionCategory {
    GAMBIT,         // Sacrifice openings (e.g. King's Gambit)
    SHARP_SICILIAN, // Najdorf, Dragon, etc.
    ENDGAME_STUDY,  // Winning/drawing studies
    IMBALANCED,     // Material imbalance (e.g. rook vs minor piece)
    SPACE_CONTROL,  // One side dominates space
    KING_SAFETY,    // Exposed king positions
    TACTICAL_STORM, // Known attacking positions
    CUSTOM_MINED,   // Positions mined from hard positions
};

// ======== Dirichlet Noise ========
// Applied to root move probabilities to encourage exploration

class DirichletNoise {
public:
    DirichletNoise(double alpha = 0.3, unsigned seed = 0);
    
    // Generate noise vector for N moves
    std::vector<double> generate(int n) const;
    
    // Blend policy with noise: p' = (1-eps)*p + eps*noise
    void apply(std::vector<double>& probabilities, double epsilon = 0.25) const;
    
private:
    double alpha_;
    mutable std::mt19937 rng_;
    
    // Gamma distribution for Dirichlet sampling
    double sample_gamma(double shape) const;
};

// ======== Curriculum Manager ========
// Manages progressive difficulty and position diversity

class CurriculumManager {
public:
    CurriculumManager();
    
    // Initialize with built-in curriculum
    void init_default_curriculum();
    
    // Add positions from hard-mining or external sources
    void add_positions(const std::vector<CurriculumPosition>& positions,
                       PositionCategory category);
    
    // Get opening position for self-play game
    // difficulty: 0.0 = easy/balanced, 1.0 = extremely sharp/unbalanced
    std::string get_opening(double difficulty) const;
    
    // Progressive curriculum: increase difficulty as network improves
    void update_difficulty(double current_elo_estimate);
    double get_current_difficulty() const { return currentDifficulty_; }
    
    // Get temperature for move selection (higher = more random)
    // Decreases as network gets stronger
    double get_temperature() const;
    
    // Position statistics
    size_t total_positions() const;
    size_t positions_by_category(PositionCategory cat) const;
    
    // Export/import curriculum
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    
private:
    std::vector<CurriculumPosition> positions_;
    std::vector<CurriculumPosition> minedPositions_;
    
    double currentDifficulty_ = 0.0;  // 0.0 to 1.0
    int gamesPlayed_ = 0;
    int currentPhase_ = 0;  // Progressive phases
    
    // Built-in sharp openings
    void add_gambit_openings();
    void add_sharp_sicilians();
    void add_endgame_studies();
    void add_imbalanced_material();
    void add_space_control_positions();
    void add_king_safety_positions();
    void add_tactical_storms();
    
    // Weighted random selection
    mutable std::mt19937 rng_;
};

// ======== Self-Play Configuration ========
// Controls exploration vs exploitation in self-play

struct SelfPlayConfig {
    // Dirichlet noise at root
    bool useDirichletNoise = true;
    double dirichletAlpha = 0.3;
    double dirichletEpsilon = 0.25;
    
    // Curriculum learning
    bool useCurriculum = true;
    double initialDifficulty = 0.0;
    double difficultyIncrement = 0.02;  // Per 1000 games
    double maxDifficulty = 1.0;
    
    // Temperature for move selection
    double temperatureInitial = 1.5;    // High at root
    double temperatureDecay = 0.95;     // Decay per ply
    double temperatureFloor = 0.2;      // Minimum temp
    
    // Forced randomness in opening
    int forcedRandomPly = 8;            // Random moves in first N plies
    double randomMoveProb = 0.25;         // Probability of random move
    
    // Adjudication to avoid draw-death
    bool adjudicateEarly = true;
    int adjudicateScoreThreshold = 600;  // cp
    int adjudicateMinPly = 30;
    int adjudicateConsecutiveMoves = 5;
    
    // Prevent repetition draws
    bool preventRepetition = true;
    int maxRepetitions = 2;             // Allow only 2-fold before forcing change
    
    // Curriculum categories to use
    std::vector<PositionCategory> categories = {
        PositionCategory::GAMBIT,
        PositionCategory::SHARP_SICILIAN,
        PositionCategory::IMBALANCED,
        PositionCategory::TACTICAL_STORM,
    };
};

// ======== Self-Play Engine Integration ========
// Hooks into search to apply curriculum and noise

class SelfPlayEngine {
public:
    SelfPlayEngine(const SelfPlayConfig& config = SelfPlayConfig{});
    
    // Call before starting a self-play game
    void prepare_game(int game_number);
    
    // Get opening position (if any)
    std::string get_opening_fen() const;
    
    // Apply Dirichlet noise to root move scores
    // Modifies the move scores in-place to encourage exploration
    void apply_root_noise(std::vector<std::pair<Move, double>>& move_scores) const;
    
    // Get temperature for current ply
    double get_temperature(int ply) const;
    
    // Check if position should be adjudicated
    bool should_adjudicate(const BoardState& pos, int ply, Value score) const;
    
    // Check if we should force non-repetition
    bool should_avoid_repetition(int repetition_count) const;
    
    // Statistics
    int games_played() const { return gamesPlayed_; }
    int draws_avoided() const { return drawsAvoided_; }
    int adjudications() const { return adjudications_; }
    
private:
    SelfPlayConfig config_;
    CurriculumManager curriculum_;
    DirichletNoise noise_;
    
    int gamesPlayed_ = 0;
    int drawsAvoided_ = 0;
    int adjudications_ = 0;
    std::string currentOpening_;
    
    mutable std::mt19937 rng_;
};

} // namespace SelfPlay
