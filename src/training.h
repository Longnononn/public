#pragma once

#include "types.h"
#include "board.h"
#include "move.h"
#include <string>
#include <vector>
#include <fstream>
#include <mutex>

namespace Training {

// Training data format (compatible with Stockfish/binpack)
#pragma pack(push, 1)
struct PackedPosition {
    // Occupancy bitboards (16 bytes)
    uint64_t whitePieces;
    uint64_t blackPieces;
    
    // King positions (2 bytes)
    uint8_t whiteKing;
    uint8_t blackKing;
    
    // Score (2 bytes)
    int16_t score;
    
    // Result: 0=loss, 1=draw, 2=win (1 byte)
    uint8_t result;
    
    // Full move counter (2 bytes)
    uint16_t move50;
    
    padding to 32 bytes for alignment
    uint8_t padding[9];
};
static_assert(sizeof(PackedPosition) == 32, "PackedPosition must be 32 bytes");
#pragma pack(pop)

// Training position with full info
struct TrainingPosition {
    std::string fen;
    Value score;           // Search score or qsearch score
    int result;            // Game outcome
    int move50;           // 50-move counter
    int ply;              // Ply from start
    bool inCheck;
    int gamePhase;        // Opening=1, Middlegame=2, Endgame=3
    
    // Metadata
    std::string gameId;
    int moveNumber;
};

// Self-play game
struct GameResult {
    std::vector<TrainingPosition> positions;
    int result;  // -1=loss, 0=draw, 1=win (from white's perspective)
    int length;
    bool adjudicated;
};

// Data generator
class DataGenerator {
public:
    DataGenerator();
    
    // Configuration
    void set_search_depth(int depth) { searchDepth = depth; }
    void set_num_games(int games) { numGames = games; }
    void set_threads(int threads) { numThreads = threads; }
    void set_output_file(const std::string& file) { outputFile = file; }
    
    // Generation methods
    void generate_selfplay();
    void generate_from_pgn(const std::string& pgnFile);
    void augment_positions();
    void filter_hard_positions();
    
    // Data processing
    void pack_to_binpack(const std::string& output);
    void export_to_plain(const std::string& output);
    void deduplicate();
    void filter_duplicates();
    void validate_dataset();
    
    // Statistics
    struct Stats {
        u64 totalPositions;
        u64 gamesGenerated;
        u64 wins;
        u64 draws;
        u64 losses;
        double avgGameLength;
        int adjudicatedGames;
    };
    Stats get_stats() const { return stats; }
    void print_stats();
    
private:
    int searchDepth = 8;        // Depth for scoring positions
    int numGames = 10000;       // Games to generate
    int numThreads = 1;         // Parallel generation
    std::string outputFile = "training_data.txt";
    
    Stats stats;
    std::vector<TrainingPosition> allPositions;
    std::mutex dataMutex;
    
    // Generation helpers
    GameResult play_one_game(int threadId);
    Value score_position(const BoardState& pos);
    int evaluate_game_phase(const BoardState& pos);
    bool should_adjudicate(const BoardState& pos, int ply);
    bool is_duplicate(const TrainingPosition& pos);
    
    // Thread worker
    void worker_thread(int threadId, int gamesToPlay);
    
    // Output
    void save_position(const TrainingPosition& pos);
    void flush_buffer();
    std::vector<TrainingPosition> buffer;
    static constexpr int BUFFER_SIZE = 1000;
};

// Data filtering and cleaning
class DataCleaner {
public:
    // Filters
    static void remove_blunders(std::vector<TrainingPosition>& data, int threshold);
    static void remove_early_moves(std::vector<TrainingPosition>& data, int movesToSkip);
    static void remove_late_endgames(std::vector<TrainingPosition>& data);
    static void balance_results(std::vector<TrainingPosition>& data);
    
    // Augmentation
    static void add_flipped_positions(std::vector<TrainingPosition>& data);
    static void add_symmetric_positions(std::vector<TrainingPosition>& data);
    
    // Validation
    static bool is_valid_position(const TrainingPosition& pos);
    static bool has_illegal_moves(const std::string& fen);
};

// EPD format for positions with known outcomes
struct EPDEntry {
    std::string fen;
    std::string bestMove;
    std::string id;
    int elo;
    std::vector<std::string> avoidMoves;
};

std::vector<EPDEntry> load_epd_file(const std::string& filename);
void save_epd_file(const std::string& filename, const std::vector<EPDEntry>& entries);

} // namespace Training
