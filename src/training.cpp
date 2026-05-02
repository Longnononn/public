#include "training.h"
#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "tt.h"
#include "thread.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>

namespace Training {

// ================ DataGenerator Implementation ================

DataGenerator::DataGenerator() = default;

void DataGenerator::generate_selfplay() {
    std::cout << "Starting self-play data generation...\n";
    std::cout << "Games: " << numGames << ", Threads: " << numThreads << ", Depth: " << searchDepth << "\n";
    
    stats = Stats{};
    
    if (numThreads == 1) {
        // Single-threaded generation
        for (int i = 0; i < numGames; ++i) {
            auto game = play_one_game(0);
            
            // Add positions
            std::lock_guard<std::mutex> lock(dataMutex);
            for (auto& pos : game.positions) {
                allPositions.push_back(pos);
                save_position(pos);
            }
            
            // Update stats
            stats.gamesGenerated++;
            if (game.result == 1) stats.wins++;
            else if (game.result == 0) stats.draws++;
            else stats.losses++;
            
            if ((i + 1) % 100 == 0) {
                std::cout << "Generated " << (i + 1) << " games, " 
                          << allPositions.size() << " positions\n";
            }
        }
    } else {
        // Multi-threaded generation
        std::vector<std::thread> workers;
        int gamesPerThread = numGames / numThreads;
        
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back(&DataGenerator::worker_thread, this, i, gamesPerThread);
        }
        
        for (auto& t : workers) {
            t.join();
        }
    }
    
    flush_buffer();
    print_stats();
}

void DataGenerator::worker_thread(int threadId, int gamesToPlay) {
    for (int i = 0; i < gamesToPlay; ++i) {
        auto game = play_one_game(threadId);
        
        std::lock_guard<std::mutex> lock(dataMutex);
        for (auto& pos : game.positions) {
            allPositions.push_back(pos);
            buffer.push_back(pos);
            
            if (buffer.size() >= BUFFER_SIZE) {
                flush_buffer();
            }
        }
        
        stats.gamesGenerated++;
        if (game.result == 1) stats.wins++;
        else if (game.result == 0) stats.draws++;
        else stats.losses++;
    }
}

GameResult DataGenerator::play_one_game(int threadId) {
    GameResult result;
    result.result = 0;
    result.adjudicated = false;
    
    BoardState board;
    board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // Random opening for diversity
    std::mt19937 rng(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count() + threadId));
    
    int randomMoves = rng() % 8 + 2;  // 2-10 random opening moves
    for (int i = 0; i < randomMoves; ++i) {
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<LEGAL>(board, moves);
        int moveCount = end - moves;
        
        if (moveCount == 0) break;
        
        // Pick random move
        int idx = rng() % moveCount;
        StateInfo st;
        board.do_move(moves[idx].move, st);
    }
    
    // Now play with engine
    std::string gameId = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    int moveCounter = 0;
    
    while (moveCounter < 200) {
        // Check adjudication
        if (should_adjudicate(board, moveCounter)) {
            result.adjudicated = true;
            break;
        }
        
        // Generate legal moves
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<LEGAL>(board, moves);
        
        if (end == moves) break;  // No legal moves
        
        // Evaluate position
        Value score = score_position(board);
        
        // Save training position
        TrainingPosition tpos;
        tpos.fen = board.fen();
        tpos.score = score;
        tpos.result = 0;  // Will be set at game end
        tpos.move50 = board.st.rule50;
        tpos.ply = board.st.pliesFromNull;
        tpos.inCheck = board.is_check();
        tpos.gamePhase = evaluate_game_phase(board);
        tpos.gameId = gameId;
        tpos.moveNumber = moveCounter;
        
        result.positions.push_back(tpos);
        
        // Make best move (with some randomness for exploration)
        Search searcher;
        searcher.clear();
        
        SearchLimits limits;
        limits.depth = searchDepth;
        limits.movetime = 100;  // 100ms per move
        
        // Start search
        searcher.start(board, limits, false);
        
        // Wait for search
        while (searcher.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        Move bestMove = searcher.best_move();
        if (!bestMove.is_ok()) {
            // Fallback to first legal move
            bestMove = moves[0].move;
        }
        
        // Make the move
        StateInfo st;
        board.do_move(bestMove, st);
        
        moveCounter++;
    }
    
    // Determine result
    ExtMove finMoves[MAX_MOVES];
    ExtMove* finEnd = generate<LEGAL>(board, finMoves);
    if (finEnd == finMoves && board.is_check()) {
        result.result = (board.sideToMove == BLACK) ? 1 : -1;  // White wins if black mated
    } else if (finEnd == finMoves || board.is_draw(0)) {
        result.result = 0;
    } else if (result.adjudicated) {
        // Determine by material or position eval
        Value eval = Eval::evaluate(board);
        if (eval > VALUE_KNOWN_WIN) result.result = 1;
        else if (eval < -VALUE_KNOWN_WIN) result.result = -1;
        else result.result = 0;
    }
    
    // Set results for all positions
    for (auto& pos : result.positions) {
        pos.result = result.result;
    }
    
    result.length = moveCounter;
    return result;
}

Value DataGenerator::score_position(const BoardState& pos) {
    // Quick qsearch for accurate score
    if (pos.is_check()) {
        // In check - need deeper search
        Search searcher;
        searcher.clear();
        
        SearchLimits limits;
        limits.depth = 6;
        
        BoardState posCopy = pos;
        searcher.start(posCopy, limits, false);
        while (searcher.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        return searcher.best_score();
    } else {
        // Use static eval + quick qsearch
        return Eval::evaluate(pos);
    }
}

int DataGenerator::evaluate_game_phase(const BoardState& pos) {
    // Simple phase detection based on material
    int material = 0;
    for (int sq_i = SQ_A1; sq_i <= SQ_H8; ++sq_i) {
        Square sq = Square(sq_i);
        Piece pc = pos.piece_on(sq);
        if (pc != NO_PIECE) {
            PieceType pt = type_of(pc);
            if (pt == QUEEN) material += 9;
            else if (pt == ROOK) material += 5;
            else if (pt == BISHOP || pt == KNIGHT) material += 3;
            else if (pt == PAWN) material += 1;
        }
    }
    
    if (material >= 60) return 1;  // Opening
    if (material >= 30) return 2;  // Middlegame
    return 3;  // Endgame
}

bool DataGenerator::should_adjudicate(const BoardState& pos, int ply) {
    // Adjudicate by score after many moves
    if (ply < 50) return false;
    
    Value eval = Eval::evaluate(pos);
    
    // Adjudicate if clearly won/lost
    if (std::abs(eval) > VALUE_MATE_IN_MAX_PLY) return true;
    
    // Adjudicate draws
    if (pos.is_material_draw()) return true;
    if (pos.st.rule50 > 95) return true;  // Near 50-move rule
    
    return false;
}

void DataGenerator::save_position(const TrainingPosition& pos) {
    buffer.push_back(pos);
    
    if (buffer.size() >= BUFFER_SIZE) {
        flush_buffer();
    }
}

void DataGenerator::flush_buffer() {
    if (buffer.empty()) return;
    
    std::ofstream file(outputFile, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << outputFile << "\n";
        return;
    }
    
    for (const auto& pos : buffer) {
        file << pos.fen << " | " << pos.score << " | " << pos.result << " | " 
             << pos.gamePhase << " | " << pos.gameId << "\n";
    }
    
    buffer.clear();
}

void DataGenerator::print_stats() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "Training Data Generation Complete\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << "Games generated: " << stats.gamesGenerated << "\n";
    std::cout << "Total positions: " << allPositions.size() << "\n";
    std::cout << "Wins: " << stats.wins << " (" << (100.0 * stats.wins / stats.gamesGenerated) << "%)\n";
    std::cout << "Draws: " << stats.draws << " (" << (100.0 * stats.draws / stats.gamesGenerated) << "%)\n";
    std::cout << "Losses: " << stats.losses << " (" << (100.0 * stats.losses / stats.gamesGenerated) << "%)\n";
    std::cout << "Avg positions/game: " << (allPositions.size() / (double)stats.gamesGenerated) << "\n";
    std::cout << "Output: " << outputFile << "\n";
    std::cout << std::string(50, '=') << "\n";
}

// ================ Data Cleaning ================

void DataCleaner::remove_blunders(std::vector<TrainingPosition>& data, int threshold) {
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [threshold](const TrainingPosition& pos) {
                return std::abs(pos.score) > threshold;
            }),
        data.end()
    );
}

void DataCleaner::remove_early_moves(std::vector<TrainingPosition>& data, int movesToSkip) {
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [movesToSkip](const TrainingPosition& pos) {
                return pos.moveNumber < movesToSkip;
            }),
        data.end()
    );
}

void DataCleaner::balance_results(std::vector<TrainingPosition>& data) {
    // Count each result type
    int wins = 0, draws = 0, losses = 0;
    for (const auto& pos : data) {
        if (pos.result == 1) wins++;
        else if (pos.result == 0) draws++;
        else losses++;
    }
    
    // Find minimum
    int minCount = std::min({wins, draws, losses});
    
    // Shuffle and limit each category
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    
    int w = 0, d = 0, l = 0;
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [&w, &d, &l, minCount](const TrainingPosition& pos) {
                if (pos.result == 1 && w >= minCount) return true;
                if (pos.result == 0 && d >= minCount) return true;
                if (pos.result == -1 && l >= minCount) return true;
                
                if (pos.result == 1) w++;
                else if (pos.result == 0) d++;
                else l++;
                
                return false;
            }),
        data.end()
    );
}

void DataCleaner::add_flipped_positions(std::vector<TrainingPosition>& data) {
    size_t originalSize = data.size();
    data.reserve(originalSize * 2);
    
    for (size_t i = 0; i < originalSize; ++i) {
        auto flipped = data[i];
        // Flip board vertically and swap colors
        // This would require actual implementation
        data.push_back(flipped);
    }
}

// ================ EPD Operations ================

std::vector<EPDEntry> load_epd_file(const std::string& filename) {
    std::vector<EPDEntry> entries;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open EPD file: " << filename << "\n";
        return entries;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        EPDEntry entry;
        // Parse EPD format
        // FEN + operations
        size_t space = line.find(' ');
        if (space == std::string::npos) continue;
        
        entry.fen = line.substr(0, space);
        // Parse operations (bm, am, id, etc.)
        
        entries.push_back(entry);
    }
    
    return entries;
}

void save_epd_file(const std::string& filename, const std::vector<EPDEntry>& entries) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to create EPD file: " << filename << "\n";
        return;
    }
    
    for (const auto& entry : entries) {
        file << entry.fen;
        if (!entry.bestMove.empty()) file << " bm " << entry.bestMove << ";";
        if (!entry.id.empty()) file << " id \"" << entry.id << "\";";
        file << "\n";
    }
}

} // namespace Training
