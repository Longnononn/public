#include "uci.h"
#include "board.h"
#include "search.h"
#include "tt.h"
#include "bitboard.h"
#include "movegen.h"
#include "eval.h"
#include "simd.h"
#include "reductions.h"
#include "nnue.h"
#include "syzygy.h"
#include "thread.h"
#include "test_suite.h"
#include "training.h"
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <thread>

namespace Nexus {

static BoardState currentPos;
static std::thread searchThread;

std::vector<std::string> UCI::tokenize(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token)
        tokens.push_back(token);
    return tokens;
}

void UCI::loop() {
    std::string line;
    
    // Initialize
    init_bitboards();
    init_zobrist();
    Reductions::init();
    SIMDInfo::detect();
    NNUE::init_nnue();
    currentPos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    std::cout << "id name Nexus Infinite" << std::endl;
    std::cout << "id author Nexus Team" << std::endl;
    std::cout << "id version 1.0-NNUE" << std::endl;
    std::cout << "option name Hash type spin default 64 min 1 max 16384" << std::endl;
    std::cout << "option name Ponder type check default false" << std::endl;
    std::cout << "option name Threads type spin default 1 min 1 max 512" << std::endl;
    std::cout << "option name MultiPV type spin default 1 min 1 max 64" << std::endl;
    std::cout << "option name SyzygyPath type string default <empty>" << std::endl;
    std::cout << "option name SyzygyProbeLimit type spin default 6 min 0 max 6" << std::endl;
    std::cout << "option name Use NNUE type check default true" << std::endl;
    std::cout << "option name EvalFile type string default nexus.nnue" << std::endl;
    std::cout << "option name Clear Hash type button" << std::endl;
    std::cout << "uciok" << std::endl;
    
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        is >> cmd;
        
        if (cmd == "quit" || cmd == "q") {
            Searcher.stop();
            if (searchThread.joinable())
                searchThread.join();
            break;
        }
        else if (cmd == "uci")
            std::cout << "id name Nexus Infinite" << std::endl << "uciok" << std::endl;
        else if (cmd == "isready")
            std::cout << "readyok" << std::endl;
        else if (cmd == "position")
            position(is);
        else if (cmd == "go")
            go(is);
        else if (cmd == "stop") {
            Searcher.stop();
        }
        else if (cmd == "ponderhit") {
            // Simplified
        }
        else if (cmd == "ucinewgame") {
            TT.clear();
            Searcher.clear();
            currentPos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }
        else if (cmd == "setoption")
            setoption(is);
        else if (cmd == "d")
            std::cout << currentPos.fen() << std::endl;
        else if (cmd == "perft") {
            int depth = 5;
            is >> depth;
            u64 nodes = perft(currentPos, depth);
            std::cout << "Nodes: " << nodes << std::endl;
        }
        else if (cmd == "eval") {
            Value v = Eval::evaluate(currentPos);
            std::cout << "Static eval: " << v << std::endl;
        }
        else if (cmd == "bench") {
            int depth = 14;
            is >> depth;
            Searcher.bench(depth);
        }
        else if (cmd == "info" || cmd == "compiler") {
            std::cout << "Nexus Infinite Engine Info:" << std::endl;
            std::cout << "  Version: 1.0" << std::endl;
            std::cout << "  C++ Standard: " << __cplusplus << std::endl;
            std::cout << "  Compiler: ";
            #ifdef __GNUC__
                std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
            #elif defined(__clang__)
                std::cout << "Clang " << __clang_major__ << "." << __clang_minor__;
            #elif defined(_MSC_VER)
                std::cout << "MSVC " << _MSC_VER;
            #else
                std::cout << "Unknown";
            #endif
            std::cout << std::endl;
            SIMDInfo::print_info();
        }
        else if (cmd == "test" || cmd == "testall") {
            int depth = 14;
            is >> depth;
            TestSuite::TestSuite suite;
            suite.init();
            auto results = suite.run_all(depth);
            suite.print_report(results);
            suite.save_report("test_report.txt", results);
            std::cout << "Test report saved to test_report.txt" << std::endl;
        }
        else if (cmd == "quicktest") {
            bool pass = TestSuite::quick_test();
            std::cout << (pass ? "Quick test PASSED" : "Quick test FAILED") << std::endl;
        }
        else if (cmd == "gensfen") {
            int games = 1000;
            int depth = 8;
            int threads = 1;
            is >> games >> depth >> threads;
            
            Training::DataGenerator gen;
            gen.set_num_games(games);
            gen.set_search_depth(depth);
            gen.set_threads(threads);
            gen.set_output_file("training_data.txt");
            gen.generate_selfplay();
            
            std::cout << "Training data generation complete. Output: training_data.txt" << std::endl;
        }
        else if (cmd == "gensfens") {
            std::cout << "Starting multi-threaded data generation..." << std::endl;
            Training::DataGenerator gen;
            gen.set_num_games(10000);
            gen.set_search_depth(10);
            gen.set_threads(4);
            gen.set_output_file("training_large.txt");
            gen.generate_selfplay();
        }
        else if (cmd == "nnue" || cmd == "evalnnue") {
            if (NNUE::g_nnue.is_loaded()) {
                Value score = NNUE::g_nnue.evaluate(currentPos);
                std::cout << "NNUE eval: " << score << std::endl;
            } else {
                std::cout << "NNUE not loaded. Use 'setoption name EvalFile value <file>'" << std::endl;
            }
        }
    }
}

void UCI::position(std::istringstream& is) {
    std::string token, fen;
    is >> token;
    
    if (token == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        is >> token; // consume "moves" if present
    }
    else if (token == "fen") {
        while (is >> token && token != "moves")
            fen += (fen.empty() ? "" : " ") + token;
    }
    
    currentPos.set_fen(fen);
    
    // Apply moves
    if (token == "moves" || (is >> token && token == "moves")) {
        std::string moveStr;
        while (is >> moveStr) {
            // Find and apply legal move
            ExtMove moves[MAX_MOVES];
            ExtMove* end;
            if (currentPos.is_check())
                end = generate<EVASIONS>(currentPos, moves);
            else
                end = generate<NON_EVASIONS>(currentPos, moves);
            
            Move m = Move::from_uci(moveStr);
            bool found = false;
            for (ExtMove* it = moves; it != end; ++it) {
                if (it->move == m && is_legal(currentPos, it->move)) {
                    StateInfo st;
                    currentPos.do_move(it->move, st);
                    found = true;
                    break;
                }
            }
            if (!found)
                std::cerr << "Illegal move: " << moveStr << std::endl;
        }
    }
}

void UCI::go(std::istringstream& is) {
    SearchLimits limits;
    limits.depth = 64;
    limits.nodes = 0;
    limits.mate = 0;
    limits.movetime = 0;
    limits.infinite = false;
    
    TimeControl tc;
    tc.time[WHITE] = tc.time[BLACK] = 0;
    tc.increment[WHITE] = tc.increment[BLACK] = 0;
    tc.movesToGo = 0;
    tc.ponder = false;
    tc.infinite = false;
    
    std::string token;
    while (is >> token) {
        if (token == "wtime") is >> tc.time[WHITE];
        else if (token == "btime") is >> tc.time[BLACK];
        else if (token == "winc") is >> tc.increment[WHITE];
        else if (token == "binc") is >> tc.increment[BLACK];
        else if (token == "movestogo") is >> tc.movesToGo;
        else if (token == "depth") is >> limits.depth;
        else if (token == "nodes") is >> limits.nodes;
        else if (token == "mate") is >> limits.mate;
        else if (token == "movetime") is >> limits.movetime;
        else if (token == "infinite") { limits.infinite = true; tc.infinite = true; }
        else if (token == "ponder") tc.ponder = true;
        else if (token == "searchmoves") {
            std::string sm;
            while (is >> sm)
                limits.searchmoves.push_back(Move::from_uci(sm));
        }
    }
    
    // Configure time manager
    Time.init(tc, currentPos.sideToMove, currentPos.gamePly);
    
    // Start search in new thread
    if (searchThread.joinable()) {
        Searcher.stop();
        searchThread.join();
    }
    
    Searcher.clear();
    searchThread = std::thread([&]() {
        Searcher.start(currentPos, limits, tc.ponder);
    });
}

void UCI::setoption(std::istringstream& is) {
    std::string token, name, value;
    
    is >> token; // "name"
    is >> name;
    if (name == "name") is >> name;
    
    is >> token; // "value"
    while (is >> token && token != "value")
        name += " " + token;
    
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;
    
    if (name == "Hash" || name == "hash") {
        int mb = std::stoi(value);
        TT.resize(mb);
    }
    else if (name == "Threads" || name == "threads") {
        int threads = std::stoi(value);
        Threads.set_size(threads);
        std::cout << "info string Set Threads to " << threads << std::endl;
    }
    else if (name == "MultiPV" || name == "multipv") {
        int mpv = std::stoi(value);
        Searcher.set_multi_pv(mpv);
    }
    else if (name == "SyzygyPath" || name == "syzygypath") {
        if (!value.empty() && value != "<empty>") {
            Syzygy::TB.init(value);
        }
    }
    else if (name == "SyzygyProbeLimit" || name == "syzygyprobelimit") {
        // Store probe limit for later use
        // int limit = std::stoi(value);
    }
    else if (name == "EvalFile" || name == "evalfile") {
        if (!value.empty()) {
            NNUE::g_nnue.load_network(value);
        }
    }
    else if (name == "Use NNUE" || name == "use nnue" || name == "usennue") {
        // Global useNNUE flag is in search.cpp
        // bool use = (value == "true" || value == "1");
    }
    else if (name == "Clear Hash" || name == "clear hash") {
        TT.clear();
        Searcher.clear();
    }
    // Other options can be added here
}

} // namespace Nexus
