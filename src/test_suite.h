#pragma once

#include "types.h"
#include "board.h"
#include "move.h"
#include <string>
#include <vector>

namespace TestSuite {

using namespace Nexus;

// Test result
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    u64 nodes;
    int timeMs;
    Value score;
    Move bestMove;
};

// Test position database
struct TestPosition {
    const char* name;
    const char* fen;
    const char* bestMove;  // Expected best move (if known)
    int depth;
    u64 expectedNodes;     // For perft
    Value expectedScore;   // For eval tests
    int timeLimit;         // Time limit in ms
};

// Complete test suite
class TestSuite {
public:
    // Initialize test positions
    void init();
    
    // Run all tests
    std::vector<TestResult> run_all(int depth = 14);
    
    // Individual test suites
    std::vector<TestResult> run_perft_tests();
    std::vector<TestResult> run_bench_suite(int depth);
    std::vector<TestResult> run_tactical_tests();
    std::vector<TestResult> run_endgame_tests();
    std::vector<TestResult> run_regression_tests();
    
    // Report generation
    void print_report(const std::vector<TestResult>& results);
    bool save_report(const std::string& filename, 
                     const std::vector<TestResult>& results);
    
private:
    
    std::vector<TestPosition> perftPositions;
    std::vector<TestPosition> tacticalPositions;
    std::vector<TestPosition> endgamePositions;
    std::vector<TestPosition> regressionPositions;
    
    void load_perft_positions();
    void load_tactical_positions();
    void load_endgame_positions();
    void load_regression_positions();
    
    // Helper
    TestResult run_single_test(const TestPosition& pos);
    bool verify_perft(const BoardState& pos, int depth, u64 expected);
    bool verify_search(const BoardState& pos, int depth, 
                       const std::string& expectedMove);
};

// Quick test runner
bool quick_test();  // Returns true if all critical tests pass

// Benchmark comparison
struct BenchmarkResult {
    u64 totalNodes;
    int totalTime;
    double avgNPS;
    std::vector<std::pair<std::string, double>> positionNPS;
};

BenchmarkResult run_benchmark(int depth = 14);

} // namespace TestSuite
