#pragma once

#include "types.h"
#include "board.h"
#include "search.h"
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace Nexus {

// Thread-local search state
struct ThreadState {
    int threadId;
    Search search;
    BoardState board;  // Local copy
    SearchLimits limits;
    std::atomic<bool> searching{false};
    std::atomic<bool> exit{false};
    
    // Statistics
    std::atomic<u64> nodes{0};
    std::atomic<int> tbHits{0};
};

// Lazy SMP Thread Pool
class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();
    
    // Initialize with n threads
    void init(size_t numThreads);
    void shutdown();
    
    // Start parallel search
    void start_search(BoardState& pos, const SearchLimits& limits);
    void stop_search();
    bool is_searching() const { return searching.load(); }
    
    // Get results
    Move best_move() const;
    Value best_score() const;
    u64 total_nodes() const;
    
    // Thread management
    size_t size() const { return threads.size(); }
    void set_size(size_t n);
    
    // Access individual threads
    ThreadState* get_thread(size_t idx);
    
    // Main thread helpers
    void wake_helper_threads();
    void sleep_helper_threads();
    
private:
    std::vector<ThreadState*> threadData;
    std::vector<std::thread> threads;
    
    std::mutex mtx;
    std::condition_variable cv;
    
    std::atomic<bool> searching{false};
    std::atomic<bool> stop{false};
    std::atomic<int> threadsSearching{0};
    
    void worker_loop(size_t threadId);
    void search_loop(ThreadState* th);
};

// Global thread pool
extern ThreadPool Threads;

// Helper functions for thread affinity
void set_thread_affinity(std::thread& th, int cpu);
void set_current_thread_affinity(int cpu);

} // namespace Nexus
