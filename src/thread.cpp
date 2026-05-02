#include "thread.h"
#include "tt.h"
#include "timeman.h"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace Nexus {

ThreadPool Threads;

// ================ Thread Affinity ================

void set_thread_affinity(std::thread& th, int cpu) {
#ifdef _WIN32
    HANDLE handle = th.native_handle();
    DWORD_PTR mask = 1ULL << cpu;
    SetThreadAffinityMask(handle, mask);
#else
    pthread_t handle = th.native_handle();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(handle, sizeof(cpuset), &cpuset);
#endif
}

void set_current_thread_affinity(int cpu) {
#ifdef _WIN32
    DWORD_PTR mask = 1ULL << cpu;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

// ================ ThreadPool Implementation ================

ThreadPool::ThreadPool() = default;

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::init(size_t numThreads) {
    shutdown();  // Clean up any existing threads
    
    if (numThreads == 0) numThreads = 1;
    
    std::cout << "Lazy SMP: Starting " << numThreads << " threads" << std::endl;
    
    // Create thread states
    for (size_t i = 0; i < numThreads; ++i) {
        ThreadState* th = new ThreadState();
        th->threadId = i;
        th->searching.store(false);
        th->exit.store(false);
        th->nodes.store(0);
        threadData.push_back(th);
    }
    
    // Launch worker threads
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(&ThreadPool::worker_loop, this, i);
        
        // Set affinity if multiple threads
        if (numThreads > 1) {
            set_thread_affinity(threads.back(), i);
        }
    }
}

void ThreadPool::shutdown() {
    // Signal all threads to exit
    stop.store(true);
    searching.store(false);
    
    for (auto& th : threadData) {
        th->exit.store(true);
        th->searching.store(false);
    }
    
    // Wake all waiting threads
    cv.notify_all();
    
    // Join all threads
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    
    threads.clear();
    
    // Clean up thread data
    for (auto& th : threadData) {
        delete th;
    }
    threadData.clear();
    
    stop.store(false);
    threadsSearching.store(0);
}

void ThreadPool::set_size(size_t n) {
    if (n == 0) n = 1;
    if (n == threadData.size()) return;
    
    shutdown();
    init(n);
}

ThreadState* ThreadPool::get_thread(size_t idx) {
    if (idx < threadData.size()) {
        return threadData[idx];
    }
    return nullptr;
}

void ThreadPool::worker_loop(size_t threadId) {
    ThreadState* th = threadData[threadId];
    
    // Set thread name (for debugging)
    #ifdef _WIN32
    // Windows thread naming
    #else
    pthread_setname_np(pthread_self(), ("nexus_" + std::to_string(threadId)).c_str());
    #endif
    
    while (!th->exit.load()) {
        // Wait for work
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { 
            return th->searching.load() || th->exit.load(); 
        });
        
        if (th->exit.load()) break;
        
        lock.unlock();
        
        // Do search work
        if (th->searching.load()) {
            search_loop(th);
        }
    }
}

void ThreadPool::search_loop(ThreadState* th) {
    // Lazy SMP: Each thread searches the same position
    // with slight variations to explore different parts of the tree
    
    th->nodes.store(0);
    threadsSearching.fetch_add(1);
    
    // Main thread (thread 0) does the main search
    // Helper threads help by searching deeper or different move orders
    if (th->threadId == 0) {
        // Main search
        th->search.start(th->board, th->limits, false);
    } else {
        // Helper thread: Search with slight modifications
        // This helps fill the TT with more entries
        
        // Add some randomness to depth for diversification
        int depthOffset = (th->threadId % 3) - 1;  // -1, 0, 1
        
        SearchLimits helperLimits = th->limits;
        helperLimits.depth = std::max(1, helperLimits.depth + depthOffset);
        
        // Helper threads search until main thread finishes
        th->search.start(th->board, helperLimits, false);
    }
    
    th->searching.store(false);
    threadsSearching.fetch_sub(1);
}

void ThreadPool::start_search(BoardState& pos, const SearchLimits& limits) {
    if (threadData.empty()) {
        init(1);
    }
    
    searching.store(true);
    stop.store(false);
    
    // Setup all threads
    for (size_t i = 0; i < threadData.size(); ++i) {
        ThreadState* th = threadData[i];
        th->board = pos;  // Copy position
        th->limits = limits;
        th->nodes.store(0);
        th->searching.store(true);
    }
    
    // Wake all threads
    cv.notify_all();
}

void ThreadPool::stop_search() {
    stop.store(true);
    searching.store(false);
    
    // Stop all individual searches
    for (auto& th : threadData) {
        th->search.stop();
        th->searching.store(false);
    }
    
    // Wait for all threads to finish
    while (threadsSearching.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

Move ThreadPool::best_move() const {
    if (threadData.empty()) return Move::none();
    
    // Return best move from main thread (thread 0)
    return threadData[0]->search.best_move();
}

Value ThreadPool::best_score() const {
    if (threadData.empty()) return VALUE_NONE;
    
    return threadData[0]->search.best_score();
}

u64 ThreadPool::total_nodes() const {
    u64 total = 0;
    for (const auto& th : threadData) {
        total += th->nodes.load();
    }
    // Also add nodes from main search
    if (!threadData.empty()) {
        total += threadData[0]->search.nodes_searched();
    }
    return total;
}

} // namespace Nexus
