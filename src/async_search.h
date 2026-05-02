#pragma once

#include "types.h"
#include "board.h"
#include "move.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace AsyncSearch {

// ======== Virtual Loss ========
// Used when CPU sends a batch to GPU but continues exploring other nodes
// The node gets a temporary penalty so other threads don't also explore it

struct VirtualLoss {
    static constexpr int DEFAULT_PENALTY = 300;  // cp penalty
    
    int penalty;        // Virtual loss value in centipawns
    int visit_count;    // How many times this node is being evaluated
    
    void apply(Value& score) const {
        // Apply virtual loss: penalize score to discourage other threads
        score -= Value(penalty * visit_count);
    }
    
    void revert(Value& score) const {
        // Revert when real evaluation returns
        score += Value(penalty * visit_count);
    }
};

// ======== Evaluation Request ========
// Queued request sent to NNUE evaluator (GPU or CPU thread pool)

struct EvalRequest {
    Key position_key;           // Zobrist hash for dedup
    std::string fen;            // Position FEN
    int depth;                  // Search depth this eval is for
    int thread_id;              // Which thread requested
    
    // Promise for async result
    std::promise<Value> result;
    std::chrono::steady_clock::time_point submitted;
    
    // Virtual loss applied
    bool virtual_loss_applied;
    
    EvalRequest() : depth(0), thread_id(-1), virtual_loss_applied(false) {}
};

// ======== Evaluation Response ========

struct EvalResponse {
    Key position_key;
    Value score;
    bool from_cache;            // Was this a TT/nnue-cache hit?
    int latency_us;             // Microseconds for evaluation
};

// ======== Async Evaluator Interface ========
// Abstract interface for batch/async evaluation

class AsyncEvaluator {
public:
    virtual ~AsyncEvaluator() = default;
    
    // Submit evaluation request (non-blocking)
    virtual std::future<Value> submit_eval(const BoardState& pos, 
                                           int thread_id = 0,
                                           bool apply_vloss = true) = 0;
    
    // Process pending requests (call from evaluator thread)
    virtual void process_batch(size_t max_batch_size = 256) = 0;
    
    // Check if evaluator is ready for more requests
    virtual bool can_accept() const = 0;
    
    // Get queue statistics
    virtual size_t queue_depth() const = 0;
    virtual double avg_latency_ms() const = 0;
};

// ======== CPU Thread Pool Evaluator ========
// Uses multiple CPU threads for batch evaluation when no GPU available

class CPUAsyncEvaluator : public AsyncEvaluator {
public:
    CPUAsyncEvaluator(int num_workers = 4, 
                      size_t max_queue = 1024);
    ~CPUAsyncEvaluator();
    
    std::future<Value> submit_eval(const BoardState& pos,
                                   int thread_id = 0,
                                   bool apply_vloss = true) override;
    void process_batch(size_t max_batch_size = 256) override;
    bool can_accept() const override;
    size_t queue_depth() const override;
    double avg_latency_ms() const override;
    
private:
    int numWorkers_;
    size_t maxQueue_;
    
    std::queue<std::shared_ptr<EvalRequest>> requestQueue_;
    mutable std::mutex queueMutex_;
    std::condition_variable queueCV_;
    
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{true};
    
    // Latency tracking
    std::atomic<uint64_t> totalLatencyUs_{0};
    std::atomic<uint64_t> evalCount_{0};
    
    // Cache for repeated positions
    struct CacheEntry {
        Value score;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::unordered_map<Key, CacheEntry> evalCache_;
    mutable std::shared_mutex cacheMutex_;
    static constexpr size_t MAX_CACHE_SIZE = 65536;
    
    void worker_loop(int worker_id);
    Value evaluate_sync(const BoardState& pos);
};

// ======== Virtual Loss Manager ========
// Tracks virtual losses applied to nodes awaiting evaluation

class VirtualLossManager {
public:
    void apply_loss(Key position_key, const VirtualLoss& loss);
    void revert_loss(Key position_key, const VirtualLoss& loss);
    bool has_loss(Key position_key) const;
    
    // Get total virtual loss for a position
    int total_penalty(Key position_key) const;
    
    // Clear old entries
    void cleanup(size_t max_age_ms = 30000);
    
private:
    struct LossEntry {
        VirtualLoss loss;
        std::chrono::steady_clock::time_point applied;
    };
    
    mutable std::shared_mutex mutex_;
    std::unordered_map<Key, std::vector<LossEntry>> losses_;
};

// ======== Async Search Node ========
// Node in the search tree that supports async evaluation

struct AsyncNode {
    BoardState position;
    Key key;
    
    // Search bounds
    Value alpha;
    Value beta;
    Depth depth;
    
    // Evaluation state
    enum EvalState {
        UNEVALUATED,    // No evaluation started
        PENDING,        // Submitted to evaluator, waiting
        EVALUATED,      // Evaluation complete
        EXPANDED,       // Children generated
    };
    
    EvalState state = UNEVALUATED;
    Value static_eval = VALUE_NONE;
    std::future<Value> eval_future;  // For async retrieval
    
    // Children (lazy expansion)
    std::vector<AsyncNode> children;
    bool children_expanded = false;
    
    // Virtual loss
    bool virtual_loss_active = false;
    
    // Best child info
    Move best_move = Move::none();
    Value best_value = VALUE_NONE;
    
    // Thread safety
    mutable std::mutex node_mutex;
    
    // For MCTS-style async search
    double policy_prior = 0.0;      // From policy network
    int visit_count = 0;
    double total_score = 0.0;       // For averaging
    
    double q_value() const {
        return visit_count > 0 ? total_score / visit_count : 0.0;
    }
    
    double uct_score(double parent_visits, double c_puct = 1.0) const;
};

// ======== Async Search Manager ========
// Coordinates CPU search threads with async evaluator

class AsyncSearchManager {
public:
    AsyncSearchManager(std::unique_ptr<AsyncEvaluator> evaluator);
    ~AsyncSearchManager();
    
    // Main search entry: performs async tree search
    Value search(BoardState& root, Depth max_depth, 
                 int max_nodes = 0, int time_ms = 0);
    
    // Set number of CPU search threads
    void set_num_threads(int n) { numThreads_ = n; }
    
    // Enable/disable virtual loss
    void set_virtual_loss(bool enabled) { useVirtualLoss_ = enabled; }
    
    // Statistics
    struct Stats {
        int nodes_searched;
        int async_evals_submitted;
        int async_evals_completed;
        int cache_hits;
        int virtual_losses_applied;
        double avg_eval_latency_ms;
        double nps;  // Nodes per second
    };
    
    Stats get_stats() const;
    void reset_stats();
    
private:
    std::unique_ptr<AsyncEvaluator> evaluator_;
    VirtualLossManager vloss_manager_;
    
    int numThreads_ = 4;
    bool useVirtualLoss_ = true;
    
    // Root node
    std::unique_ptr<AsyncNode> root_;
    
    // Search control
    std::atomic<bool> stop_{false};
    std::atomic<int> nodes_searched_{0};
    
    // Statistics
    Stats stats_{};
    
    // Worker threads
    std::vector<std::thread> search_threads_;
    
    // Main search loop for each thread
    void search_thread_loop(int thread_id);
    
    // Select node using UCT / virtual loss
    AsyncNode* select_node(AsyncNode* root);
    
    // Expand node and submit eval
    void expand_and_evaluate(AsyncNode* node, int thread_id);
    
    // Backpropagate results
    void backpropagate(AsyncNode* node, Value score);
    
    // Check time/node limits
    bool should_stop() const;
};

// ======== Hybrid Search Interface ========
// Combines traditional alpha-beta with async neural eval

class HybridSearch {
public:
    // Traditional search but with async NNUE evaluation
    // When evaluator is busy, uses virtual loss and continues exploring
    Value negamax_async(BoardState& pos, Depth depth, 
                        Value alpha, Value beta,
                        AsyncEvaluator& evaluator,
                        int thread_id = 0);
    
    // Quick synchronous fallback when evaluator is saturated
    Value quick_eval_fallback(const BoardState& pos);
    
private:
    VirtualLossManager vloss_;
    
    // Threshold: if evaluator queue > this, use fallback
    static constexpr size_t QUEUE_SATURATION_THRESHOLD = 512;
};

} // namespace AsyncSearch
