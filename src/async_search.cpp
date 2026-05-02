#include "async_search.h"
#include "eval.h"
#include "movegen.h"
#include "board.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace AsyncSearch {

// ============== CPUAsyncEvaluator ==============

CPUAsyncEvaluator::CPUAsyncEvaluator(int num_workers, size_t max_queue)
    : numWorkers_(num_workers)
    , maxQueue_(max_queue)
{
    // Start worker threads
    for (int i = 0; i < numWorkers_; ++i) {
        workers_.emplace_back(&CPUAsyncEvaluator::worker_loop, this, i);
    }
}

CPUAsyncEvaluator::~CPUAsyncEvaluator() {
    running_ = false;
    queueCV_.notify_all();
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

std::future<Value> CPUAsyncEvaluator::submit_eval(
    const BoardState& pos, int thread_id, bool apply_vloss) {
    
    auto request = std::make_shared<EvalRequest>();
    request->position_key = pos.get_key();
    request->fen = pos.get_fen();
    request->depth = 0;
    request->thread_id = thread_id;
    request->virtual_loss_applied = apply_vloss;
    request->submitted = std::chrono::steady_clock::now();
    
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        
        // Check queue depth
        if (requestQueue_.size() >= maxQueue_) {
            // Queue full, evaluate synchronously
            lock.unlock();
            Value score = evaluate_sync(pos);
            
            std::promise<Value> p;
            p.set_value(score);
            return p.get_future();
        }
        
        requestQueue_.push(request);
    }
    
    queueCV_.notify_one();
    return request->result.get_future();
}

void CPUAsyncEvaluator::worker_loop(int worker_id) {
    while (running_) {
        std::shared_ptr<EvalRequest> request;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCV_.wait(lock, [this] {
                return !requestQueue_.empty() || !running_;
            });
            
            if (!running_) break;
            
            if (requestQueue_.empty()) continue;
            
            request = requestQueue_.front();
            requestQueue_.pop();
        }
        
        // Evaluate position
        auto start = std::chrono::steady_clock::now();
        
        // Check cache first
        {
            std::shared_lock<std::shared_mutex> lock(cacheMutex_);
            auto it = evalCache_.find(request->position_key);
            if (it != evalCache_.end()) {
                // Cache hit
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - it->second.timestamp).count();
                
                if (age < 30000) {  // 30 second cache TTL
                    request->result.set_value(it->second.score);
                    evalCount_++;
                    continue;
                }
            }
        }
        
        // Actual evaluation
        BoardState pos;
        pos.set_fen(request->fen);
        Value score = evaluate_sync(pos);
        
        // Update cache
        {
            std::unique_lock<std::shared_mutex> lock(cacheMutex_);
            if (evalCache_.size() < MAX_CACHE_SIZE) {
                evalCache_[request->position_key] = {score, std::chrono::steady_clock::now()};
            } else {
                // Simple eviction: remove random entry
                auto it = evalCache_.begin();
                std::advance(it, request->position_key % evalCache_.size());
                evalCache_.erase(it);
                evalCache_[request->position_key] = {score, std::chrono::steady_clock::now()};
            }
        }
        
        request->result.set_value(score);
        
        // Track latency
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        totalLatencyUs_ += elapsed;
        evalCount_++;
    }
}

Value CPUAsyncEvaluator::evaluate_sync(const BoardState& pos) {
    // Use existing classical evaluator
    return Eval::evaluate(pos);
}

void CPUAsyncEvaluator::process_batch(size_t max_batch) {
    // Workers process continuously, this is a no-op for CPU implementation
    // For GPU implementation, would batch requests and evaluate together
}

bool CPUAsyncEvaluator::can_accept() const {
    std::shared_lock<std::shared_mutex> lock(cacheMutex_);
    return requestQueue_.size() < maxQueue_;
}

size_t CPUAsyncEvaluator::queue_depth() const {
    std::shared_lock<std::shared_mutex> lock(cacheMutex_);
    return requestQueue_.size();
}

double CPUAsyncEvaluator::avg_latency_ms() const {
    uint64_t count = evalCount_.load();
    if (count == 0) return 0.0;
    
    return (totalLatencyUs_.load() / count) / 1000.0;
}

// ============== VirtualLossManager ==============

void VirtualLossManager::apply_loss(Key position_key, const VirtualLoss& loss) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& entries = losses_[position_key];
    
    LossEntry entry;
    entry.loss = loss;
    entry.applied = std::chrono::steady_clock::now();
    entries.push_back(entry);
}

void VirtualLossManager::revert_loss(Key position_key, const VirtualLoss& loss) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = losses_.find(position_key);
    if (it != losses_.end()) {
        it->second.pop_back();  // Remove last applied loss
        if (it->second.empty()) {
            losses_.erase(it);
        }
    }
}

bool VirtualLossManager::has_loss(Key position_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return losses_.find(position_key) != losses_.end();
}

int VirtualLossManager::total_penalty(Key position_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = losses_.find(position_key);
    if (it == losses_.end()) return 0;
    
    int total = 0;
    for (const auto& entry : it->second) {
        total += entry.loss.penalty * entry.loss.visit_count;
    }
    return total;
}

void VirtualLossManager::cleanup(size_t max_age_ms) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = losses_.begin(); it != losses_.end(); ) {
        auto& entries = it->second;
        
        entries.erase(
            std::remove_if(entries.begin(), entries.end(),
                [now, max_age_ms](const LossEntry& e) {
                    auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - e.applied).count();
                    return age > max_age_ms;
                }),
            entries.end()
        );
        
        if (entries.empty()) {
            it = losses_.erase(it);
        } else {
            ++it;
        }
    }
}

// ============== AsyncNode ==============

double AsyncNode::uct_score(double parent_visits, double c_puct) const {
    if (visit_count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    
    double exploitation = q_value();
    double exploration = c_puct * policy_prior * 
                        std::sqrt(parent_visits) / (1 + visit_count);
    
    return exploitation + exploration;
}

// ============== AsyncSearchManager ==============

AsyncSearchManager::AsyncSearchManager(std::unique_ptr<AsyncEvaluator> evaluator)
    : evaluator_(std::move(evaluator))
{
}

AsyncSearchManager::~AsyncSearchManager() {
    stop_ = true;
    
    for (auto& thread : search_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

Value AsyncSearchManager::search(BoardState& root_pos, Depth max_depth,
                                  int max_nodes, int time_ms) {
    // Initialize root node
    root_ = std::make_unique<AsyncNode>();
    root_->position = root_pos;
    root_->key = root_pos.get_key();
    root_->depth = max_depth;
    root_->alpha = -VALUE_INFINITE;
    root_->beta = VALUE_INFINITE;
    
    // Reset state
    stop_ = false;
    nodes_searched_ = 0;
    stats_ = {};
    
    // Start search threads
    search_threads_.clear();
    for (int i = 0; i < numThreads_; ++i) {
        search_threads_.emplace_back(&AsyncSearchManager::search_thread_loop, this, i);
    }
    
    // Wait for completion
    for (auto& thread : search_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    return root_->best_value;
}

void AsyncSearchManager::search_thread_loop(int thread_id) {
    while (!should_stop()) {
        AsyncNode* node = select_node(root_.get());
        
        if (!node) {
            // No more nodes to explore
            break;
        }
        
        expand_and_evaluate(node, thread_id);
        nodes_searched_++;
        stats_.nodes_searched++;
    }
}

AsyncNode* AsyncSearchManager::select_node(AsyncNode* root) {
    // UCT selection with virtual loss consideration
    AsyncNode* current = root;
    
    while (current->state == AsyncNode::EXPANDED && !current->children.empty()) {
        AsyncNode* best_child = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        
        double parent_visits = static_cast<double>(current->visit_count);
        
        for (auto& child : current->children) {
            // Apply virtual loss penalty if node being evaluated
            if (useVirtualLoss_ && vloss_manager_.has_loss(child.key)) {
                continue;  // Skip nodes with virtual loss
            }
            
            double score = child.uct_score(parent_visits);
            if (score > best_score) {
                best_score = score;
                best_child = &child;
            }
        }
        
        if (!best_child) break;
        current = best_child;
    }
    
    return current;
}

void AsyncSearchManager::expand_and_evaluate(AsyncNode* node, int thread_id) {
    if (node->state == AsyncNode::UNEVALUATED) {
        // Submit async evaluation
        node->state = AsyncNode::PENDING;
        node->eval_future = evaluator_->submit_eval(node->position, thread_id, useVirtualLoss_);
        
        if (useVirtualLoss_) {
            VirtualLoss vloss;
            vloss.visit_count = 1;
            vloss_manager_.apply_loss(node->key, vloss);
            node->virtual_loss_active = true;
            stats_.virtual_losses_applied++;
        }
        
        stats_.async_evals_submitted++;
        
        // Don't wait - continue to other nodes
        return;
    }
    
    if (node->state == AsyncNode::PENDING) {
        // Check if evaluation is ready
        if (node->eval_future.wait_for(std::chrono::milliseconds(0)) == 
            std::future_status::ready) {
            
            // Get result
            node->static_eval = node->eval_future.get();
            node->state = AsyncNode::EVALUATED;
            
            // Revert virtual loss
            if (node->virtual_loss_active) {
                VirtualLoss vloss;
                vloss.visit_count = 1;
                vloss_manager_.revert_loss(node->key, vloss);
                node->virtual_loss_active = false;
            }
            
            stats_.async_evals_completed++;
            
            // Expand children (generate moves)
            node->children_expanded = true;
            node->state = AsyncNode::EXPANDED;
            
            // Generate moves
            ExtMove moves[MAX_MOVES];
            ExtMove* end = generate_moves(node->position, moves, false);
            
            for (ExtMove* it = moves; it != end; ++it) {
                AsyncNode child;
                child.position = node->position;
                StateInfo st;
                child.position.do_move(it->move, st);
                child.key = child.position.get_key();
                child.depth = node->depth - 1;
                child.alpha = node->alpha;
                child.beta = node->beta;
                // Default policy prior (would come from policy network)
                child.policy_prior = 1.0 / std::distance(moves, end);
                
                node->children.push_back(std::move(child));
            }
            
            // Backpropagate
            backpropagate(node, node->static_eval);
        }
    }
}

void AsyncSearchManager::backpropagate(AsyncNode* node, Value score) {
    while (node) {
        node->visit_count++;
        node->total_score += score;  // Convert Value to double
        
        // Update best
        if (node->best_value == VALUE_NONE || score > node->best_value) {
            node->best_value = score;
        }
        
        node = node->parent;  // Would need parent pointer
        // Simplified: just update current node
    }
}

bool AsyncSearchManager::should_stop() const {
    return stop_.load();
}

AsyncSearchManager::Stats AsyncSearchManager::get_stats() const {
    return stats_;
}

void AsyncSearchManager::reset_stats() {
    stats_ = {};
}

// ============== HybridSearch ==============

Value HybridSearch::negamax_async(
    BoardState& pos, Depth depth, Value alpha, Value beta,
    AsyncEvaluator& evaluator, int thread_id) {
    
    // Check if evaluator is saturated
    if (evaluator.queue_depth() > QUEUE_SATURATION_THRESHOLD) {
        // Use quick fallback to avoid blocking
        return quick_eval_fallback(pos);
    }
    
    // Submit async evaluation
    auto eval_future = evaluator.submit_eval(pos, thread_id, true);
    
    // While waiting, do a shallow search or generate moves
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate_moves(pos, moves, false);
    
    if (depth <= 1 || moves == end) {
        // Leaf node, wait for eval
        return eval_future.get();
    }
    
    // Do limited search with virtual loss applied
    Value best = -VALUE_INFINITE;
    
    for (ExtMove* it = moves; it != end; ++it) {
        StateInfo st;
        pos.do_move(it->move, st);
        
        Value value = -negamax_async(pos, depth - 1, -beta, -alpha, evaluator, thread_id);
        
        pos.undo_move(it->move, st);
        
        if (value > best) {
            best = value;
        }
        
        if (best >= beta) {
            break;  // Beta cutoff
        }
    }
    
    // Get actual eval and blend with search result
    Value actual_eval = eval_future.get();
    
    // Blend: prefer search result at higher depths
    double search_weight = std::min(depth / 10.0, 1.0);
    return Value(search_weight * best + (1.0 - search_weight) * actual_eval);
}

Value HybridSearch::quick_eval_fallback(const BoardState& pos) {
    // Fast classical evaluation when evaluator is saturated
    return Eval::evaluate(pos);
}

} // namespace AsyncSearch
