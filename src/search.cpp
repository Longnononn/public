#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "tt.h"
#include "timeman.h"
#include "see.h"
#include "history.h"
#include "reductions.h"
#include "simd.h"
#include "nnue.h"
#include "syzygy.h"
#include "thread.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

// Initialize history manager
static Nexus::HistoryManager History;

// Use NNUE if available
static bool useNNUE = false;
static bool useSyzygy = false;

namespace Nexus {

Search Searcher;

// Evaluation wrapper - NNUE with classical fallback
static inline Value evaluate_position(const BoardState& pos) {
    if (useNNUE && NNUE::g_nnue.is_loaded()) {
        Value nnueScore = NNUE::g_nnue.evaluate(pos);
        if (nnueScore != VALUE_NONE) {
            return nnueScore;
        }
    }
    // Fallback to classical evaluation
    return Eval::evaluate(pos);
}

// TT score adjustment for mate distance
Value score_to_tt(Value s, int ply) {
    return s >= VALUE_MATE_IN_MAX_PLY  ? Value(int(s) + ply)
         : s <= VALUE_MATED_IN_MAX_PLY ? Value(int(s) - ply) : s;
}

Value score_from_tt(int s, int ply) {
    return s == VALUE_NONE             ? VALUE_NONE
         : s >= VALUE_MATE_IN_MAX_PLY  ? Value(s - ply)
         : s <= VALUE_MATED_IN_MAX_PLY ? Value(s + ply) : Value(s);
}

// Mate distance pruning
Value mate_distance(int ply) {
    return Value(int(VALUE_MATE) - ply);
}

Search::Search() : running(false), stopOnPonderhit(false), nodesSearched(0), 
                   selDepth(0), completedDepth(Depth(0)), previousScore(VALUE_NONE), multiPV(1) {
    clear();
}

void Search::clear() {
    std::memset(mainHistory, 0, sizeof(mainHistory));
    std::memset(captureHistory, 0, sizeof(captureHistory));
    std::memset(contHistoryBuf, 0, sizeof(contHistoryBuf));
    
    // Set up continuation history pointers
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            continuationHistory[i][j] = &contHistoryBuf[i][j][0];
    
    // Clear new history system
    History.clear();
    
    // Initialize reduction tables
    Reductions::init();
    
    // Initialize NNUE if available
    if (NNUE::g_nnue.is_loaded()) {
        useNNUE = true;
    }
    
    // Check Syzygy availability
    useSyzygy = Syzygy::TB.is_initialized();
    
    TT.new_search();
}

void Search::start(BoardState& pos, const SearchLimits& limits, bool ponder) {
    running = true;
    stopOnPonderhit = false;
    nodesSearched = 0;
    completedDepth = Depth(0);
    selDepth = 0;
    
    // Reset NNUE accumulator for root position
    if (useNNUE) {
        NNUE::g_nnue.reset_accumulator(pos);
    }
    
    // Check Syzygy at root
    if (useSyzygy && Syzygy::TB.is_tb_position(pos)) {
        Move tbMove = Move::none();
        Value tbScore = VALUE_NONE;
        
        if (Syzygy::TB.root_probe(pos, tbMove, tbScore)) {
            // Use tablebase move directly
            rootMoves.clear();
            rootMoves.emplace_back(tbMove);
            rootMoves[0].score = tbScore;
            rootMoves[0].pv[0] = tbMove;
            
            std::cout << "info string Tablebase hit!" << std::endl;
            return;
        }
    }
    
    // Generate root moves
    ExtMove moves[MAX_MOVES];
    ExtMove* end;
    if (pos.is_check())
        end = generate<EVASIONS>(pos, moves);
    else
        end = generate<NON_EVASIONS>(pos, moves);
    
    rootMoves.clear();
    for (ExtMove* it = moves; it != end; ++it) {
        if (is_legal(pos, it->move)) {
            if (limits.searchmoves.empty() || 
                std::find(limits.searchmoves.begin(), limits.searchmoves.end(), it->move) != limits.searchmoves.end())
                rootMoves.emplace_back(it->move);
        }
    }
    
    if (rootMoves.empty()) {
        running = false;
        return;
    }
    
    iterative_deepening(pos);
    running = false;
}

void Search::stop() {
    running = false;
}

void Search::iterative_deepening(BoardState& pos) {
    SearchStack ss[MAX_PLY + 12];
    Move pv[MAX_PLY + 1] = {};
    
    for (int i = 0; i < MAX_PLY + 12; ++i) {
        ss[i].pv = (i < MAX_PLY) ? &pv[i] : nullptr;
        ss[i].ply = i;
        ss[i].currentMove = Move::none();
        ss[i].excludedMove = Move::none();
        ss[i].doubleExtensions = 0;
        ss[i].cutOffCnt = 0;
        ss[i].continuationHistory = continuationHistory[false][false];
    }
    
    // Aspiration window initial
    Value prevScore = VALUE_ZERO;
    
    for (int idepth = 1; idepth <= 64; ++idepth) {
        Depth depth = Depth(idepth);
        if (!running) break;
        
        selDepth = 0;
        
        Value score = aspiration_window(pos, ss, prevScore, depth);
        
        if (!running) break;
        
        prevScore = score;
        previousScore = score;
        completedDepth = depth;
        
        // Sort root moves by score
        std::stable_sort(rootMoves.begin(), rootMoves.end());
        
        // Print info
        if (!rootMoves.empty()) {
            auto& rm = rootMoves[0];
            std::cout << "info depth " << depth 
                      << " seldepth " << selDepth
                      << " score " << (std::abs((int)rm.score) >= VALUE_MATE_IN_MAX_PLY ? 
                          "mate " + std::to_string((int)(VALUE_MATE - Value(std::abs((int)rm.score))) / 2 * (rm.score > 0 ? 1 : -1)) :
                          "cp " + std::to_string((int)rm.score))
                      << " nodes " << nodesSearched
                      << " pv";
            for (int i = 0; i < rm.pvSize; ++i)
                std::cout << " " << rm.pv[i].to_uci();
            std::cout << std::endl;
        }
        
        if (Time.can_stop_on_fail_low() && depth >= 8)
            break;
        
        if (Time.should_stop())
            break;
    }
    
    // Output best move
    if (!rootMoves.empty()) {
        std::cout << "bestmove " << rootMoves[0].pv[0].to_uci() << std::endl;
    }
}

Value Search::aspiration_window(BoardState& pos, SearchStack* ss, Value prevScore, Depth depth) {
    Value delta = SearchParams::AspirationDelta;
    Value alpha = std::max(prevScore - delta, -VALUE_INFINITE);
    Value beta  = std::min(prevScore + delta, VALUE_INFINITE);
    
    // Wider windows for first iterations or after mate threat
    if (depth <= 4 || std::abs(prevScore) >= VALUE_KNOWN_WIN) {
        alpha = -VALUE_INFINITE;
        beta = VALUE_INFINITE;
        delta = VALUE_INFINITE / 2;
    }
    
    // Track fail-low/fail-high count for dynamic adjustment
    int failCount = 0;
    
    while (true) {
        Value score = negamax<true>(pos, ss, alpha, beta, depth, false);
        
        if (!running)
            return VALUE_NONE;
        
        if (score <= alpha) {
            // Failed low - widen window aggressively
            beta = (alpha + beta) / 2;
            alpha = std::max(score - delta, -VALUE_INFINITE);
            if (alpha < -VALUE_MATE_IN_MAX_PLY) alpha = -VALUE_INFINITE;
            
            // Increase window faster after multiple fails
            delta = Value(static_cast<int>(int(delta) * SearchParams::AspirationGrowth) + (failCount++ * 5));
        }
        else if (score >= beta) {
            // Failed high - widen window
            beta = std::min(score + delta, VALUE_INFINITE);
            if (beta > VALUE_MATE_IN_MAX_PLY) beta = VALUE_INFINITE;
            
            // Increase window faster after multiple fails
            delta = Value(static_cast<int>(int(delta) * SearchParams::AspirationGrowth) + (failCount++ * 5));
        }
        else {
            // Within window - success
            return score;
        }
        
        if (Time.should_stop())
            return score;
    }
}

template<bool PvNode>
Value Search::negamax(BoardState& pos, SearchStack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {
    if (!running) return VALUE_NONE;
    
    nodesSearched++;
    
    const bool rootNode = (ss->ply == 0);
    const bool inCheck = pos.is_check();
    
    ss->currentMove = Move::none();
    ss->doubleExtensions = (ss->ply > 0) ? (ss-1)->doubleExtensions : 0;
    
    // Check limits
    if (nodesSearched % 1024 == 0 && Time.should_stop())
        return VALUE_NONE;
    
    // Qsearch at depth <= 0 (unless in check)
    if (depth <= 0 && !inCheck)
        return qsearch<PvNode>(pos, ss, alpha, beta);
    
    if (depth <= 0)
        depth = Depth(0);
    
    // Mate distance pruning
    alpha = std::max(alpha, (Value)(-VALUE_MATE + ss->ply));
    beta = std::min(beta, (Value)(VALUE_MATE - ss->ply - 1));
    if (alpha >= beta)
        return alpha;
    
    // Draw detection
    if (!rootNode && (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY))
        return VALUE_DRAW;
    
    selDepth = std::max(selDepth, ss->ply);
    
    // Prefetch TT entry before probe (async, helps with latency)
    if (ss->excludedMove == Move::none())
        TT.prefetch(pos.st.key);
    
    // TT probe
    bool ttHit = false;
    TTEntry* tte = nullptr;
    if (ss->excludedMove == Move::none()) {
        bool found = false;
        tte = TT.probe(pos.st.key, found);
        ttHit = found && tte->key16 != 0;
    }
    
    Move ttMove = (ttHit && Move(tte->move).is_ok()) ? Move(tte->move) : Move::none();
    Value ttValue = ttHit ? score_from_tt(tte->value, ss->ply) : VALUE_NONE;
    int ttDepth = ttHit ? tte->depth : 0;
    Bound ttBound = ttHit ? (Bound)tte->bound : BOUND_NONE;
    
    // TT cutoff
    if (!PvNode && !inCheck && ttHit && tte->depth >= depth
        && ttValue != VALUE_NONE
        && ((ttBound == BOUND_UPPER && ttValue <= alpha) ||
            (ttBound == BOUND_LOWER && ttValue >= beta) ||
            (ttBound == BOUND_EXACT))) {
        if (ttValue >= beta && ttMove.is_ok() && !ttMove.is_promotion())
            update_quiet_stats(pos, ss, ttMove, nullptr, 0, stat_bonus(depth));
        return ttValue;
    }
    
    // Static evaluation
    Value eval;
    if (inCheck) {
        eval = VALUE_NONE;
        ss->staticEval = VALUE_NONE;
    } else if (ttHit) {
        eval = tte->eval != VALUE_NONE ? Value(tte->eval) : evaluate_position(pos);
        ss->staticEval = eval;
        if (ttValue != VALUE_NONE && 
            ((ttBound == BOUND_UPPER && ttValue < eval) ||
             (ttBound == BOUND_LOWER && ttValue > eval)))
            eval = ttValue;
    } else {
        eval = ss->staticEval = evaluate_position(pos);
        // Store static eval in TT
        if (ss->excludedMove == Move::none())
            TT.store(pos.st.key, VALUE_NONE, false, BOUND_NONE, Depth(0), Move::none(), eval);
    }
    
    // Improving flag
    bool improving = !inCheck && ss->ply >= 2 && ss->staticEval > (ss-2)->staticEval;
    
    // Step 8: Futility pruning
    if (!PvNode && !inCheck && ss->excludedMove == Move::none() && eval < VALUE_KNOWN_WIN) {
        // Reverse futility pruning (static null move)
        if (depth <= 8 && eval - Value(168 * depth) >= beta && eval < VALUE_TB_WIN_IN_MAX_PLY && !rootNode)
            return eval;
        
        // Null move pruning
        if (!rootNode && (ss-1)->currentMove != Move::null() && eval >= beta
            && ss->staticEval >= beta - Value(30 * depth - 170 * improving - 86 * !pos.pieces(~pos.sideToMove))
            && !pos.is_material_draw() && pos.pieces(~pos.sideToMove) && (ss->ply >= pos.st.pliesFromNull || PvNode)
            && depth >= 3) {
            
            int R = 4 + depth / 3 + std::min((int)((eval - beta) / 200), 3);
            
            StateInfo nullSt;
            pos.do_null_move(nullSt);
            ss->currentMove = Move::null();
            (ss+1)->continuationHistory = continuationHistory[false][false];
            
            Value nullValue = -negamax<false>(pos, ss+1, -beta, -beta+1, Depth(int(depth) - R), !cutNode);
            
            pos.undo_null_move(nullSt);
            
            if (nullValue >= beta) {
                if (nullValue >= VALUE_MATE_IN_MAX_PLY)
                    nullValue = beta;
                if (std::abs((int)beta) < VALUE_KNOWN_WIN && depth < 14)
                    return nullValue;
            }
        }
        
        // Razoring
        if (eval + Value(256 * depth) <= alpha && eval < VALUE_KNOWN_WIN && !pos.is_material_draw()) {
            if (depth <= 3) {
                Value qval = qsearch<false>(pos, ss, alpha, Value(int(alpha) + 1));
                if (qval <= alpha)
                    return qval;
            }
        }
    }
    
    // Step 9: ProbCut pruning - beta cutoff prediction at reduced depth
    // If we have a good capture that causes a cutoff at reduced depth, likely to at full depth
    if (!PvNode && !inCheck && depth >= PruningThresholds::ProbCutDepth 
        && std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY && ss->excludedMove == Move::none()) {
        
        Value probCutBeta = beta + Value(PruningThresholds::ProbCutMargin);
        Depth probCutDepth = Depth(int(depth) - 3);
        
        // Try captures first with SEE >= probCutMargin
        ExtMove captureList[MAX_MOVES];
        ExtMove* capEnd = generate<CAPTURES>(pos, captureList);
        
        int capturesTried = 0;
        for (ExtMove* it = captureList; it != capEnd && capturesTried < 3; ++it) {
            if (!is_legal(pos, it->move))
                continue;
            
            // Only try captures with positive SEE
            if (SEE::see_value(pos, it->move) < 0)
                continue;
            
            capturesTried++;
            
            StateInfo st;
            pos.do_move(it->move, st);
            Value value = -negamax<false>(pos, ss+1, -probCutBeta, -probCutBeta + 1, probCutDepth, !cutNode);
            pos.undo_move(it->move, st);
            
            if (value >= probCutBeta)
                return value;
        }
    }
    
    // Step 10: Internal iterative reductions
    if (depth >= 4 && PvNode && ttMove == Move::none())
        depth = Depth(int(depth) - 1);
    
    // Move generation and sorting
    ExtMove moveList[MAX_MOVES];
    ExtMove* end;
    
    if (inCheck)
        end = generate<EVASIONS>(pos, moveList);
    else
        end = generate<NON_EVASIONS>(pos, moveList);
    
    int moveCount = end - moveList;
    
    // Score moves with SEE and improved history
    for (ExtMove* it = moveList; it != end; ++it) {
        if (it->move == ttMove)
            it->value = 2000000000;
        else if (pos.is_capture(it->move)) {
            // Use SEE for accurate capture scoring
            int seeVal = SEE::see_value(pos, it->move);
            it->value = 1000000 + seeVal;
        }
        else if (it->move == Move::none()) {
            it->value = -1000000;
        }
        else {
            // Combined history score
            int his = mainHistory[pos.sideToMove][it->move.from_to()];
            // Add continuation history contribution
            if (ss->ply > 1 && ss->currentMove.is_ok()) {
                // Would integrate continuation history here
            }
            it->value = his;
        }
    }
    
    // Sort by value descending
    std::sort(moveList, end);
    
    // Step 11: Search moves
    Move bestMove = Move::none();
    Value bestValue = -VALUE_INFINITE;
    Value value = bestValue;
    int movesSearched = 0;
    
    Move quietsSearched[64];
    Move capturesSearched[64];
    int quietCount = 0, captureCount = 0;
    
    for (ExtMove* it = moveList; it != end; ++it) {
        Move m = it->move;
        
        if (m == ss->excludedMove)
            continue;
        
        if (!is_legal(pos, m))
            continue;
        
        // Late move pruning
        if (!rootNode && !inCheck && !pos.is_capture_or_promotion(m) && bestValue > VALUE_MATED_IN_MAX_PLY) {
            int lmpDepth = improving ? 3 + depth * depth : (3 + depth * depth) / 2;
            if (movesSearched >= lmpDepth)
                continue;
        }
        
        movesSearched++;
        
        // SEE pruning
        if (!rootNode && !inCheck && depth <= 8 && !pos.is_material_draw() && !inCheck) {
            if (!pos.is_capture_or_promotion(m) && depth <= 5 && movesSearched >= 4 + 3 * improving)
                continue;
        }
        
        bool givesCheck = pos.gives_check(m);
        bool captureOrPromotion = pos.is_capture_or_promotion(m);
        
        // Singular extension
        int extension = 0;
        if (PvNode && !rootNode && depth >= 7 && m == ttMove && ttHit && std::abs((int)ttValue) < VALUE_TB_WIN_IN_MAX_PLY
            && (ttBound & BOUND_LOWER) && ttDepth >= depth - 3) {
            Value singularBeta = ttValue - Value(2 * depth);
            Depth singularDepth = Depth((int(depth) - 1) / 2);
            ss->excludedMove = m;
            Value singularValue = negamax<false>(pos, ss, singularBeta - Value(1), singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();
            
            if (singularValue < singularBeta) {
                extension = 1;
                if (!PvNode && singularValue < singularBeta - Value(18) && ss->doubleExtensions <= 6) {
                    extension = 2;
                    ss->doubleExtensions++;
                }
            } else if (singularValue > singularBeta && singularBeta >= beta)
                return singularBeta;
            else if (ttValue >= beta)
                extension = -2;
        }
        else if (givesCheck)
            extension = 1;
        
        Depth newDepth = Depth(int(depth) - 1 + extension);
        // Prevent runaway extensions causing exponential search growth
        if (int(newDepth) > int(depth) && int(depth) < 4)
            newDepth = depth;  // Cap at current depth for low depths
        
        // Make move
        StateInfo st;
        pos.do_move(m, st);
        ss->currentMove = m;
        
        // Update NNUE accumulator
        if (useNNUE) {
            NNUE::g_nnue.update(m, pos);
        }
        
        // LMR - Late Move Reduction with advanced tables
        int R = 0;
        if (depth >= 2 && movesSearched > 1 && !captureOrPromotion && !inCheck && !givesCheck && !pos.advanced_pawn_push(m)) {
            // Use advanced reduction tables
            R = Reductions::get_reduction(depth, movesSearched, improving, cutNode);
            
            // PV nodes get less reduction
            if (PvNode) R = std::max(0, R - 1);
            
            // History-based adjustment
            int histScore = mainHistory[pos.sideToMove][m.from_to()];
            R -= Reductions::history_reduction(histScore);
            
            // SEE-based adjustment for captures
            if (pos.is_capture(m)) {
                int seeScore = SEE::see_value(pos, m);
                R -= Reductions::see_reduction(seeScore);
            }
            
            // Clamp to valid range
            R = std::clamp(R, 0, int(newDepth));
        }
        
        // PVS / Full window search
        if (PvNode && movesSearched == 1) {
            value = -negamax<true>(pos, ss+1, -beta, -alpha, newDepth, false);
        } else {
            // Reduced/zero-window search
            if (R > 0) {
                value = -negamax<false>(pos, ss+1, Value(-(int(alpha)+1)), -alpha, Depth(int(newDepth) - R), true);
            } else {
                value = -negamax<false>(pos, ss+1, Value(-(int(alpha)+1)), -alpha, newDepth, !PvNode);
            }
            
            // Re-search if reduced depth or zero-window beat alpha
            if (value > alpha) {
                if (R > 0)
                    value = -negamax<false>(pos, ss+1, Value(-(int(alpha)+1)), -alpha, newDepth, true);
                
                // Full window search for PV nodes
                if (PvNode && value > alpha)
                    value = -negamax<true>(pos, ss+1, -beta, -alpha, newDepth, false);
            }
        }
        
        // Restore NNUE state (simplified - actual restore via stack)
        if (useNNUE) {
            NNUE::g_nnue.undo();
        }
        
        pos.undo_move(m, st);
        
        if (!running)
            return VALUE_NONE;
        
        // Update best
        if (value > bestValue) {
            bestValue = value;
            bestMove = m;
            
            if (rootNode) {
                auto it_rm = std::find_if(rootMoves.begin(), rootMoves.end(),
                    [&m](const RootMove& rm) { return rm.pv[0] == m; });
                if (it_rm != rootMoves.end()) {
                    it_rm->score = value;
                    it_rm->pv[0] = m;
                    it_rm->pvSize = 1;
                    if (ss[1].pv) {
                        for (int i = 0; ss[1].pv[i].is_ok() && it_rm->pvSize < 64; ++i)
                            it_rm->pv[it_rm->pvSize++] = ss[1].pv[i];
                    }
                }
            }
            
            if (value > alpha) {
                alpha = value;
                
                if (PvNode && ss->pv)
                    ss->pv[0] = m;
                
                if (value >= beta) {
                    // Beta cutoff
                    if (!captureOrPromotion) {
                        update_quiet_stats(pos, ss, m, quietsSearched, quietCount, stat_bonus(depth));
                    }
                    TT.store(pos.st.key, score_to_tt(value, ss->ply), PvNode, BOUND_LOWER, depth, m, ss->staticEval);
                    return value;
                }
            }
        }
        
        if (!captureOrPromotion && quietCount < 64)
            quietsSearched[quietCount++] = m;
        else if (captureOrPromotion && captureCount < 64)
            capturesSearched[captureCount++] = m;
    }
    
    // Checkmate / stalemate
    if (movesSearched == 0) {
        if (inCheck)
            return mated_in(ss->ply);
        else
            return VALUE_DRAW;
    }
    
    // Store in TT
    Bound b = (bestValue >= beta) ? BOUND_LOWER : BOUND_UPPER;
    if (PvNode && bestValue > alpha && bestValue < beta)
        b = BOUND_EXACT;
    
    if (ss->excludedMove == Move::none())
        TT.store(pos.st.key, score_to_tt(bestValue, ss->ply), PvNode, b, depth, bestMove, ss->staticEval);
    
    return bestValue;
}

template<bool PvNode>
Value Search::qsearch(BoardState& pos, SearchStack* ss, Value alpha, Value beta) {
    if (!running) return VALUE_NONE;
    
    nodesSearched++;
    
    const bool inCheck = pos.is_check();
    
    // Check limits
    if (nodesSearched % 1024 == 0 && Time.should_stop())
        return VALUE_NONE;
    
    // Draw detection
    if (pos.is_draw(ss->ply) || ss->ply >= MAX_PLY)
        return VALUE_DRAW;
    
    selDepth = std::max(selDepth, ss->ply);
    
    // TT probe
    bool ttHit = false;
    bool found = false;
    TTEntry* tte = TT.probe(pos.st.key, found);
    ttHit = found && tte->key16 != 0;
    
    Value ttValue = ttHit ? score_from_tt(tte->value, ss->ply) : VALUE_NONE;
    Move ttMove = (ttHit && Move(tte->move).is_ok()) ? Move(tte->move) : Move::none();
    
    if (!PvNode && ttHit && tte->depth >= 0
        && ttValue != VALUE_NONE
        && ((tte->bound == BOUND_UPPER && ttValue <= alpha) ||
            (tte->bound == BOUND_LOWER && ttValue >= beta) ||
            (tte->bound == BOUND_EXACT)))
        return ttValue;
    
    // Stand pat
    Value bestValue;
    if (inCheck) {
        bestValue = Value(-int(VALUE_MATE) + ss->ply);
    } else {
        bestValue = evaluate_position(pos);
        ss->staticEval = bestValue;
        
        if (bestValue >= beta)
            return bestValue;
        if (bestValue > alpha)
            alpha = bestValue;
    }
    
    // Generate moves
    ExtMove moveList[MAX_MOVES];
    ExtMove* end;
    
    if (inCheck)
        end = generate<EVASIONS>(pos, moveList);
    else
        end = generate<CAPTURES>(pos, moveList);
    
    // Score and sort
    for (ExtMove* it = moveList; it != end; ++it) {
        if (it->move == ttMove)
            it->value = 2000000000;
        else if (pos.is_capture(it->move)) {
            PieceType captured = type_of(pos.piece_on(it->move.to_sq()));
            if (it->move.is_enpassant()) captured = PAWN;
            int seeVal = piece_value[captured];
            it->value = 1000000 + seeVal;
        } else {
            it->value = 0;
        }
    }
    std::sort(moveList, end);
    
    Move bestMove = Move::none();
    
    for (ExtMove* it = moveList; it != end; ++it) {
        if (!is_legal(pos, it->move))
            continue;
        
        if (!inCheck && !pos.is_capture_or_promotion(it->move))
            continue; // Delta pruning: only captures/checks in qsearch
        
        StateInfo st;
        pos.do_move(it->move, st);
        Value value = -qsearch<false>(pos, ss+1, -beta, -alpha);
        pos.undo_move(it->move, st);
        
        if (!running)
            return VALUE_NONE;
        
        if (value > bestValue) {
            bestValue = value;
            bestMove = it->move;
            
            if (value > alpha) {
                alpha = value;
                if (value >= beta)
                    break;
            }
        }
    }
    
    // Store in TT
    Bound b = (bestValue >= beta) ? BOUND_LOWER : BOUND_UPPER;
    if (PvNode && bestValue > alpha && bestValue < beta)
        b = BOUND_EXACT;
    
    TT.store(pos.st.key, score_to_tt(bestValue, ss->ply), PvNode, b, Depth(0), bestMove, ss->staticEval);
    
    return bestValue;
}

void Search::update_quiet_stats(BoardState& pos, SearchStack* ss, Move bestMove, 
                                Move* quiets, int quietCount, int bonus) {
    Color us = pos.sideToMove;
    
    if (!pos.is_capture(bestMove)) {
        // Update main history with Stockfish-style formula
        int16_t& entry = mainHistory[us][bestMove.from_to()];
        entry += bonus - entry * std::abs(bonus) / 16384;
        entry = std::clamp(entry, (int16_t)-8192, (int16_t)8192);
        
        // Update counter-move history
        if (ss->ply > 0 && (ss-1)->currentMove.is_ok()) {
            Piece prevPiece = pos.piece_on((ss-1)->currentMove.to_sq());
            Square prevTo = (ss-1)->currentMove.to_sq();
            History.counterMove.set(prevPiece, prevTo, bestMove);
        }
    }
    
    // Penalize non-best quiets
    int malus = -bonus;
    for (int i = 0; i < quietCount; ++i) {
        if (quiets[i] != bestMove) {
            int16_t& entry = mainHistory[us][quiets[i].from_to()];
            entry += malus - entry * std::abs(malus) / 16384;
            entry = std::clamp(entry, (int16_t)-8192, (int16_t)8192);
        }
    }
}

void Search::update_capture_stats(Move m, Move* captures, int captureCount, int bonus) {
    // Simplified capture history update
    (void)m;
    (void)captures;
    (void)captureCount;
    (void)bonus;
}

int Search::reduction(bool i, Depth d, int mn, Value delta, Value rootDelta) {
    (void)delta;
    (void)rootDelta;
    
    // Use advanced reduction tables
    return Reductions::get_reduction(d, mn, i, false);
}

// Explicit instantiations
template Value Search::negamax<true>(BoardState&, SearchStack*, Value, Value, Depth, bool);
template Value Search::negamax<false>(BoardState&, SearchStack*, Value, Value, Depth, bool);
template Value Search::qsearch<true>(BoardState&, SearchStack*, Value, Value);
template Value Search::qsearch<false>(BoardState&, SearchStack*, Value, Value);

// Bench positions - standard test positions for benchmarking
static const char* BenchFENs[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // Start
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  // Endgame
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",  // Promotion
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",  // Complex
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",  // Open
    "3K4/8/8/8/8/8/4k3/4b3 w - - 0 1",  // King endgame
    "8/8/8/3k4/3P4/3K4/8/8 b - - 0 1",  // Opposition
    nullptr
};

u64 Search::bench(int depth) {
    u64 totalNodes = 0;
    BoardState pos;
    
    std::cout << "Benchmark started (depth " << depth << ")" << std::endl;
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; BenchFENs[i]; ++i) {
        pos.set_fen(BenchFENs[i]);
        clear();
        TT.new_search();
        
        SearchLimits limits;
        limits.depth = depth;
        limits.multiPV = 1;
        
        u64 nodesBefore = nodesSearched;
        start = std::chrono::steady_clock::now();
        
        // Generate root moves
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<NON_EVASIONS>(pos, moves);
        rootMoves.clear();
        for (ExtMove* it = moves; it != end; ++it) {
            if (is_legal(pos, it->move))
                rootMoves.emplace_back(it->move);
        }
        
        // Quick search
        SearchStack ss[MAX_PLY + 12];
        Move pv[MAX_PLY + 1] = {};
        for (int j = 0; j < MAX_PLY + 12; ++j) {
            ss[j].pv = (j < MAX_PLY) ? &pv[j] : nullptr;
            ss[j].ply = j;
        }
        
        // Search to specified depth
        running = true;
        for (int d = 1; d <= depth; ++d) {
            negamax<true>(pos, ss, -VALUE_INFINITE, VALUE_INFINITE, Depth(d), false);
        }
        running = false;
        
        u64 nodes = nodesSearched - nodesBefore;
        totalNodes += nodes;
        std::cout << "Position " << (i+1) << ": " << nodes << " nodes" << std::endl;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "==================" << std::endl;
    std::cout << "Total: " << totalNodes << " nodes" << std::endl;
    std::cout << "Time:  " << elapsed << " ms" << std::endl;
    if (elapsed > 0)
        std::cout << "NPS:   " << (totalNodes * 1000 / elapsed) << std::endl;
    
    return totalNodes;
}

} // namespace Nexus
