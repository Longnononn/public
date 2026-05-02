#include "movegen.h"
#include <algorithm>

namespace Nexus {

namespace {

// Pawn pushes
ExtMove* generate_pawn_pushes(const BoardState& pos, ExtMove* moves, u64 pawns, Color us) {
    Direction up = pawn_push(us);
    u64 empty = pos.empty_squares();
    
    // Single pushes
    u64 push1 = shift(pawns, up) & empty;
    while (push1) {
        Square to = pop_lsb(push1);
        Square from = Square(to - up);
        
        if (relative_rank(us, rank_of(to)) == RANK_8) {
            *moves++ = ExtMove{make_promotion(from, to, QUEEN), 0};
            *moves++ = ExtMove{make_promotion(from, to, ROOK), 0};
            *moves++ = ExtMove{make_promotion(from, to, BISHOP), 0};
            *moves++ = ExtMove{make_promotion(from, to, KNIGHT), 0};
        } else {
            *moves++ = ExtMove{make_move(from, to), 0};
        }
    }
    
    // Double pushes
    u64 push2 = shift(push1 & (us == WHITE ? Rank3BB : Rank6BB), up) & empty;
    while (push2) {
        Square to = pop_lsb(push2);
        Square from = Square(to - 2 * up);
        *moves++ = ExtMove{make_move(from, to), 0};
    }
    
    return moves;
}

// Pawn captures (including en passant)
ExtMove* generate_pawn_captures(const BoardState& pos, ExtMove* moves, u64 pawns, Color us) {
    Direction up_east = pawn_capture_east(us);
    Direction up_west = pawn_capture_west(us);
    u64 enemies = pos.pieces(~us);
    
    // East captures
    u64 caps_east = shift(pawns, up_east) & enemies;
    while (caps_east) {
        Square to = pop_lsb(caps_east);
        Square from = Square(to - up_east);
        if (relative_rank(us, rank_of(to)) == RANK_8) {
            *moves++ = ExtMove{make_promotion(from, to, QUEEN), 0};
            *moves++ = ExtMove{make_promotion(from, to, ROOK), 0};
            *moves++ = ExtMove{make_promotion(from, to, BISHOP), 0};
            *moves++ = ExtMove{make_promotion(from, to, KNIGHT), 0};
        } else {
            *moves++ = ExtMove{make_move(from, to), 0};
        }
    }
    
    // West captures
    u64 caps_west = shift(pawns, up_west) & enemies;
    while (caps_west) {
        Square to = pop_lsb(caps_west);
        Square from = Square(to - up_west);
        if (relative_rank(us, rank_of(to)) == RANK_8) {
            *moves++ = ExtMove{make_promotion(from, to, QUEEN), 0};
            *moves++ = ExtMove{make_promotion(from, to, ROOK), 0};
            *moves++ = ExtMove{make_promotion(from, to, BISHOP), 0};
            *moves++ = ExtMove{make_promotion(from, to, KNIGHT), 0};
        } else {
            *moves++ = ExtMove{make_move(from, to), 0};
        }
    }
    
    // En passant
    if (pos.st.epSquare != SQ_NONE) {
        u64 epBB = square_bb(pos.st.epSquare);
        u64 epPawns = pawns & ((us == WHITE ? shift(epBB, SOUTH_EAST) | shift(epBB, SOUTH_WEST)
                                           : shift(epBB, NORTH_EAST) | shift(epBB, NORTH_WEST)));
        while (epPawns) {
            Square from = pop_lsb(epPawns);
            *moves++ = ExtMove{make_enpassant(from, pos.st.epSquare), 0};
        }
    }
    
    return moves;
}

// Generate moves for non-pawn pieces
ExtMove* generate_piece_moves(const BoardState& pos, ExtMove* moves, PieceType pt, Color us, u64 targets) {
    u64 pieces = pos.pieces(us, pt);
    
    while (pieces) {
        Square from = pop_lsb(pieces);
        u64 attacks;
        switch (pt) {
            case KNIGHT: attacks = knight_attacks(from); break;
            case BISHOP: attacks = bishop_attacks(from, pos.all_pieces()); break;
            case ROOK:   attacks = rook_attacks(from, pos.all_pieces()); break;
            case QUEEN:  attacks = queen_attacks(from, pos.all_pieces()); break;
            case KING:   attacks = king_attacks(from); break;
            default:     attacks = 0; break;
        }
        
        attacks &= targets;
        while (attacks) {
            Square to = pop_lsb(attacks);
            *moves++ = ExtMove{make_move(from, to), 0};
        }
    }
    
    return moves;
}

// Castling
ExtMove* generate_castling(const BoardState& pos, ExtMove* moves, Color us, CastlingRights cr, 
                           Square kfrom, Square rfrom, Square kto, Square rto) {
    if (!pos.can_castle(cr)) return moves;
    
    // Check rook is on correct square
    if (pos.piece_on(rfrom) != make_piece(us, ROOK)) return moves;
    
    // Check path is clear
    u64 between = BetweenBB[kfrom][rfrom] | square_bb(rto);
    if (between & pos.all_pieces()) return moves;
    
    // Check king doesn't pass through check
    u64 king_path = BetweenBB[kfrom][kto] | square_bb(kto);
    while (king_path) {
        Square sq = pop_lsb(king_path);
        if (pos.attackers_to(sq) & pos.pieces(~us)) return moves;
    }
    
    *moves++ = ExtMove{make_castling(kfrom, kto), 0};
    return moves;
}

// Check if king is safe after a move
bool is_legal_move(const BoardState& pos, Move m) {
    Color us = pos.sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = pos.piece_on(from);
    PieceType pt = type_of(pc);
    
    Square ksq = pos.kingSquare[us];
    
    // If moving the king, check destination is safe
    if (pt == KING) {
        return !(pos.attackers_to(to, pos.all_pieces() ^ square_bb(from)) & pos.pieces(~us));
    }
    
    // Check if we're in check
    u64 checkers = pos.checkers();
    if (checkers) {
        // If double check, only king moves are legal
        if (more_than_one(checkers))
            return false;
        
        Square checkerSq = lsb(checkers);
        PieceType checkerType = type_of(pos.piece_on(checkerSq));
        
        // Must capture checker or block line
        if (checkerType == KNIGHT || checkerType == PAWN) {
            // Must capture
            if (to != checkerSq) return false;
        } else {
            // Can capture or block
            if (!(square_bb(to) & (BetweenBB[ksq][checkerSq] | square_bb(checkerSq))))
                return false;
        }
    }
    
    // Check if pinned
    u64 pinned = pos.blockers_for_king(us);
    if (pinned & square_bb(from)) {
        // Pinned piece must move along the pin line
        if (!(square_bb(to) & LineBB[from][ksq]))
            return false;
    }
    
    // En passant special check
    if (m.is_enpassant()) {
        Square capsq = make_square(file_of(to), rank_of(from));
        u64 occ = pos.all_pieces() ^ square_bb(from) ^ square_bb(capsq) ^ square_bb(to);
        return !(rook_attacks(ksq, occ) & pos.pieces(~us, ROOK, QUEEN))
            && !(bishop_attacks(ksq, occ) & pos.pieces(~us, BISHOP, QUEEN));
    }
    
    return true;
}

} // anonymous namespace

template<GenType Type>
ExtMove* generate(const BoardState& pos, ExtMove* moveList) {
    static_assert(Type != LEGAL, "LEGAL generation uses pseudolegal + filter");
    
    Color us = pos.sideToMove;
    u64 targets;
    
    if (Type == CAPTURES) {
        targets = pos.pieces(~us);
        moveList = generate_pawn_captures(pos, moveList, pos.pieces(us, PAWN), us);
    }
    else if (Type == QUIETS) {
        targets = pos.empty_squares();
        moveList = generate_pawn_pushes(pos, moveList, pos.pieces(us, PAWN), us);
    }
    else if (Type == QUIET_CHECKS) {
        targets = pos.empty_squares();
        // Simplified: just generate all quiets (filter checks later)
        moveList = generate_pawn_pushes(pos, moveList, pos.pieces(us, PAWN), us);
    }
    else if (Type == EVASIONS) {
        u64 checkers = pos.checkers();
        if (more_than_one(checkers)) {
            // Double check: only king moves
            return generate_piece_moves(pos, moveList, KING, us, ~pos.pieces(us));
        }
        
        Square checkerSq = lsb(checkers);
        targets = square_bb(checkerSq) | BetweenBB[pos.kingSquare[us]][checkerSq];
        moveList = generate_pawn_captures(pos, moveList, pos.pieces(us, PAWN), us);
        moveList = generate_pawn_pushes(pos, moveList, pos.pieces(us, PAWN) & ~pos.blockers_for_king(us), us);
    }
    else if (Type == NON_EVASIONS) {
        targets = ~pos.pieces(us);
        moveList = generate_pawn_captures(pos, moveList, pos.pieces(us, PAWN), us);
        moveList = generate_pawn_pushes(pos, moveList, pos.pieces(us, PAWN), us);
    }
    
    // Non-pawn moves
    if (Type != EVASIONS || !more_than_one(pos.checkers())) {
        moveList = generate_piece_moves(pos, moveList, KNIGHT, us, targets);
        moveList = generate_piece_moves(pos, moveList, BISHOP, us, targets);
        moveList = generate_piece_moves(pos, moveList, ROOK, us, targets);
        moveList = generate_piece_moves(pos, moveList, QUEEN, us, targets);
    }
    
    // King moves (always allowed except castling during evasions with multiple checkers)
    u64 kingTargets = (Type == CAPTURES) ? pos.pieces(~us)
                     : (Type == QUIETS || Type == QUIET_CHECKS) ? pos.empty_squares()
                     : (Type == EVASIONS) ? ~pos.pieces(us)
                     : ~pos.pieces(us);
    moveList = generate_piece_moves(pos, moveList, KING, us, kingTargets);
    
    // Castling
    if (Type != CAPTURES && Type != EVASIONS) {
        if (us == WHITE) {
            moveList = generate_castling(pos, moveList, us, WHITE_OO, SQ_E1, SQ_H1, SQ_G1, SQ_F1);
            moveList = generate_castling(pos, moveList, us, WHITE_OOO, SQ_E1, SQ_A1, SQ_C1, SQ_D1);
        } else {
            moveList = generate_castling(pos, moveList, us, BLACK_OO, SQ_E8, SQ_H8, SQ_G8, SQ_F8);
            moveList = generate_castling(pos, moveList, us, BLACK_OOO, SQ_E8, SQ_A8, SQ_C8, SQ_D8);
        }
    }
    
    return moveList;
}

template ExtMove* generate<CAPTURES>(const BoardState&, ExtMove*);
template ExtMove* generate<QUIETS>(const BoardState&, ExtMove*);
template ExtMove* generate<EVASIONS>(const BoardState&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const BoardState&, ExtMove*);
template ExtMove* generate<QUIET_CHECKS>(const BoardState&, ExtMove*);

bool is_legal(const BoardState& pos, Move m) {
    return is_legal_move(pos, m);
}

u64 perft(BoardState& pos, int depth) {
    if (depth == 0) return 1;
    
    ExtMove moves[MAX_MOVES];
    ExtMove* end;
    
    if (pos.is_check()) {
        end = generate<EVASIONS>(pos, moves);
    } else {
        end = generate<NON_EVASIONS>(pos, moves);
    }
    
    if (depth == 1) {
        int count = 0;
        for (ExtMove* it = moves; it != end; ++it) {
            if (is_legal(pos, it->move)) count++;
        }
        return count;
    }
    
    u64 nodes = 0;
    StateInfo st;
    for (ExtMove* it = moves; it != end; ++it) {
        if (!is_legal(pos, it->move)) continue;
        pos.do_move(it->move, st);
        nodes += perft(pos, depth - 1);
        pos.undo_move(it->move, st);
    }
    return nodes;
}

// Explicit instantiations
template ExtMove* generate<LEGAL>(const BoardState& pos, ExtMove* moveList);
template ExtMove* generate<CAPTURES>(const BoardState& pos, ExtMove* moveList);
template ExtMove* generate<QUIETS>(const BoardState& pos, ExtMove* moveList);
template ExtMove* generate<EVASIONS>(const BoardState& pos, ExtMove* moveList);
template ExtMove* generate<NON_EVASIONS>(const BoardState& pos, ExtMove* moveList);
template ExtMove* generate<QUIET_CHECKS>(const BoardState& pos, ExtMove* moveList);

} // namespace Nexus
