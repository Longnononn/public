#include "board.h"
#include <sstream>
#include <cstring>
#include <cstdlib>

namespace Nexus {

std::array<std::array<Key, SQUARE_NB>, PIECE_NB> ZobristPiece;
std::array<Key, FILE_NB> ZobristEnpassant;
std::array<Key, CASTLING_RIGHT_NB> ZobristCastling;
Key ZobristSide;
Key ZobristNoPawns;

// Pseudo-random number generator for Zobrist keys (SplitMix64)
static u64 splitmix64(u64& state) {
    u64 z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void init_zobrist() {
    u64 state = 0x123456789abcdef0ULL;
    for (int pc = 0; pc < PIECE_NB; ++pc)
        for (int sq = 0; sq < SQUARE_NB; ++sq)
            ZobristPiece[pc][sq] = make_key(splitmix64(state));
    for (int f = 0; f < FILE_NB; ++f)
        ZobristEnpassant[f] = make_key(splitmix64(state));
    for (int cr = 0; cr < CASTLING_RIGHT_NB; ++cr)
        ZobristCastling[cr] = make_key(splitmix64(state));
    ZobristSide = make_key(splitmix64(state));
    ZobristNoPawns = make_key(splitmix64(state));
}

u64 BoardState::attackers_to(Square s, u64 occupied) const {
    return (pawn_attacks(WHITE, s) & colorBB[BLACK] & pieceBB[PAWN])
         | (pawn_attacks(BLACK, s) & colorBB[WHITE] & pieceBB[PAWN])
         | (knight_attacks(s) & pieceBB[KNIGHT])
         | (king_attacks(s) & pieceBB[KING])
         | (rook_attacks(s, occupied) & (pieceBB[ROOK] | pieceBB[QUEEN]))
         | (bishop_attacks(s, occupied) & (pieceBB[BISHOP] | pieceBB[QUEEN]));
}

u64 BoardState::blockers_for_king(Color c) const {
    u64 blockers = 0;
    Square ksq = kingSquare[c];
    u64 occupied = all_pieces() ^ square_bb(ksq);
    u64 snipers = ((PseudoAttacks[ROOK][ksq] & (pieceBB[ROOK] | pieceBB[QUEEN]))
                  | (PseudoAttacks[BISHOP][ksq] & (pieceBB[BISHOP] | pieceBB[QUEEN])))
                  & pieces(~c);
    
    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        u64 b = BetweenBB[ksq][sniperSq] & occupied;
        if (b && !more_than_one(b))
            blockers |= b & pieces(c);
    }
    return blockers;
}

u64 BoardState::pinners_for_king(Color c) const {
    u64 pinners = 0;
    Square ksq = kingSquare[c];
    u64 occupied = all_pieces() ^ square_bb(ksq);
    u64 snipers = ((PseudoAttacks[ROOK][ksq] & (pieceBB[ROOK] | pieceBB[QUEEN]))
                  | (PseudoAttacks[BISHOP][ksq] & (pieceBB[BISHOP] | pieceBB[QUEEN])))
                  & pieces(~c);
    
    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        u64 b = BetweenBB[ksq][sniperSq] & occupied;
        if (b && !more_than_one(b) && (b & pieces(c)))
            pinners |= square_bb(sniperSq);
    }
    return pinners;
}

bool BoardState::gives_check(Move m) const {
    Square from = m.from_sq();
    Square to = m.to_sq();
    PieceType pt = type_of(board[from]);
    Color us = sideToMove;
    Square ksq = kingSquare[~us];
    
    // Direct check
    switch (pt) {
        case PAWN:   return pawn_attacks(us, to) & square_bb(ksq);
        case KNIGHT: return knight_attacks(to) & square_bb(ksq);
        case BISHOP: return bishop_attacks(to, all_pieces() ^ square_bb(from)) & square_bb(ksq);
        case ROOK:   return rook_attacks(to, all_pieces() ^ square_bb(from)) & square_bb(ksq);
        case QUEEN:  return queen_attacks(to, all_pieces() ^ square_bb(from)) & square_bb(ksq);
        default: break;
    }
    
    // Discovered check
    if (m.is_enpassant()) {
        // Complex case: removing pawn may open lines
        return attackers_to(ksq, all_pieces() ^ square_bb(from) ^ square_bb(to)
                          ^ square_bb(make_square(file_of(to), rank_of(from)))) & pieces(us);
    }
    
    if (m.is_castling()) {
        // King doesn't directly check, rook might
        Square rto = (to > from) ? Square(to - 1) : Square(to + 1);
        return rook_attacks(rto, all_pieces() ^ square_bb(from)) & square_bb(ksq);
    }
    
    // Check if move opens a line for another piece
    return (attackers_to(ksq, all_pieces() ^ square_bb(from) ^ square_bb(to)) & pieces(us)) != 0;
}

bool BoardState::advanced_pawn_push(Move m) const {
    return type_of(board[m.from_sq()]) == PAWN
        && relative_rank(sideToMove, rank_of(m.to_sq())) >= RANK_6;
}

bool BoardState::is_capture(Move m) const {
    return (all_pieces() & square_bb(m.to_sq())) || m.is_enpassant();
}

bool BoardState::is_capture_or_promotion(Move m) const {
    return is_capture(m) || m.is_promotion();
}

void BoardState::set_piece(Square s, Piece p) {
    pieceBB[type_of(p)] |= square_bb(s);
    colorBB[color_of(p)] |= square_bb(s);
    board[s] = p;
    st.key ^= ZobristPiece[p][s];
    
    if (type_of(p) == PAWN)
        st.pawnKey ^= ZobristPiece[p][s];
    else
        st.nonPawnMaterial[color_of(p)] += Value(piece_value[type_of(p)]);
}

void BoardState::remove_piece(Square s) {
    Piece p = board[s];
    pieceBB[type_of(p)] ^= square_bb(s);
    colorBB[color_of(p)] ^= square_bb(s);
    board[s] = NO_PIECE;
    st.key ^= ZobristPiece[p][s];
    
    if (type_of(p) == PAWN)
        st.pawnKey ^= ZobristPiece[p][s];
    else
        st.nonPawnMaterial[color_of(p)] -= Value(piece_value[type_of(p)]);
}

void BoardState::move_piece(Square from, Square to) {
    Piece p = board[from];
    u64 from_to = square_bb(from) | square_bb(to);
    pieceBB[type_of(p)] ^= from_to;
    colorBB[color_of(p)] ^= from_to;
    board[from] = NO_PIECE;
    board[to] = p;
    st.key ^= ZobristPiece[p][from] ^ ZobristPiece[p][to];
    
    if (type_of(p) == PAWN)
        st.pawnKey ^= ZobristPiece[p][from] ^ ZobristPiece[p][to];
}

void BoardState::do_move(Move m, StateInfo& newSt) {
    newSt = st; // Copy current state
    newSt.previous = &st;
    st.previous = nullptr; // Not used after copy
    
    // Save pointer to new state (caller must keep newSt alive)
    Key key = st.key;
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = board[from];
    PieceType pt = type_of(pc);
    
    st.capturedPiece = board[to];
    
    if (m.is_castling()) {
        // King move
        move_piece(from, to);
        kingSquare[us] = to;
        
        // Rook move
        Square rfrom = (to > from) ? Square(to + 1) : Square(to - 2);
        Square rto = (to > from) ? Square(to - 1) : Square(to + 1);
        move_piece(rfrom, rto);
        
        st.key ^= ZobristCastling[st.castlingRights];
        st.castlingRights &= ~(us == WHITE ? (WHITE_OO | WHITE_OOO) : (BLACK_OO | BLACK_OOO));
        st.key ^= ZobristCastling[st.castlingRights];
    }
    else if (m.is_enpassant()) {
        Square capsq = make_square(file_of(to), rank_of(from));
        remove_piece(capsq);
        move_piece(from, to);
        st.key ^= ZobristPiece[make_piece(~us, PAWN)][capsq];
    }
    else if (m.is_promotion()) {
        remove_piece(from);
        Piece promo = make_piece(us, m.promotion_type());
        set_piece(to, promo);
    }
    else {
        // Normal move (including captures)
        if (st.capturedPiece != NO_PIECE)
            remove_piece(to);
        move_piece(from, to);
    }
    
    // Update king square
    if (pt == KING)
        kingSquare[us] = to;
    
    // Update castling rights
    update_castling_rights(from, to);
    
    // Update en passant
    if (st.epSquare != SQ_NONE)
        st.key ^= ZobristEnpassant[file_of(st.epSquare)];
    
    if (pt == PAWN && std::abs(rank_of(to) - rank_of(from)) == 2) {
        st.epSquare = make_square(file_of(from), Rank((rank_of(from) + rank_of(to)) / 2));
        st.key ^= ZobristEnpassant[file_of(st.epSquare)];
    } else {
        st.epSquare = SQ_NONE;
    }
    
    // Update rule50
    if (pt == PAWN || st.capturedPiece != NO_PIECE)
        st.rule50 = 0;
    else
        st.rule50++;
    
    // Update side to move
    st.key ^= ZobristSide;
    sideToMove = ~us;
    gamePly++;
    st.pliesFromNull = st.pliesFromNull + 1;
    
    // Store key in history
    keyHistory[historyIndex++] = st.key;
}

void BoardState::undo_move(Move m, const StateInfo& restoredSt) {
    Color us = ~sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = board[to];
    
    if (m.is_castling()) {
        Square rfrom = (to > from) ? Square(to + 1) : Square(to - 2);
        Square rto = (to > from) ? Square(to - 1) : Square(to + 1);
        move_piece(rto, rfrom);
        move_piece(to, from);
        kingSquare[us] = from;
    }
    else if (m.is_enpassant()) {
        move_piece(to, from);
        Square capsq = make_square(file_of(to), rank_of(from));
        set_piece(capsq, make_piece(~us, PAWN));
    }
    else if (m.is_promotion()) {
        remove_piece(to);
        set_piece(from, make_piece(us, PAWN));
        if (restoredSt.capturedPiece != NO_PIECE)
            set_piece(to, restoredSt.capturedPiece);
    }
    else {
        move_piece(to, from);
        if (restoredSt.capturedPiece != NO_PIECE)
            set_piece(to, restoredSt.capturedPiece);
    }
    
    if (type_of(pc) == KING)
        kingSquare[us] = from;
    
    sideToMove = us;
    gamePly--;
    historyIndex--;
    
    // Restore state
    st = restoredSt;
}

void BoardState::do_null_move(StateInfo& newSt) {
    newSt = st;
    newSt.previous = &st;
    
    if (st.epSquare != SQ_NONE) {
        st.key ^= ZobristEnpassant[file_of(st.epSquare)];
        st.epSquare = SQ_NONE;
    }
    
    st.key ^= ZobristSide;
    st.rule50++;
    st.pliesFromNull = 0;
    sideToMove = ~sideToMove;
    gamePly++;
    keyHistory[historyIndex++] = st.key;
}

void BoardState::undo_null_move(const StateInfo& restoredSt) {
    sideToMove = ~sideToMove;
    gamePly--;
    historyIndex--;
    st = restoredSt;
}

void BoardState::update_castling_rights(Square from, Square to) {
    st.key ^= ZobristCastling[st.castlingRights];
    
    if (from == SQ_E1) st.castlingRights &= ~(WHITE_OO | WHITE_OOO);
    else if (from == SQ_E8) st.castlingRights &= ~(BLACK_OO | BLACK_OOO);
    else if (from == SQ_A1 || to == SQ_A1) st.castlingRights &= ~WHITE_OOO;
    else if (from == SQ_H1 || to == SQ_H1) st.castlingRights &= ~WHITE_OO;
    else if (from == SQ_A8 || to == SQ_A8) st.castlingRights &= ~BLACK_OOO;
    else if (from == SQ_H8 || to == SQ_H8) st.castlingRights &= ~BLACK_OO;
    
    st.key ^= ZobristCastling[st.castlingRights];
}

bool BoardState::is_draw(int ply) const {
    if (st.rule50 > 99 && (!is_check() || Move::none().is_ok()))
        return true;
    
    // Repetition
    int end = std::min(st.rule50, historyIndex);
    for (int i = 4; i <= end; i += 2)
        if (keyHistory[historyIndex - i] == st.key)
            return true;
    
    return false;
}

bool BoardState::has_repeated() const {
    int end = std::min(st.rule50, historyIndex);
    for (int i = 4; i <= end; i += 2)
        if (keyHistory[historyIndex - i] == st.key)
            return true;
    return false;
}

bool BoardState::is_material_draw() const {
    // K vs K
    if (!pieceBB[PAWN] && !pieceBB[ROOK] && !pieceBB[QUEEN]) {
        // K+B vs K or K+N vs K
        if (!pieceBB[KNIGHT] && !pieceBB[BISHOP])
            return true;
        
        // K+B vs K+B (same color bishops)
        if (!pieceBB[KNIGHT] && count_bits(pieceBB[BISHOP]) == 1)
            return false; // K+B vs K+N is drawish but not automatic
        
        // K+N vs K+N
        if (!pieceBB[BISHOP] && count_bits(pieceBB[KNIGHT]) <= 1)
            return true;
    }
    return false;
}

void BoardState::set_fen(const std::string& fen) {
    // Reset
    std::fill(pieceBB.begin(), pieceBB.end(), 0);
    std::fill(colorBB.begin(), colorBB.end(), 0);
    std::fill(board.begin(), board.end(), NO_PIECE);
    std::fill(pieceCount.begin(), pieceCount.end(), 0);
    st = StateInfo{};
    gamePly = 0;
    historyIndex = 0;
    
    std::istringstream iss(fen);
    std::string token;
    
    // Piece placement
    iss >> token;
    Square sq = SQ_A8;
    for (char c : token) {
        if (c == '/')
            sq = Square(sq - 16); // Next rank down
        else if (isdigit(c))
            sq = Square(sq + (c - '0'));
        else {
            Color col = isupper(c) ? WHITE : BLACK;
            PieceType pt;
            switch (tolower(c)) {
                case 'p': pt = PAWN; break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK; break;
                case 'q': pt = QUEEN; break;
                case 'k': pt = KING; break;
                default: continue;
            }
            set_piece(sq, make_piece(col, pt));
            if (pt == KING) kingSquare[col] = sq;
            sq = Square(sq + 1);
        }
    }
    
    // Side to move
    iss >> token;
    sideToMove = (token == "w") ? WHITE : BLACK;
    if (sideToMove == BLACK)
        st.key ^= ZobristSide;
    
    // Castling
    iss >> token;
    st.castlingRights = NO_CASTLING;
    for (char c : token) {
        if (c == 'K') st.castlingRights |= WHITE_OO;
        if (c == 'Q') st.castlingRights |= WHITE_OOO;
        if (c == 'k') st.castlingRights |= BLACK_OO;
        if (c == 'q') st.castlingRights |= BLACK_OOO;
    }
    st.key ^= ZobristCastling[st.castlingRights];
    
    // En passant
    iss >> token;
    st.epSquare = (token == "-") ? SQ_NONE : Square((token[0] - 'a') + (token[1] - '1') * 8);
    if (st.epSquare != SQ_NONE)
        st.key ^= ZobristEnpassant[file_of(st.epSquare)];
    
    // Rule50
    iss >> st.rule50;
    
    // Fullmove clock (ignored for now)
    iss >> token;
    
    // Count pieces
    for (Square s = SQ_A1; s <= SQ_H8; s = Square(s + 1)) {
        if (board[s] != NO_PIECE)
            pieceCount[color_of(board[s])]++;
    }
}

std::string BoardState::fen() const {
    std::ostringstream oss;
    
    for (Rank r = RANK_8; r >= RANK_1; r = Rank(r - 1)) {
        int empty = 0;
        for (File f = FILE_A; f <= FILE_H; f = File(f + 1)) {
            Square sq = make_square(f, r);
            if (board[sq] == NO_PIECE)
                empty++;
            else {
                if (empty) { oss << empty; empty = 0; }
                char c;
                switch (type_of(board[sq])) {
                    case PAWN: c = 'p'; break;
                    case KNIGHT: c = 'n'; break;
                    case BISHOP: c = 'b'; break;
                    case ROOK: c = 'r'; break;
                    case QUEEN: c = 'q'; break;
                    case KING: c = 'k'; break;
                    default: c = '?'; break;
                }
                if (color_of(board[sq]) == WHITE) c = toupper(c);
                oss << c;
            }
        }
        if (empty) oss << empty;
        if (r > RANK_1) oss << '/';
    }
    
    oss << ' ' << (sideToMove == WHITE ? 'w' : 'b');
    
    oss << ' ';
    if (st.castlingRights == NO_CASTLING) oss << '-';
    else {
        if (st.castlingRights & WHITE_OO) oss << 'K';
        if (st.castlingRights & WHITE_OOO) oss << 'Q';
        if (st.castlingRights & BLACK_OO) oss << 'k';
        if (st.castlingRights & BLACK_OOO) oss << 'q';
    }
    
    oss << ' ';
    if (st.epSquare == SQ_NONE) oss << '-';
    else {
        oss << char('a' + file_of(st.epSquare));
        oss << char('1' + rank_of(st.epSquare));
    }
    
    oss << ' ' << st.rule50;
    oss << ' ' << (gamePly / 2 + 1);
    
    return oss.str();
}

bool BoardState::is_ok() const {
    return pieceBB[KING] == (pieces(WHITE, KING) | pieces(BLACK, KING))
        && count_bits(pieces(WHITE, KING)) == 1
        && count_bits(pieces(BLACK, KING)) == 1;
}

} // namespace Nexus
