#pragma once

#include "types.h"
#include "bitboard.h"
#include "move.h"
#include <string>
#include <array>
#include <stack>

namespace Nexus {

struct StateInfo {
    Key key;
    Key pawnKey;
    Key materialKey;
    Value nonPawnMaterial[COLOR_NB];
    CastlingRights castlingRights;
    int rule50;
    Square epSquare;
    int pliesFromNull;
    Value psqScore;
    
    // Capture info for undo
    Piece capturedPiece;
    
    // Previous state pointer (for undo chain)
    StateInfo* previous;
};

class BoardState {
public:
    BoardState() = default;
    
    // Core bitboards
    std::array<u64, PIECE_TYPE_NB> pieceBB;
    std::array<u64, COLOR_NB> colorBB;
    
    // Piece on each square
    std::array<Piece, SQUARE_NB> board;
    
    // King squares
    std::array<Square, COLOR_NB> kingSquare;
    
    // Game state
    std::array<int, COLOR_NB> pieceCount;
    Color sideToMove;
    StateInfo st;
    int gamePly;
    
    // Repetition detection
    std::array<Key, 512> keyHistory;
    int historyIndex;
    
    // Accessors
    u64 pieces(Color c) const { return colorBB[c]; }
    u64 pieces(PieceType pt) const { return pieceBB[pt]; }
    u64 pieces(Color c, PieceType pt) const { return colorBB[c] & pieceBB[pt]; }
    u64 pieces(Color c, PieceType pt1, PieceType pt2) const { 
        return colorBB[c] & (pieceBB[pt1] | pieceBB[pt2]); 
    }
    u64 all_pieces() const { return colorBB[WHITE] | colorBB[BLACK]; }
    u64 empty_squares() const { return ~all_pieces(); }
    
    Piece piece_on(Square s) const { return board[s]; }
    Square king_square(Color c) const { return kingSquare[c]; }
    
    // Checks
    bool is_check() const { return checkers(); }
    u64 checkers() const { return attackers_to(kingSquare[sideToMove]) & pieces(~sideToMove); }
    u64 blockers_for_king(Color c) const;
    u64 pinners_for_king(Color c) const;
    u64 attackers_to(Square s, u64 occupied) const;
    u64 attackers_to(Square s) const { return attackers_to(s, all_pieces()); }
    
    bool gives_check(Move m) const;
    bool advanced_pawn_push(Move m) const;
    
    // Move application
    void do_move(Move m, StateInfo& newSt);
    void undo_move(Move m, const StateInfo& restoredSt);
    void do_null_move(StateInfo& newSt);
    void undo_null_move(const StateInfo& restoredSt);
    
    // State queries
    bool is_capture(Move m) const;
    bool is_capture_or_promotion(Move m) const;
    bool is_check() const { return checkers(); }
    PieceType captured_piece_type() const { return type_of(st.capturedPiece); }
    bool can_castle(CastlingRights cr) const { return st.castlingRights & cr; }
    bool is_draw(int ply) const;
    bool has_repeated() const;
    bool is_material_draw() const;
    
    // FEN
    void set_fen(const std::string& fen);
    std::string fen() const;
    
    // Validity
    bool is_ok() const;
    
    // Static evaluation (incrementally updated)
    Value psq_score() const { return st.psqScore; }
    
private:
    void set_piece(Square s, Piece p);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    
    void update_key(Square s, Piece p);
    void update_castling_rights(Square from, Square to);
};

// Zobrist keys (initialized at startup)
extern std::array<std::array<Key, SQUARE_NB>, PIECE_NB> ZobristPiece;
extern std::array<Key, FILE_NB> ZobristEnpassant;
extern std::array<Key, CASTLING_RIGHT_NB> ZobristCastling;
extern Key ZobristSide;
extern Key ZobristNoPawns;

void init_zobrist();

} // namespace Nexus
