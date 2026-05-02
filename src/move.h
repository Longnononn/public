#pragma once

#include "types.h"
#include <string>

namespace Nexus {

class Move {
    u16 m;
public:
    constexpr Move() : m(0) {}
    constexpr explicit Move(u16 move) : m(move) {}
    constexpr Move(Square from, Square to, MoveType type = NORMAL) 
        : m(from | (to << 6) | type) {}
    constexpr Move(Square from, Square to, PieceType pt) 
        : m(from | (to << 6) | PROMOTION | ((pt - KNIGHT) << 12)) {}

    constexpr Square from_sq() const { return Square(m & 0x3F); }
    constexpr Square to_sq() const { return Square((m >> 6) & 0x3F); }
    constexpr int from_to() const { return m & 0xFFF; }
    
    constexpr PieceType promotion_type() const { 
        return PieceType(((m >> 12) & 3) + KNIGHT); 
    }
    
    constexpr MoveType type_of() const { return MoveType(m & (3 << 14)); }
    constexpr bool is_special() const { return m & (3 << 14); }
    constexpr bool is_promotion() const { return (m & PROMOTION) == PROMOTION; }
    constexpr bool is_enpassant() const { return (m & ENPASSANT) == ENPASSANT; }
    constexpr bool is_castling() const { return (m & CASTLING) == CASTLING; }
    
    constexpr bool operator==(const Move& o) const { return m == o.m; }
    constexpr bool operator!=(const Move& o) const { return m != o.m; }
    constexpr operator u16() const { return m; }
    constexpr bool is_ok() const { return from_sq() != to_sq(); }
    
    static constexpr Move none() { return Move(0); }
    static constexpr Move null() { return Move(0xFFFF); }
    
    constexpr bool is_null() const { return m == 0xFFFF; }
    
    std::string to_uci() const;
    static Move from_uci(const std::string& str, const struct BoardState* pos = nullptr);
};

constexpr Move make_move(Square from, Square to) { return Move(from, to, NORMAL); }
constexpr Move make_enpassant(Square from, Square to) { return Move(from, to, ENPASSANT); }
constexpr Move make_castling(Square from, Square to) { return Move(from, to, CASTLING); }
constexpr Move make_promotion(Square from, Square to, PieceType pt) { return Move(from, to, pt); }

struct ExtMove {
    Move move;
    int value;
    
    operator Move() const { return move; }
    bool operator<(const ExtMove& o) const { return value > o.value; } // descending
};

constexpr bool is_ok(Move m) { return m.is_ok(); }

} // namespace Nexus
