#pragma once

#include <cstdint>
#include <cassert>
#include <string>

namespace Nexus {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

enum Color : i32 { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : i32 {
    NO_PIECE_TYPE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3,
    ROOK = 4, QUEEN = 5, KING = 6, ALL_PIECES = 7,
    PIECE_TYPE_NB = 8
};

enum Piece : i32 {
    NO_PIECE, W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

constexpr Piece make_piece(Color c, PieceType pt) {
    return Piece((c << 3) + pt);
}
constexpr Color color_of(Piece pc) {
    return Color(pc >> 3);
}
constexpr PieceType type_of(Piece pc) {
    return PieceType(pc & 7);
}

enum File : i32 { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : i32 { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };

enum Square : i32 {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQUARE_NB = 64, SQ_ZERO = 0
};

constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr Square make_square(File f, Rank r) { return Square((r << 3) | f); }
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }
constexpr Rank relative_rank(Color c, Rank r) { return Rank(r ^ (c * 7)); }
constexpr bool same_file(Square a, Square b) { return file_of(a) == file_of(b); }
constexpr bool same_rank(Square a, Square b) { return rank_of(a) == rank_of(b); }
constexpr int square_distance(Square a, Square b) {
    return std::max(std::abs(file_of(a) - file_of(b)), std::abs(rank_of(a) - rank_of(b)));
}
constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

enum CastlingRights : i32 {
    NO_CASTLING,
    WHITE_OO  = 1,
    WHITE_OOO = 2,
    BLACK_OO  = 4,
    BLACK_OOO = 8,
    ANY_CASTLING = 15,
    CASTLING_RIGHT_NB = 16
};

enum Depth : i32 {
    DEPTH_ZERO = 0,
    DEPTH_MAX = 100
};

constexpr CastlingRights operator|(CastlingRights a, CastlingRights b) { return CastlingRights((int)a | (int)b); }
constexpr CastlingRights operator&(CastlingRights a, CastlingRights b) { return CastlingRights((int)a & (int)b); }
constexpr CastlingRights operator~(CastlingRights a) { return CastlingRights(~(int)a); }
inline CastlingRights& operator|=(CastlingRights& a, CastlingRights b) { a = a | b; return a; }
inline CastlingRights& operator&=(CastlingRights& a, CastlingRights b) { a = a & b; return a; }

enum Direction : i32 {
    NORTH = 8, EAST = 1, SOUTH = -8, WEST = -1,
    NORTH_EAST = 9, NORTH_WEST = 7,
    SOUTH_EAST = -7, SOUTH_WEST = -9
};

constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }
constexpr Direction pawn_capture_east(Color c) { return c == WHITE ? NORTH_EAST : SOUTH_EAST; }
constexpr Direction pawn_capture_west(Color c) { return c == WHITE ? NORTH_WEST : SOUTH_WEST; }

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { s = s + d; return s; }
constexpr Square& operator-=(Square& s, Direction d) { s = s - d; return s; }

enum Value : i32 {
    VALUE_ZERO = 0,
    VALUE_DRAW = 0,
    VALUE_KNOWN_WIN = 10000,
    VALUE_MATE = 32000,
    VALUE_INFINITE = 32001,
    VALUE_NONE = 32002,
    VALUE_MATE_IN_MAX_PLY = VALUE_MATE - 256,
    VALUE_MATED_IN_MAX_PLY = -VALUE_MATE + 256,
    VALUE_TB_WIN_IN_MAX_PLY = VALUE_MATE_IN_MAX_PLY - 1000,
    VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY,
    VALUE_TB_WIN = VALUE_MATE_IN_MAX_PLY - 2,
    PawnValueMg = 126, PawnValueEg = 208,
    KnightValueMg = 781, KnightValueEg = 854,
    BishopValueMg = 825, BishopValueEg = 915,
    RookValueMg = 1276, RookValueEg = 1380,
    QueenValueMg = 2538, QueenValueEg = 2682,
    MidgameLimit = 15258, EndgameLimit = 3915
};

constexpr Value operator+(Value a, Value b) { return Value((int)a + (int)b); }
constexpr Value operator-(Value a, Value b) { return Value((int)a - (int)b); }
constexpr Value operator-(Value a) { return Value(-(int)a); }
constexpr Value operator+(Value a, int b) { return Value((int)a + b); }
constexpr Value operator-(Value a, int b) { return Value((int)a - b); }
constexpr Value operator+(int a, Value b) { return Value(a + (int)b); }
constexpr Value operator*(Value a, int b) { return Value((int)a * b); }
constexpr Value operator/(Value a, int b) { return Value((int)a / b); }
inline Value& operator+=(Value& a, Value b) { a = a + b; return a; }
inline Value& operator-=(Value& a, Value b) { a = a - b; return a; }

constexpr int piece_value[PIECE_TYPE_NB] = {
    0, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg, 0, 0
};

enum Bound : u8 { BOUND_NONE, BOUND_UPPER, BOUND_LOWER, BOUND_EXACT };
enum MoveType : u16 {
    NORMAL,
    PROMOTION = 1 << 14,
    ENPASSANT = 2 << 14,
    CASTLING  = 3 << 14
};

enum Key : u64 {};
constexpr Key make_key(u64 k) { return Key(k); }
constexpr Key operator^(Key a, Key b) { return Key((u64)a ^ (u64)b); }
inline Key& operator^=(Key& a, Key b) { a = a ^ b; return a; }

constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY   = 246;

inline Value mate_in(int ply) { return Value(int(VALUE_MATE) - ply); }
inline Value mated_in(int ply) { return Value(-int(VALUE_MATE) + ply); }

constexpr u64 FileABB = 0x0101010101010101ULL;
constexpr u64 FileBBB = FileABB << 1;
constexpr u64 FileCBB = FileABB << 2;
constexpr u64 FileDBB = FileABB << 3;
constexpr u64 FileEBB = FileABB << 4;
constexpr u64 FileFBB = FileABB << 5;
constexpr u64 FileGBB = FileABB << 6;
constexpr u64 FileHBB = FileABB << 7;
constexpr u64 Rank1BB = 0xFFULL;
constexpr u64 Rank2BB = Rank1BB << (8 * 1);
constexpr u64 Rank3BB = Rank1BB << (8 * 2);
constexpr u64 Rank4BB = Rank1BB << (8 * 3);
constexpr u64 Rank5BB = Rank1BB << (8 * 4);
constexpr u64 Rank6BB = Rank1BB << (8 * 5);
constexpr u64 Rank7BB = Rank1BB << (8 * 6);
constexpr u64 Rank8BB = Rank1BB << (8 * 7);

} // namespace Nexus
