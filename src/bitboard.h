#pragma once

#include "types.h"

namespace Nexus {

// Bitboard utilities
constexpr u64 square_bb(Square s) { return 1ULL << s; }
constexpr bool more_than_one(u64 b) { return b && (b & (b - 1)); }

inline Square lsb(u64 b) {
    assert(b);
    return Square(__builtin_ctzll(b));
}

inline Square pop_lsb(u64& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline int count_bits(u64 b) {
    return __builtin_popcountll(b);
}

constexpr u64 shift(u64 b, Direction d) {
    return d == NORTH      ? b << 8
         : d == SOUTH      ? b >> 8
         : d == EAST       ? (b & ~FileHBB) << 1
         : d == WEST       ? (b & ~FileABB) >> 1
         : d == NORTH_EAST ? (b & ~FileHBB) << 9
         : d == NORTH_WEST ? (b & ~FileABB) << 7
         : d == SOUTH_EAST ? (b & ~FileHBB) >> 7
         : d == SOUTH_WEST ? (b & ~FileABB) >> 9
         : 0;
}

constexpr u64 file_bb(File f) { return FileABB << f; }
constexpr u64 rank_bb(Rank r) { return Rank1BB << (8 * r); }

extern u64 SquareBB[SQUARE_NB];
extern u64 LineBB[SQUARE_NB][SQUARE_NB];
extern u64 BetweenBB[SQUARE_NB][SQUARE_NB];
extern u64 PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
extern u64 PawnAttacks[COLOR_NB][SQUARE_NB];

// Magic bitboards for sliding pieces
struct Magic {
    u64* attacks;
    u64 mask;
    u64 magic;
    unsigned shift;

    u64 index(u64 occupied) const {
        return attacks[((occupied & mask) * magic) >> shift];
    }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

inline u64 rook_attacks(Square s, u64 occupied) { return RookMagics[s].index(occupied); }
inline u64 bishop_attacks(Square s, u64 occupied) { return BishopMagics[s].index(occupied); }
inline u64 queen_attacks(Square s, u64 occupied) { return rook_attacks(s, occupied) | bishop_attacks(s, occupied); }

inline u64 pawn_attacks(Color c, Square s) { return PawnAttacks[c][s]; }
inline u64 knight_attacks(Square s) { return PseudoAttacks[KNIGHT][s]; }
inline u64 king_attacks(Square s) { return PseudoAttacks[KING][s]; }

// Attack generation helpers
u64 sliding_attack(PieceType pt, Square sq, u64 occupied);

// Initialization
void init_bitboards();

// Precomputed tables
extern u64 DistanceRingBB[SQUARE_NB][8];
extern int SquareDistance[SQUARE_NB][SQUARE_NB];

} // namespace Nexus
