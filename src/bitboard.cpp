#include "bitboard.h"
#include <cstring>

namespace Nexus {

u64 SquareBB[SQUARE_NB];
u64 LineBB[SQUARE_NB][SQUARE_NB];
u64 BetweenBB[SQUARE_NB][SQUARE_NB];
u64 PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
u64 PawnAttacks[COLOR_NB][SQUARE_NB];

u64 DistanceRingBB[SQUARE_NB][8];
int SquareDistance[SQUARE_NB][SQUARE_NB];

// Magic bitboard tables
u64 RookTable[0x19000];  // 102400 entries
u64 BishopTable[0x1480]; // 5248 entries

Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

namespace {

// Rook and bishop occupancy masks (excluding edges)
constexpr u64 RookMasks[SQUARE_NB] = {
    0x000101010101017EULL, 0x000202020202027CULL, 0x000404040404047AULL, 0x0008080808080876ULL,
    0x001010101010106EULL, 0x002020202020205EULL, 0x004040404040403EULL, 0x008080808080807EULL,
    0x0001010101017E00ULL, 0x0002020202027C00ULL, 0x0004040404047A00ULL, 0x0008080808087600ULL,
    0x0010101010106E00ULL, 0x0020202020205E00ULL, 0x0040404040403E00ULL, 0x0080808080807E00ULL,
    0x00010101017E0000ULL, 0x00020202027C0000ULL, 0x00040404047A0000ULL, 0x0008080808760000ULL,
    0x00101010106E0000ULL, 0x00202020205E0000ULL, 0x00404040403E0000ULL, 0x00808080807E0000ULL,
    0x000101017E000000ULL, 0x000202027C000000ULL, 0x000404047A000000ULL, 0x0008080876000000ULL,
    0x001010106E000000ULL, 0x002020205E000000ULL, 0x004040403E000000ULL, 0x008080807E000000ULL,
    0x0001017E00000000ULL, 0x0002027C00000000ULL, 0x0004047A00000000ULL, 0x0008087600000000ULL,
    0x0010106E00000000ULL, 0x0020205E00000000ULL, 0x0040403E00000000ULL, 0x0080807E00000000ULL,
    0x00017E0000000000ULL, 0x00027C0000000000ULL, 0x00047A0000000000ULL, 0x0008760000000000ULL,
    0x00106E0000000000ULL, 0x00205E0000000000ULL, 0x00403E0000000000ULL, 0x00807E0000000000ULL,
    0x007E000000000000ULL, 0x007C000000000000ULL, 0x007A000000000000ULL, 0x0076000000000000ULL,
    0x006E000000000000ULL, 0x005E000000000000ULL, 0x003E000000000000ULL, 0x007E000000000000ULL
};

constexpr u64 BishopMasks[SQUARE_NB] = {
    0x0040201008040200ULL, 0x0000402010080400ULL, 0x0000004020100A00ULL, 0x0000000040221400ULL,
    0x0000000002442800ULL, 0x0000000204885000ULL, 0x000002040A102000ULL, 0x0004081020400000ULL,
    0x0020100804020000ULL, 0x0040201008040000ULL, 0x00004020100A0000ULL, 0x0000004022140000ULL,
    0x0000000244280000ULL, 0x0000020488500000ULL, 0x0002040A10200000ULL, 0x0004081020400000ULL,
    0x0010080402000200ULL, 0x0020100804000400ULL, 0x004020100A000A00ULL, 0x0000402214001400ULL,
    0x0000242800002800ULL, 0x0002048800005000ULL, 0x00040A1000002000ULL, 0x0008102000004000ULL,
    0x0008040200020400ULL, 0x0010080400040800ULL, 0x0020100A000A1000ULL, 0x0040221400142200ULL,
    0x0000280000280000ULL, 0x0004880000500000ULL, 0x000A100000200000ULL, 0x0010200000400000ULL,
    0x0004020002040800ULL, 0x0008040004081000ULL, 0x00100A000A102000ULL, 0x0022140014224000ULL,
    0x0042800028000000ULL, 0x0088000050000000ULL, 0x0010000020000000ULL, 0x0020000040000000ULL,
    0x0002000204081000ULL, 0x0004000408102000ULL, 0x000A000A10204000ULL, 0x0014001422400000ULL,
    0x0028002800000000ULL, 0x0050005000000000ULL, 0x0020002000000000ULL, 0x0040004000000000ULL,
    0x0000020408102000ULL, 0x0000040810204000ULL, 0x00000A1020400000ULL, 0x0000142240000000ULL,
    0x0000280000000000ULL, 0x0000500000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL
};

constexpr u64 RookMagics_const[SQUARE_NB] = {
    0x0080001020280080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
    0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080008080004000ULL,
    0x0080200200401000ULL, 0x0040100100200800ULL, 0x0080080100100400ULL, 0x0080040800200200ULL,
    0x0080020400400100ULL, 0x0080010200800080ULL, 0x0080008002000400ULL, 0x0080008080800200ULL,
    0x0080200010008020ULL, 0x0040100008005010ULL, 0x0080080004003010ULL, 0x0080040002005010ULL,
    0x0080020001002010ULL, 0x0080010000801810ULL, 0x0080008001000810ULL, 0x0080008080004020ULL,
    0x0080200010002080ULL, 0x0040100008001008ULL, 0x0080080004000808ULL, 0x0080040002000408ULL,
    0x0080020001000208ULL, 0x0080010000800308ULL, 0x0080008001000204ULL, 0x0080008080004020ULL,
    0x0080200010002080ULL, 0x0040100008001008ULL, 0x0080080004000808ULL, 0x0080040002000408ULL,
    0x0080020001000208ULL, 0x0080010000800308ULL, 0x0080008001000204ULL, 0x0080008080004020ULL,
    0x0080200010002080ULL, 0x0040100008001008ULL, 0x0080080004000808ULL, 0x0080040002000408ULL,
    0x0080020001000208ULL, 0x0080010000800308ULL, 0x0080008001000204ULL, 0x0080008080004020ULL,
    0x0080200010002080ULL, 0x0040100008001008ULL, 0x0080080004000808ULL, 0x0080040002000408ULL,
    0x0080020001000208ULL, 0x0080010000800308ULL, 0x0080008001000204ULL, 0x0080008080004020ULL,
    0x0080200010002080ULL, 0x0040100008001008ULL, 0x0080080004000808ULL, 0x0080040002000408ULL,
    0x0080020001000208ULL, 0x0080010000800308ULL, 0x0080008001000204ULL, 0x0080008080004020ULL
};

constexpr u64 BishopMagics_const[SQUARE_NB] = {
    0x0040201008040200ULL, 0x0000402010080400ULL, 0x0000004020100A00ULL, 0x0000000040221400ULL,
    0x0000000002442800ULL, 0x0000000204885000ULL, 0x000002040A102000ULL, 0x0004081020400000ULL,
    0x0020100804020000ULL, 0x0040201008040000ULL, 0x00004020100A0000ULL, 0x0000004022140000ULL,
    0x0000000244280000ULL, 0x0000020488500000ULL, 0x0002040A10200000ULL, 0x0004081020400000ULL,
    0x0010080402000200ULL, 0x0020100804000400ULL, 0x004020100A000A00ULL, 0x0000402214001400ULL,
    0x0000242800002800ULL, 0x0002048800005000ULL, 0x00040A1000002000ULL, 0x0008102000004000ULL,
    0x0008040200020400ULL, 0x0010080400040800ULL, 0x0020100A000A1000ULL, 0x0040221400142200ULL,
    0x0000280000280000ULL, 0x0004880000500000ULL, 0x000A100000200000ULL, 0x0010200000400000ULL,
    0x0004020002040800ULL, 0x0008040004081000ULL, 0x00100A000A102000ULL, 0x0022140014224000ULL,
    0x0042800028000000ULL, 0x0088000050000000ULL, 0x0010000020000000ULL, 0x0020000040000000ULL,
    0x0002000204081000ULL, 0x0004000408102000ULL, 0x000A000A10204000ULL, 0x0014001422400000ULL,
    0x0028002800000000ULL, 0x0050005000000000ULL, 0x0020002000000000ULL, 0x0040004000000000ULL,
    0x0000020408102000ULL, 0x0000040810204000ULL, 0x00000A1020400000ULL, 0x0000142240000000ULL,
    0x0000280000000000ULL, 0x0000500000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL
};

constexpr unsigned RookShifts[SQUARE_NB] = {
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
};

constexpr unsigned BishopShifts[SQUARE_NB] = {
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

u64 sliding_attack(PieceType pt, Square sq, u64 occupied) {
    u64 attacks = 0;
    const Direction directions[2][4] = {
        { NORTH, SOUTH, EAST, WEST },
        { NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST }
    };
    int idx = (pt == BISHOP) ? 1 : 0;
    for (int i = 0; i < 4; ++i) {
        Direction d = directions[idx][i];
        Square s = sq + d;
        while (is_ok(s) && square_distance(s, s - d) == 1) {
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
            s += d;
        }
    }
    return attacks;
}

// Generate all subsets of a mask (for magic initialization)
u64 index_to_u64(int index, int bits, u64 mask) {
    u64 result = 0;
    for (int i = 0; i < bits; ++i) {
        int j = pop_lsb(mask);
        if (index & (1 << i))
            result |= (1ULL << j);
    }
    return result;
}

void init_magics(Magic magics[SQUARE_NB], u64 table[], u64 masks[], u64 magics_const[], unsigned shifts[], Square deltas[]) {
    int count = 0;
    for (Square s = SQ_A1; s <= SQ_H8; s = Square(s + 1)) {
        int shift = 64 - shifts[s];
        int bits = count_bits(masks[s]);
        magics[s].mask = masks[s];
        magics[s].magic = magics_const[s];
        magics[s].shift = shifts[s];
        magics[s].attacks = &table[count];

        // Enumerate all occupancy subsets and fill attack table
        for (int i = 0; i < (1 << bits); ++i) {
            u64 occupied = 0;
            u64 m = masks[s];
            for (int j = 0; j < bits; ++j) {
                Square sq = pop_lsb(m);
                if (i & (1 << j))
                    occupied |= (1ULL << sq);
            }
            u64 idx = ((occupied * magics_const[s]) >> shifts[s]);
            magics[s].attacks[idx] = sliding_attack(deltas[0] == NORTH ? ROOK : BISHOP, s, occupied);
        }
        count += (1 << bits);
    }
}

void init_magics_bb() {
    Square rook_deltas[4] = { NORTH, EAST, SOUTH, WEST };
    Square bishop_deltas[4] = { NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST };

    init_magics(RookMagics, RookTable, (u64*)RookMasks, (u64*)RookMagics_const, (unsigned*)RookShifts, rook_deltas);
    init_magics(BishopMagics, BishopTable, (u64*)BishopMasks, (u64*)BishopMagics_const, (unsigned*)BishopShifts, bishop_deltas);
}

} // anonymous namespace

void init_bitboards() {
    for (Square s = SQ_A1; s <= SQ_H8; s = Square(s + 1))
        SquareBB[s] = 1ULL << s;

    // Line and Between tables
    for (Square s1 = SQ_A1; s1 <= SQ_H8; s1 = Square(s1 + 1)) {
        for (Square s2 = SQ_A1; s2 <= SQ_H8; s2 = Square(s2 + 1)) {
            if (s1 == s2) continue;
            LineBB[s1][s2] = 0;
            BetweenBB[s1][s2] = 0;

            // Same rank
            if (rank_of(s1) == rank_of(s2)) {
                u64 line = rank_bb(rank_of(s1));
                LineBB[s1][s2] = line;
                if (s1 < s2)
                    BetweenBB[s1][s2] = line & ((1ULL << s2) - 1) & ~((1ULL << (s1 + 1)) - 1);
                else
                    BetweenBB[s1][s2] = line & ((1ULL << (s1)) - 1) & ~((1ULL << (s2 + 1)) - 1);
            }
            // Same file
            else if (file_of(s1) == file_of(s2)) {
                u64 line = file_bb(file_of(s1));
                LineBB[s1][s2] = line;
                if (s1 < s2)
                    BetweenBB[s1][s2] = line & ((1ULL << s2) - 1) & ~((1ULL << (s1 + 1)) - 1);
                else
                    BetweenBB[s1][s2] = line & ((1ULL << s1) - 1) & ~((1ULL << (s2 + 1)) - 1);
            }
            // Same diagonal (a1-h8)
            else if ((rank_of(s1) - rank_of(s2)) == (file_of(s1) - file_of(s2))) {
                // Build diagonal bitboard
                u64 diag = 0;
                int df = file_of(s1) - file_of(s2);
                int dr = rank_of(s1) - rank_of(s2);
                if (df == dr) {
                    for (int i = 0; i < 8; ++i) {
                        int f = file_of(s1) - i;
                        int r = rank_of(s1) - i;
                        if (f >= 0 && f < 8 && r >= 0 && r < 8)
                            diag |= square_bb(make_square(File(f), Rank(r)));
                    }
                } else {
                    for (int i = 0; i < 8; ++i) {
                        int f = file_of(s1) + i;
                        int r = rank_of(s1) - i;
                        if (f >= 0 && f < 8 && r >= 0 && r < 8)
                            diag |= square_bb(make_square(File(f), Rank(r)));
                    }
                }
                LineBB[s1][s2] = diag;
                // Between
                if (s1 < s2)
                    BetweenBB[s1][s2] = diag & ((1ULL << s2) - 1) & ~((1ULL << (s1 + 1)) - 1);
                else
                    BetweenBB[s1][s2] = diag & ((1ULL << s1) - 1) & ~((1ULL << (s2 + 1)) - 1);
            }
        }
    }

    // Pseudo-attacks
    for (Square s = SQ_A1; s <= SQ_H8; s = Square(s + 1)) {
        // Knight
        int r = rank_of(s), f = file_of(s);
        const int dr[8] = { -2,-2,-1,-1, 1, 1, 2, 2 };
        const int df[8] = { -1, 1,-2, 2,-2, 2,-1, 1 };
        for (int i = 0; i < 8; ++i) {
            int nr = r + dr[i], nf = f + df[i];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                PseudoAttacks[KNIGHT][s] |= square_bb(make_square(File(nf), Rank(nr)));
        }
        // King
        for (int dr2 = -1; dr2 <= 1; ++dr2)
            for (int df2 = -1; df2 <= 1; ++df2)
                if (dr2 != 0 || df2 != 0) {
                    int nr = r + dr2, nf = f + df2;
                    if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                        PseudoAttacks[KING][s] |= square_bb(make_square(File(nf), Rank(nr)));
                }

        // Pawn attacks
        int r2 = r + 1;
        if (r2 < 8) {
            if (f > 0) PawnAttacks[WHITE][s] |= square_bb(make_square(File(f - 1), Rank(r2)));
            if (f < 7) PawnAttacks[WHITE][s] |= square_bb(make_square(File(f + 1), Rank(r2)));
        }
        r2 = r - 1;
        if (r2 >= 0) {
            if (f > 0) PawnAttacks[BLACK][s] |= square_bb(make_square(File(f - 1), Rank(r2)));
            if (f < 7) PawnAttacks[BLACK][s] |= square_bb(make_square(File(f + 1), Rank(r2)));
        }
    }

    // Distance tables
    for (Square s1 = SQ_A1; s1 <= SQ_H8; s1 = Square(s1 + 1)) {
        for (Square s2 = SQ_A1; s2 <= SQ_H8; s2 = Square(s2 + 1)) {
            SquareDistance[s1][s2] = std::max(
                std::abs(file_of(s1) - file_of(s2)),
                std::abs(rank_of(s1) - rank_of(s2))
            );
            if (s1 != s2) {
                int d = SquareDistance[s1][s2];
                DistanceRingBB[s1][d - 1] |= square_bb(s2);
            }
        }
    }

    init_magics_bb();
}

} // namespace Nexus
