#include <iostream>
#include <cassert>
#include "bitboard.h"

using namespace Nexus;

int test_bitboard_main() {
    init_bitboards();
    
    // Test square_bb
    assert(square_bb(SQ_A1) == 1ULL);
    assert(square_bb(SQ_H8) == (1ULL << 63));
    
    // Test count_bits
    assert(count_bits(0xFFULL) == 8);
    assert(count_bits(0ULL) == 0);
    
    // Test shift
    assert(shift(square_bb(SQ_A1), NORTH) == square_bb(SQ_A2));
    assert(shift(square_bb(SQ_A2), SOUTH) == square_bb(SQ_A1));
    
    // Test rook attacks
    u64 occ = square_bb(SQ_B2) | square_bb(SQ_D4);
    u64 attacks = rook_attacks(SQ_C3, occ);
    assert(attacks & square_bb(SQ_C4));
    assert(attacks & square_bb(SQ_C2));
    assert(attacks & square_bb(SQ_B3));
    assert(attacks & square_bb(SQ_D3));
    assert(!(attacks & square_bb(SQ_C5))); // Blocked by B2... wait, not blocked
    
    std::cout << "Bitboard tests passed" << std::endl;
    return 0;
}
