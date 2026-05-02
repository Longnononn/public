#include <iostream>
#include <cassert>
#include "bitboard.h"
#include "board.h"
#include "movegen.h"

using namespace Nexus;

int test_bitboard_main();

int main() {
    init_bitboards();
    init_zobrist();
    
    // Test bitboards
    test_bitboard_main();
    
    // Test perft from starting position
    BoardState pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    assert(perft(pos, 0) == 1);
    assert(perft(pos, 1) == 20);
    assert(perft(pos, 2) == 400);
    assert(perft(pos, 3) == 8902);
    // Perft(4) = 197281, Perft(5) = 4865609, Perft(6) = 119060324
    // assert(perft(pos, 4) == 197281); // Slow in debug
    
    // Test position 2
    pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -");
    assert(perft(pos, 1) == 48);
    assert(perft(pos, 2) == 2039);
    
    // Test position 3
    pos.set_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -");
    assert(perft(pos, 1) == 14);
    assert(perft(pos, 2) == 191);
    assert(perft(pos, 3) == 2812);
    
    // Test position 4 (ep)
    pos.set_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
    assert(perft(pos, 1) == 6);
    assert(perft(pos, 2) == 264);
    
    // Test position 5
    pos.set_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    assert(perft(pos, 1) == 44);
    assert(perft(pos, 2) == 1486);
    
    std::cout << "All perft tests passed" << std::endl;
    return 0;
}
