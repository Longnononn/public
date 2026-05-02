#include "see.h"
#include <algorithm>

namespace Nexus {

// Get all attacks to a square considering current occupied squares
u64 SEE::attacks_to(const BoardState& pos, Square sq, u64 occupied) {
    u64 attacks = 0;
    
    // Pawn attacks
    attacks |= pawn_attacks(WHITE, sq) & pos.pieces(BLACK, PAWN);
    attacks |= pawn_attacks(BLACK, sq) & pos.pieces(WHITE, PAWN);
    
    // Knight attacks
    attacks |= knight_attacks(sq) & (pos.pieces(WHITE, KNIGHT) | pos.pieces(BLACK, KNIGHT));
    
    // Bishop attacks
    u64 bishops = pos.pieces(WHITE, BISHOP) | pos.pieces(BLACK, BISHOP);
    u64 queens = pos.pieces(WHITE, QUEEN) | pos.pieces(BLACK, QUEEN);
    attacks |= bishop_attacks(sq, occupied) & (bishops | queens);
    
    // Rook attacks
    u64 rooks = pos.pieces(WHITE, ROOK) | pos.pieces(BLACK, ROOK);
    attacks |= rook_attacks(sq, occupied) & (rooks | queens);
    
    // King attacks
    attacks |= king_attacks(sq) & (pos.pieces(WHITE, KING) | pos.pieces(BLACK, KING));
    
    return attacks;
}

// Find least valuable attacker
PieceType SEE::get_lva(const BoardState& pos, Square sq, Color c, u64& occupied) {
    u64 attackers = attacks_to(pos, sq, occupied) & pos.pieces(c);
    
    if (!attackers) return NO_PIECE_TYPE;
    
    // Check in order of increasing value: Pawn, Knight, Bishop, Rook, Queen, King
    u64 pawns = attackers & pos.pieces(PAWN);
    if (pawns) {
        Square from = lsb(pawns);
        occupied &= ~square_bb(from);
        return PAWN;
    }
    
    u64 knights = attackers & pos.pieces(KNIGHT);
    if (knights) {
        Square from = lsb(knights);
        occupied &= ~square_bb(from);
        return KNIGHT;
    }
    
    u64 bishops = attackers & pos.pieces(BISHOP);
    if (bishops) {
        Square from = lsb(bishops);
        occupied &= ~square_bb(from);
        return BISHOP;
    }
    
    u64 rooks = attackers & pos.pieces(ROOK);
    if (rooks) {
        Square from = lsb(rooks);
        occupied &= ~square_bb(from);
        return ROOK;
    }
    
    u64 queens = attackers & pos.pieces(QUEEN);
    if (queens) {
        Square from = lsb(queens);
        occupied &= ~square_bb(from);
        return QUEEN;
    }
    
    u64 kings = attackers & pos.pieces(KING);
    if (kings) {
        Square from = lsb(kings);
        occupied &= ~square_bb(from);
        return KING;
    }
    
    return NO_PIECE_TYPE;
}

int SEE::see_value(const BoardState& pos, Move capture) {
    if (!pos.is_capture(capture))
        return 0;
    
    Square from = capture.from_sq();
    Square to = capture.to_sq();
    Color us = pos.sideToMove;
    
    // Get captured piece value
    Piece captured = pos.piece_on(to);
    int capturedValue = PieceValues[type_of(captured)];
    
    // If en passant, captured piece is a pawn
    if (capture.is_enpassant()) {
        capturedValue = PieceValues[PAWN];
    }
    
    // Make the capture on a copy of occupied squares
    u64 occupied = pos.all_pieces();
    occupied &= ~square_bb(from);
    if (capture.is_enpassant()) {
        // Remove the captured pawn
        Square epSquare = capture.to_sq() + (us == WHITE ? SOUTH : NORTH);
        occupied &= ~square_bb(epSquare);
    }
    occupied |= square_bb(to);
    
    // Start the swap sequence
    int balance = capturedValue;
    PieceType capturingPiece = type_of(pos.piece_on(from));
    
    Color current = ~us;
    
    // Swap off loop
    while (true) {
        // Find opponent's least valuable attacker
        PieceType lva = get_lva(pos, to, current, occupied);
        if (lva == NO_PIECE_TYPE)
            break;
        
        // Opponent captures back
        balance -= PieceValues[capturingPiece];
        
        // If balance goes negative and we're the side to move, this is losing
        if (balance < 0 && current == ~us)
            break;
        
        // Our turn to recapture
        capturingPiece = lva;
        
        // Check if we have any recaptures
        PieceType ourLva = get_lva(pos, to, us, occupied);
        if (ourLva == NO_PIECE_TYPE) {
            // No more recaptures
            if (balance < 0)
                balance = -balance;
            break;
        }
        
        // We recapture
        balance += PieceValues[lva];
        capturingPiece = ourLva;
        
        // If balance is winning, continue to see if opponent can recapture
        if (balance > 0 && balance > PieceValues[ourLva])
            break;
    }
    
    return balance;
}

bool SEE::is_capture_winning(const BoardState& pos, Move capture) {
    return see_ge(pos, capture, 0);
}

bool SEE::see_ge(const BoardState& pos, Move capture, int threshold) {
    if (!pos.is_capture(capture))
        return threshold <= 0;
    
    Square from = capture.from_sq();
    Square to = capture.to_sq();
    Color us = pos.sideToMove;
    
    // Get captured piece value
    Piece captured = pos.piece_on(to);
    int capturedValue = PieceValues[type_of(captured)];
    
    if (capture.is_enpassant()) {
        capturedValue = PieceValues[PAWN];
    }
    
    // Start with the capture value
    int balance = capturedValue - threshold;
    
    // If we can't even meet threshold after first capture
    if (balance < 0)
        return false;
    
    // Value of piece we used to capture
    PieceType capturingPiece = type_of(pos.piece_on(from));
    balance -= PieceValues[capturingPiece];
    
    // If balance is now negative, we need to see if we can recapture
    if (balance < 0) {
        // Check if there are recaptures
        u64 occupied = pos.all_pieces();
        occupied &= ~square_bb(from);
        if (capture.is_enpassant()) {
            Square epSquare = to + (us == WHITE ? SOUTH : NORTH);
            occupied &= ~square_bb(epSquare);
        }
        occupied |= square_bb(to);
        
        Color current = ~us;
        
        while (true) {
            PieceType lva = get_lva(pos, to, current, occupied);
            if (lva == NO_PIECE_TYPE)
                return false; // No recapture, we lose the piece
            
            balance += PieceValues[capturingPiece];
            if (balance >= 0)
                return true; // We can recapture and be >= threshold
            
            capturingPiece = lva;
            balance -= PieceValues[lva];
            
            if (balance >= 0)
                return false;
            
            current = ~current;
        }
    }
    
    return balance >= 0;
}

} // namespace Nexus
