#pragma once

#include "types.h"
#include "board.h"
#include "move.h"

namespace Nexus {

enum GenType {
    CAPTURES,
    QUIETS,
    QUIET_CHECKS,
    EVASIONS,
    NON_EVASIONS,
    LEGAL
};

template<GenType Type>
ExtMove* generate(const BoardState& pos, ExtMove* moveList);

// Helper to check if a move is legal (used by movegen LEGAL)
bool is_legal(const BoardState& pos, Move m);

// Perft for testing
u64 perft(BoardState& pos, int depth);

} // namespace Nexus
