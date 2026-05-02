#include "move.h"
#include <sstream>
#include <cctype>

namespace Nexus {

std::string Move::to_uci() const {
    if (is_null()) return "0000";
    static const char* files = "abcdefgh";
    static const char* ranks = "12345678";
    std::string result;
    result += files[file_of(from_sq())];
    result += ranks[rank_of(from_sq())];
    result += files[file_of(to_sq())];
    result += ranks[rank_of(to_sq())];
    if (is_promotion()) {
        static const char* promos = "nbrq";
        result += promos[promotion_type() - KNIGHT];
    }
    return result;
}

Move Move::from_uci(const std::string& str, const struct BoardState*) {
    if (str == "0000") return null();
    
    auto file_from = str[0] - 'a';
    auto rank_from = str[1] - '1';
    auto file_to = str[2] - 'a';
    auto rank_to = str[3] - '1';
    
    Square from = make_square(File(file_from), Rank(rank_from));
    Square to = make_square(File(file_to), Rank(rank_to));
    
    if (str.length() == 5) {
        char p = std::tolower(str[4]);
        PieceType pt = (p == 'n') ? KNIGHT : (p == 'b') ? BISHOP : (p == 'r') ? ROOK : QUEEN;
        return make_promotion(from, to, pt);
    }
    
    return make_move(from, to);
}

} // namespace Nexus
