#include "syzygy.h"
#include <iostream>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

namespace Syzygy {

Tablebases TB;

Tablebases::Tablebases() = default;
Tablebases::~Tablebases() { close(); }

bool Tablebases::init(const std::string& path) {
    if (path.empty()) return false;
    
    tbPath = path;
    
    // Check if directory exists and contains .rtbw/.rtbz files
    #ifdef _WIN32
    WIN32_FIND_DATA findData;
    HANDLE hFind;
    
    std::string searchPath = path + "\\*.rtbw";
    hFind = FindFirstFile(searchPath.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        int count = 0;
        do {
            count++;
        } while (FindNextFile(hFind, &findData));
        FindClose(hFind);
        
        if (count > 0) {
            initialized = true;
            maxPieces = 5;  // Detect from files
            std::cout << "Syzygy: Found " << count << " WDL tablebases in " << path << std::endl;
            
            // Check for 6-piece tables
            searchPath = path + "\\*.rtbz";
            hFind = FindFirstFile(searchPath.c_str(), &findData);
            if (hFind != INVALID_HANDLE_VALUE) {
                int dtzCount = 0;
                do {
                    dtzCount++;
                } while (FindNextFile(hFind, &findData));
                FindClose(hFind);
                std::cout << "Syzygy: Found " << dtzCount << " DTZ tablebases" << std::endl;
            }
            
            return true;
        }
    }
    #else
    // Linux implementation
    DIR* dir = opendir(path.c_str());
    if (dir) {
        int count = 0;
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name.size() > 5 && name.substr(name.size() - 5) == ".rtbw") {
                count++;
            }
        }
        closedir(dir);
        
        if (count > 0) {
            initialized = true;
            maxPieces = 5;
            std::cout << "Syzygy: Found " << count << " WDL tablebases" << std::endl;
            return true;
        }
    }
    #endif
    
    std::cerr << "Syzygy: No tablebases found in " << path << std::endl;
    return false;
}

void Tablebases::close() {
    initialized = false;
    maxPieces = 0;
    tbPath.clear();
}

bool Tablebases::is_tb_position(const BoardState& pos) const {
    if (!initialized) return false;
    
    // Count pieces
    int pieceCount = 0;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        if (pos.piece_on(sq) != NO_PIECE) {
            pieceCount++;
        }
    }
    
    return pieceCount <= maxPieces;
}

WDLResult Tablebases::probe_wdl(const BoardState& pos) const {
    if (!initialized || !is_tb_position(pos)) {
        return WDL_FAIL;
    }
    
    // Placeholder - actual implementation requires Fathom library integration
    // or custom tablebase probing code
    
    // Simplified: return draw for positions with few pieces
    // In real implementation, this would probe the .rtbw files
    
    return WDL_FAIL;  // Not implemented - requires TB files
}

int Tablebases::probe_dtz(const BoardState& pos) const {
    if (!initialized || !is_tb_position(pos)) {
        return -1;  // Fail
    }
    
    // Placeholder - requires .rtbz file probing
    return -1;
}

bool Tablebases::root_probe(const BoardState& pos, Move& bestMove, Value& bestScore) const {
    if (!initialized || !is_tb_position(pos)) {
        return false;
    }
    
    // WDL probe all root moves
    WDLResult bestWdl = WDL_FAIL;
    bestScore = VALUE_NONE;
    
    // Generate all legal moves and probe each
    // This is simplified - actual implementation would:
    // 1. Generate all legal moves
    // 2. For each move, make move and probe WDL
    // 3. Choose move with best WDL
    // 4. If winning, use DTZ to find shortest path
    
    return false;  // Placeholder
}

bool Tablebases::root_probe_wdl(const BoardState& pos, Move& bestMove, Value& bestScore) const {
    return root_probe(pos, bestMove, bestScore);
}

void Tablebases::filter_root_moves(const BoardState& pos, std::vector<Move>& moves) const {
    if (!initialized || !is_tb_position(pos)) return;
    
    // Filter moves that lead to known losses
    // Keep only moves that lead to draws or wins
    
    std::vector<Move> filtered;
    for (const Move& m : moves) {
        // Make move, probe, undo
        // If not loss, keep
        filtered.push_back(m);
    }
    
    moves = std::move(filtered);
}

int Tablebases::probe_table(const BoardState& pos, bool wdl) const {
    (void)pos;
    (void)wdl;
    // Internal probing - would use Fathom or custom decoder
    return -1;
}

int Tablebases::map_score(WDLResult wdl, int dtz) const {
    (void)dtz;
    // Map WDL + DTZ to search score
    switch (wdl) {
        case WDL_WIN: return 32000;
        case WDL_CURSED_WIN: return 100;
        case WDL_DRAW: return 0;
        case WDL_BLESSED_LOSS: return -100;
        case WDL_LOSS: return -32000;
        default: return 0;
    }
}

} // namespace Syzygy
