#include "datapipeline.h"
#include "board.h"
#include "training.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace DataPipeline {

// ================ PositionHash ================

PositionHash compute_hash(const std::string& fen) {
    // Simple FNV-1a hash for deduplication
    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    
    uint64_t hash = FNV_OFFSET_BASIS;
    for (char c : fen) {
        hash ^= static_cast<uint64_t>(c);
        hash *= FNV_PRIME;
    }
    
    return PositionHash{hash};
}

// ================ DataPipeline Implementation ================

DataPipeline::DataPipeline(uint32_t filters)
    : activeFilters(filters)
    , entropyThreshold(0.1)
    , scoreThreshold(Value(500))
    , minPly(10)
    , maxPly(300)
    , openingPly(10)
{
}

std::vector<FilteredPosition> DataPipeline::process(
    const std::vector<Training::TrainingPosition>& raw)
{
    std::cout << "Processing " << raw.size() << " positions...\n";
    
    // Convert to filtered format
    std::vector<FilteredPosition> data;
    data.reserve(raw.size());
    
    for (const auto& pos : raw) {
        FilteredPosition fp;
        fp.fen = pos.fen;
        fp.score = pos.score;
        fp.result = pos.result;
        fp.gamePhase = pos.gamePhase;
        fp.ply = pos.ply;
        fp.move50 = pos.move50;
        fp.inCheck = pos.inCheck;
        fp.gameId = pos.gameId;
        
        // Compute derived metrics
        fp.entropy = compute_entropy(fp);
        fp.isQuiet = !fp.inCheck && std::abs(fp.score) < VALUE_KNOWN_WIN;
        fp.isSharp = fp.entropy > 0.5;
        fp.materialImbalance = count_material(fp.fen);
        fp.numLegalMoves = 0;  // Would need actual move generation
        
        data.push_back(fp);
    }
    
    std::cout << "Converted to " << data.size() << " filtered positions\n";
    
    // Apply filters in order
    if (activeFilters & FILTER_DUPLICATE) {
        std::cout << "Removing duplicates...\n";
        data = remove_duplicates(data);
    }
    
    if (activeFilters & FILTER_FORCED_MATE) {
        std::cout << "Removing forced mates...\n";
        data = remove_forced_mates(data);
    }
    
    if (activeFilters & FILTER_LOW_ENTROPY) {
        std::cout << "Filtering by entropy...\n";
        data = filter_by_entropy(data);
    }
    
    if (activeFilters & FILTER_CHECK) {
        std::cout << "Removing positions in check...\n";
        data.erase(
            std::remove_if(data.begin(), data.end(),
                [](const FilteredPosition& p) { return p.inCheck; }),
            data.end()
        );
    }
    
    if (activeFilters & FILTER_OPENING) {
        std::cout << "Removing early opening positions...\n";
        data.erase(
            std::remove_if(data.begin(), data.end(),
                [this](const FilteredPosition& p) { return p.ply < openingPly; }),
            data.end()
        );
    }
    
    if (activeFilters & FILTER_ENDGAME) {
        std::cout << "Filtering endgame...\n";
        data = filter_by_phase(data);
    }
    
    if (activeFilters & FILTER_HIGH_SCORE) {
        std::cout << "Filtering high scores...\n";
        data.erase(
            std::remove_if(data.begin(), data.end(),
                [this](const FilteredPosition& p) { 
                    return std::abs(p.score) > scoreThreshold; 
                }),
            data.end()
        );
    }
    
    // Always balance results
    std::cout << "Balancing results...\n";
    data = balance_results(data);
    
    std::cout << "Final: " << data.size() << " positions\n";
    
    return data;
}

std::vector<FilteredPosition> DataPipeline::remove_duplicates(
    std::vector<FilteredPosition>& data)
{
    std::unordered_set<PositionHash, PositionHashHasher> seen;
    std::vector<FilteredPosition> result;
    result.reserve(data.size());
    
    size_t removed = 0;
    for (auto& pos : data) {
        PositionHash h = compute_hash(pos.fen);
        
        if (seen.find(h) == seen.end()) {
            seen.insert(h);
            result.push_back(pos);
        } else {
            removed++;
        }
    }
    
    std::cout << "  Removed " << removed << " duplicates\n";
    return result;
}

std::vector<FilteredPosition> DataPipeline::remove_forced_mates(
    std::vector<FilteredPosition>& data)
{
    size_t removed = 0;
    
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [&removed](const FilteredPosition& pos) {
                bool isMate = std::abs(pos.score) >= VALUE_MATE_IN_MAX_PLY;
                if (isMate) removed++;
                return isMate;
            }),
        data.end()
    );
    
    std::cout << "  Removed " << removed << " forced mates\n";
    return data;
}

std::vector<FilteredPosition> DataPipeline::filter_by_entropy(
    std::vector<FilteredPosition>& data)
{
    size_t removed = 0;
    
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [this, &removed](const FilteredPosition& pos) {
                bool low = pos.entropy < entropyThreshold;
                if (low) removed++;
                return low;
            }),
        data.end()
    );
    
    std::cout << "  Removed " << removed << " low-entropy positions\n";
    return data;
}

std::vector<FilteredPosition> DataPipeline::filter_by_phase(
    std::vector<FilteredPosition>& data)
{
    // Keep balanced mix of phases
    size_t endgameRemoved = 0;
    
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [&endgameRemoved](const FilteredPosition& pos) {
                // Remove very late endgame (under 6 pieces)
                bool lateEndgame = pos.materialImbalance < 6;
                if (lateEndgame) endgameRemoved++;
                return lateEndgame;
            }),
        data.end()
    );
    
    std::cout << "  Removed " << endgameRemoved << " late endgame positions\n";
    return data;
}

std::vector<FilteredPosition> DataPipeline::balance_results(
    std::vector<FilteredPosition>& data)
{
    // Count results
    int wins = 0, draws = 0, losses = 0;
    for (const auto& pos : data) {
        if (pos.result == 1) wins++;
        else if (pos.result == 0) draws++;
        else if (pos.result == -1) losses++;
    }
    
    int minCount = std::min({wins, draws, losses});
    
    if (minCount == 0) {
        std::cout << "  Warning: unbalanced results (W/D/L: " 
                  << wins << "/" << draws << "/" << losses << ")\n";
        return data;
    }
    
    // Shuffle and limit
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    
    int w = 0, d = 0, l = 0;
    std::vector<FilteredPosition> balanced;
    balanced.reserve(minCount * 3);
    
    for (const auto& pos : data) {
        bool keep = false;
        if (pos.result == 1 && w < minCount) { w++; keep = true; }
        else if (pos.result == 0 && d < minCount) { d++; keep = true; }
        else if (pos.result == -1 && l < minCount) { l++; keep = true; }
        
        if (keep) balanced.push_back(pos);
    }
    
    std::cout << "  Balanced to " << balanced.size() 
              << " (W/D/L: " << w << "/" << d << "/" << l << ")\n";
    return balanced;
}

std::vector<FilteredPosition> DataPipeline::filter_sharp_positions(
    std::vector<FilteredPosition>& data, double percentile)
{
    if (data.empty()) return data;
    
    // Sort by entropy
    std::vector<FilteredPosition> sorted = data;
    std::sort(sorted.begin(), sorted.end(),
        [](const FilteredPosition& a, const FilteredPosition& b) {
            return a.entropy > b.entropy;
        });
    
    // Keep top percentile
    size_t keepCount = static_cast<size_t>(sorted.size() * percentile);
    keepCount = std::max(keepCount, size_t(1));
    
    std::cout << "  Kept top " << percentile * 100 << "% (" 
              << keepCount << ") sharp positions\n";
    
    return std::vector<FilteredPosition>(
        sorted.begin(), sorted.begin() + keepCount);
}

std::vector<FilteredPosition> DataPipeline::filter_quiet_positions(
    std::vector<FilteredPosition>& data)
{
    data.erase(
        std::remove_if(data.begin(), data.end(),
            [](const FilteredPosition& pos) { return !pos.isQuiet; }),
        data.end()
    );
    
    return data;
}

std::vector<FilteredPosition> DataPipeline::mine_hard_positions(
    const std::vector<FilteredPosition>& enginePositions,
    const std::vector<FilteredPosition>& referencePositions)
{
    std::vector<FilteredPosition> hard;
    
    // Match positions by FEN and find disagreements
    for (const auto& ep : enginePositions) {
        auto it = std::find_if(referencePositions.begin(), referencePositions.end(),
            [&ep](const FilteredPosition& rp) {
                return rp.fen == ep.fen;
            });
        
        if (it != referencePositions.end()) {
            double scoreDiff = std::abs(it->score - ep.score);
            if (scoreDiff > 50) {  // 50 cp difference threshold
                // Use reference score as "truth"
                FilteredPosition corrected = ep;
                corrected.score = it->score;
                corrected.isSharp = true;
                hard.push_back(corrected);
            }
        }
    }
    
    std::cout << "Mined " << hard.size() << " hard positions\n";
    return hard;
}

QualityMetrics DataPipeline::analyze(
    const std::vector<FilteredPosition>& data) const
{
    QualityMetrics m{};
    m.totalPositions = data.size();
    m.afterFiltering = data.size();
    
    if (data.empty()) return m;
    
    // Count categories
    int quiet = 0, sharp = 0, endgame = 0, opening = 0;
    double totalEntropy = 0.0;
    double scoreSqSum = 0.0;
    double scoreSum = 0.0;
    
    for (const auto& pos : data) {
        if (pos.isQuiet) quiet++;
        if (pos.isSharp) sharp++;
        if (pos.gamePhase == 3) endgame++;
        if (pos.ply < 20) opening++;
        
        totalEntropy += pos.entropy;
        scoreSum += pos.score;
        scoreSqSum += pos.score * pos.score;
    }
    
    m.quietPositions = quiet;
    m.tacticalPositions = sharp;
    m.endgamePositions = endgame;
    m.openingPositions = opening;
    m.avgEntropy = totalEntropy / data.size();
    
    double meanScore = scoreSum / data.size();
    m.scoreVariance = scoreSqSum / data.size() - meanScore * meanScore;
    
    // Result balance
    int wins = 0, draws = 0, losses = 0;
    for (const auto& pos : data) {
        if (pos.result == 1) wins++;
        else if (pos.result == 0) draws++;
        else losses++;
    }
    
    double total = wins + draws + losses;
    if (total > 0) {
        m.resultBalance = 1.0 - (
            std::abs(wins/total - 0.33) +
            std::abs(draws/total - 0.33) +
            std::abs(losses/total - 0.33)
        ) / 2.0;
    }
    
    return m;
}

void QualityMetrics::print() const
{
    std::cout << "\n=== Data Quality Metrics ===\n";
    std::cout << "Total positions: " << totalPositions << "\n";
    std::cout << "After filtering:   " << afterFiltering << "\n";
    std::cout << "Duplicates removed:" << duplicatesRemoved << "\n";
    std::cout << "Forced mates rem.:" << forcedMatesRemoved << "\n";
    std::cout << "Quiet positions:   " << quietPositions << "\n";
    std::cout << "Tactical positions:" << tacticalPositions << "\n";
    std::cout << "Endgame positions: " << endgamePositions << "\n";
    std::cout << "Opening positions: " << openingPositions << "\n";
    std::cout << "Low entropy rem.:  " << lowEntropyRemoved << "\n";
    std::cout << "\nAverage entropy:   " << avgEntropy << "\n";
    std::cout << "Score variance:    " << scoreVariance << "\n";
    std::cout << "Result balance:    " << resultBalance 
              << " (1.0 = perfect)\n";
}

void DataPipeline::export_plain(const std::string& filename,
                                const std::vector<FilteredPosition>& data)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }
    
    for (const auto& pos : data) {
        file << pos.fen << " | " 
             << pos.score << " | " 
             << pos.result << " | " 
             << pos.gamePhase << " | " 
             << pos.entropy << "\n";
    }
    
    std::cout << "Exported " << data.size() << " positions to " << filename << "\n";
}

void DataPipeline::export_binpack(const std::string& filename,
                                  const std::vector<FilteredPosition>& data)
{
    // Binary packed format for efficient storage
    // Placeholder - would implement actual binpack format
    std::cout << "Binpack export: TODO\n";
}

void DataPipeline::export_for_nnue(const std::string& filename,
                                   const std::vector<FilteredPosition>& data)
{
    // Export in format expected by train_nnue.py
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }
    
    for (const auto& pos : data) {
        // Format: FEN | score | result | phase | game_id
        file << pos.fen << " | " 
             << pos.score << " | " 
             << pos.result << " | " 
             << pos.gamePhase << " | " 
             << pos.gameId << "\n";
    }
    
    std::cout << "Exported " << data.size() 
              << " positions for NNUE training to " << filename << "\n";
}

// ================ Helper Functions ================

double DataPipeline::compute_entropy(const FilteredPosition& pos) const
{
    // Position entropy based on:
    // 1. Number of pieces (more pieces = higher entropy)
    // 2. Material imbalance (closer to equal = higher entropy)
    // 3. Score magnitude (closer to 0 = higher entropy)
    
    double pieceEntropy = std::min(pos.materialImbalance / 32.0, 1.0);
    
    double scoreNormalized = std::abs(pos.score) / 1000.0;
    double scoreEntropy = 1.0 - std::min(scoreNormalized, 1.0);
    
    // Combine
    return 0.6 * pieceEntropy + 0.4 * scoreEntropy;
}

bool DataPipeline::is_forced_mate(const FilteredPosition& pos) const
{
    return std::abs(pos.score) >= VALUE_MATE_IN_MAX_PLY;
}

int DataPipeline::count_material(const std::string& fen) const
{
    // Count total pieces from FEN board
    int count = 0;
    size_t space = fen.find(' ');
    std::string board = fen.substr(0, space);
    
    for (char c : board) {
        if (c >= 'A' && c <= 'Z') count++;
        else if (c >= 'a' && c <= 'z') count++;
    }
    
    return count;
}

// ================ StreamingPipeline ================

StreamingPipeline::StreamingPipeline(
    const std::string& inputFile,
    const std::string& outputFile,
    uint32_t filters)
    : inputPath(inputFile)
    , outputPath(outputFile)
    , filters(filters)
    , processedCount(0)
    , writtenCount(0)
{
}

void StreamingPipeline::process_stream()
{
    std::ifstream in(inputPath);
    std::ofstream out(outputPath);
    
    if (!in.is_open()) {
        std::cerr << "Failed to open input: " << inputPath << "\n";
        return;
    }
    if (!out.is_open()) {
        std::cerr << "Failed to open output: " << outputPath << "\n";
        return;
    }
    
    std::string line;
    while (std::getline(in, line)) {
        processedCount++;
        
        // Parse position
        // Simple FEN extraction
        line = line.substr(0, line.find('|'));
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) continue;
        
        // Check deduplication
        if (filters & FILTER_DUPLICATE) {
            PositionHash h = compute_hash(line);
            if (seenCache.find(h) != seenCache.end()) {
                continue;
            }
            
            seenCache.insert(h);
            
            // Cache size limit
            if (seenCache.size() > CACHE_SIZE) {
                // Simple eviction: clear and rebuild periodically
                if (processedCount % (CACHE_SIZE * 10) == 0) {
                    seenCache.clear();
                }
            }
        }
        
        // Write
        out << line << "\n";
        writtenCount++;
        
        if (processedCount % 100000 == 0) {
            std::cout << "Processed " << processedCount 
                      << ", written " << writtenCount 
                      << " (" << (100.0 * writtenCount / processedCount) 
                      << "%)\r" << std::flush;
        }
    }
    
    std::cout << "\nDone. Processed " << processedCount 
              << ", wrote " << writtenCount << "\n";
}

// ================ DiversitySampler ================

std::vector<FilteredPosition> DiversitySampler::sample_diverse(
    const std::vector<FilteredPosition>& data,
    size_t targetSize)
{
    if (data.size() <= targetSize) return data;
    
    // Reservoir sampling with diversity boost
    std::vector<FilteredPosition> result;
    result.reserve(targetSize);
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    // Stratified by entropy buckets
    std::vector<std::vector<const FilteredPosition*>> buckets(10);
    
    for (const auto& pos : data) {
        int bucket = static_cast<int>(pos.entropy * 9.999);
        bucket = std::clamp(bucket, 0, 9);
        buckets[bucket].push_back(&pos);
    }
    
    // Sample from each bucket proportionally
    size_t perBucket = targetSize / 10;
    
    for (auto& bucket : buckets) {
        if (bucket.empty()) continue;
        
        std::shuffle(bucket.begin(), bucket.end(), g);
        
        size_t take = std::min(perBucket, bucket.size());
        for (size_t i = 0; i < take; ++i) {
            result.push_back(*bucket[i]);
        }
    }
    
    // Fill remaining
    while (result.size() < targetSize) {
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        result.push_back(data[dist(g)]);
    }
    
    std::shuffle(result.begin(), result.end(), g);
    result.resize(targetSize);
    
    return result;
}

std::vector<FilteredPosition> DiversitySampler::sample_stratified(
    const std::vector<FilteredPosition>& data,
    size_t perCategory)
{
    // Group by phase
    std::vector<FilteredPosition> opening, middlegame, endgame;
    
    for (const auto& pos : data) {
        switch (pos.gamePhase) {
            case 1: opening.push_back(pos); break;
            case 2: middlegame.push_back(pos); break;
            case 3: endgame.push_back(pos); break;
        }
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    std::vector<FilteredPosition> result;
    result.reserve(perCategory * 3);
    
    auto sample = [&g, &result, perCategory](std::vector<FilteredPosition>& vec) {
        std::shuffle(vec.begin(), vec.end(), g);
        size_t take = std::min(perCategory, vec.size());
        for (size_t i = 0; i < take; ++i) {
            result.push_back(vec[i]);
        }
    };
    
    sample(opening);
    sample(middlegame);
    sample(endgame);
    
    return result;
}

std::vector<std::vector<FilteredPosition>> DiversitySampler::cluster_positions(
    const std::vector<FilteredPosition>& data,
    int numClusters)
{
    // K-means-like clustering by entropy and score
    std::vector<std::vector<FilteredPosition>> clusters(numClusters);
    
    if (data.empty()) return clusters;
    
    // Simple: divide into equal buckets based on combined metric
    std::vector<std::pair<double, const FilteredPosition*>> scored;
    scored.reserve(data.size());
    
    for (const auto& pos : data) {
        double metric = pos.entropy * 0.5 + 
                       (std::abs(pos.score) / 1000.0) * 0.3 +
                       (pos.materialImbalance / 32.0) * 0.2;
        scored.push_back({metric, &pos});
    }
    
    std::sort(scored.begin(), scored.end());
    
    // Distribute into clusters
    size_t perCluster = scored.size() / numClusters;
    
    for (int i = 0; i < numClusters; ++i) {
        size_t start = i * perCluster;
        size_t end = (i == numClusters - 1) ? scored.size() : (i + 1) * perCluster;
        
        for (size_t j = start; j < end; ++j) {
            clusters[i].push_back(*scored[j].second);
        }
    }
    
    return clusters;
}

} // namespace DataPipeline
