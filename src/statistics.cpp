#include "statistics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <ranges>
#include <vector>

namespace vmcp {

FPType

    // Helper function for Blocking, fills blocks-vectors with desired statistic
    std::vector<Energy>
    EvalStatBlocks_(std::vector<Energy> const &energies, IntType numEnergies, IntType blockSize,
                    IntType numOfBlocks) {
    std::vector<Energy> blockStats;
    blockStats.reserve(static_cast<UIntType>(numOfBlocks));
    assert((numEnergies % blockSize) == 0);

    IntType currentBlock = 0;
    std::generate_n(std::back_inserter(blockStats), numOfBlocks,
                    [&energies, &numEnergies, &blockSize, &currentBlock, &stat]() {
                        AccumulatorSet accBlock;
                        auto start = energies.begin() + currentBlock * blockSize;
                        auto end = start + blockSize;

                        FillAcc_(accBlock, std::vector<Energy>(start, end));
                        ++currentBlock;
                        return Energy{GetStatValue_(accBlock, stat)};
                    });
    return blockStats;
}

// Core helper method for Blocking
BlockingResult BlockingAnalysis_(std::vector<Energy> const &energies) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    // Check numEnergies is a power of 2 using bitwise AND between the number n and (n - 1)
    assert((numEnergies & (numEnergies - 1)) == 0);
    assert(numEnergies > 1);

    std::vector<IntType> blockSizes;
    std::vector<FPType> means;
    std::vector<FPType> stdDevs;

    UIntType reservedSize = static_cast<UIntType>(numEnergies / 2);
    blockSizes.reserve(reservedSize);
    means.reserve(reservedSize);
    stdDevs.reserve(reservedSize);

    for (IntType blockSize = 2; blockSize <= numEnergies / 2; blockSize *= 2) {
        IntType numOfBlocks = static_cast<IntType>(static_cast<FPType>(numEnergies) / blockSize);
        // Evaluate mean of each block
        std::vector<Energy> blockMeans = EvalStatBlocks_(energies, numEnergies, blockSize, numOfBlocks, mean);
        // Evaluate second moment (mean squared) of each block
        std::vector<Energy> blockSecondM =
            EvalStatBlocks_(energies, numEnergies, blockSize, numOfBlocks, secondMoment);

        AccumulatorSet accBlocksMeans;
        AccumulatorSet accBlocksSecondM;

        FillAcc_(accBlocksMeans, blockMeans);
        FillAcc_(accBlocksSecondM, blockSecondM);

        // Statistics
        blockSizes.push_back(blockSize);
        FPType meanOfMeans = GetStatValue_(accBlocksMeans, mean);
        means.push_back(meanOfMeans);
        FPType stdDev =
            std::sqrt((GetStatValue_(accBlocksSecondM, mean) - std::pow(meanOfMeans, 2)) / (numOfBlocks - 1));
        stdDevs.push_back(stdDev);
    }
    assert(blockSizes.size() == means.size());
    assert(blockSizes.size() == stdDevs.size());
    return BlockingResult{blockSizes, means, stdDevs};
}

VMCResult BlockingOut(std::vector<Energy> const &energies) {
    BlockingResult blockingResult = BlockingAnalysis_(energies);

    // Find maximum value of standard dev.
    auto maxIt = std::max_element(blockingResult.stdDevs.begin(), blockingResult.stdDevs.end());
    FPType bestStdDev = *maxIt;

    // Compute the index of the maximum std. dev. element
    IntType index = static_cast<IntType>(std::distance(blockingResult.stdDevs.begin(), maxIt));
    FPType mean = blockingResult.means[static_cast<UIntType>(index)];
    return VMCResult{mean, bestStdDev};
}

// Helper function for Bootstrapping
std::vector<Energy> GenerateBootstrapSample_(std::vector<Energy> const &energies,
                                             UIntType const &numEnergies_, RandomGenerator &gen,
                                             std::uniform_int_distribution<> &dist) {
    std::vector<Energy> sample;
    // Resample with replacement
    std::generate_n(std::back_inserter(sample), numEnergies_,
                    [&]() { return energies[static_cast<UIntType>(dist(gen))]; });
    return sample;
}

// Helper function for Bootstrapping, fills sample-vectors with desired statistic
std::vector<Energy> FillBootstrapVec_(std::vector<Energy> const &energies, IntType numEnergies,
                                      IntType numSamples, RandomGenerator &gen,
                                      std::uniform_int_distribution<> &uIntDis, StatisticType stat) {
    std::vector<Energy> bootstrapVector;
    AccumulatorSet accSample;

    std::generate_n(std::back_inserter(bootstrapVector), numSamples,
                    [&accSample, &energies, &numEnergies, &gen, &uIntDis, &stat]() {
                        accSample = {};
                        std::vector<Energy> bootstrapSample = GenerateBootstrapSample_(
                            energies, static_cast<UIntType>(numEnergies), gen, uIntDis);
                        FillAcc_(accSample, bootstrapSample);
                        return Energy{GetStatValue_(accSample, stat)};
                    });

    return bootstrapVector;
}

VMCResult BootstrapAnalysis(std::vector<Energy> const &energies, IntType const numSamples,
                            RandomGenerator &gen) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    assert(numEnergies > 1);
    std::uniform_int_distribution<> uIntDis(0, (numEnergies - 1));

    // Calculate mean of each sample vector and place into bootstrapMeans
    std::vector<Energy> bootstrapMeans =
        FillBootstrapVec_(energies, numEnergies, numSamples, gen, uIntDis, mean);
    // Calculate second moment (squared mean) of each sample vector and place into bootstrapMeans
    std::vector<Energy> bootstrapSecondM =
        FillBootstrapVec_(energies, numEnergies, numSamples, gen, uIntDis, secondMoment);

    AccumulatorSet accBlocksMeans;
    AccumulatorSet accBlocksSecondM;
    FillAcc_(accBlocksMeans, bootstrapMeans);
    FillAcc_(accBlocksSecondM, bootstrapSecondM);

    // Calculate statistics of each means
    FPType meanOfMeans = GetStatValue_(accBlocksMeans, mean);
    FPType variance = (GetStatValue_(accBlocksSecondM, mean) - std::pow(meanOfMeans, 2)) / (numEnergies - 1);
    return VMCResult{meanOfMeans, variance};
}

ConfInterval GetConfInt(FPType mean, FPType variance) {
    ConfInterval confInterval;
    confInterval.min = mean - std::sqrt(variance) * zScore;
    confInterval.max = mean + std::sqrt(variance) * zScore;
    return confInterval;
}

} // namespace vmcp
