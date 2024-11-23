//!
//! @file statistics.inl
//! @brief Definition of the templated statistics algorithms
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the templated statistics algorithms declared in the header.
//! Among the helper functions used in said definitions, the templated ones are also defined here, while the
//! non-templated ones are in the .inl file.
//! @see statistics.hpp
//!

#ifndef VMCPROJECT_STATISTICS_INL
#define VMCPROJECT_STATISTICS_INL

#include "statistics.hpp"

#include <algorithm>
#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <fstream>
#include <numeric>
#include <ranges>
#include <vector>

namespace vmcp {

//! @defgroup helpers Helpers
//! @brief Help the core functions.
//! @{

//! @brief Calculates the mean
//! @param v The energies and positions, where only the energies will be averaged
//! @return The mean
template <Dimension D, ParticNum N>
Energy GetMean(std::vector<LocEnAndPoss<D, N>> const &v) {
    assert(v.size() > 1);
    auto const size = std::ssize(v);

    Energy const mean = std::accumulate(v.begin(), v.end(), Energy{0},
                                        [](Energy e, LocEnAndPoss<D, N> leps) { return e + leps.localEn; }) /
                        static_cast<FPType>(size);
    return mean;
};

//! @brief Calculates the error on the mean (by taking just one standard deviation)
//! @param v The energies and positions, where only the energies will be averaged
//! @return The standard deviation
template <Dimension D, ParticNum N>
Energy GetStdDev(std::vector<LocEnAndPoss<D, N>> const &v) {
    assert(v.size() > 1);
    auto const size = std::ssize(v);

    Energy const mean = GetMean(v);
    EnSquared const meanVar = std::accumulate(v.begin(), v.end(), EnSquared{0},
                                              [mean](EnSquared es, LocEnAndPoss<D, N> const &leps) {
                                                  return es + (leps.localEn - mean) * (leps.localEn - mean);
                                              }) /
                              static_cast<FPType>(size * (size - 1));
    return sqrt(meanVar);
}

//! @brief Calculates the mean or error on the mean (by taking just one standard deviation)
//! depending on call
//! @param v The energies and positions, where only the energies will be averaged
//! @param stat The desired statistic (mean or standard deviation)
//! @return The evaluation of the desired statistic
template <Dimension D, ParticNum N>
Energy GetStat(std::vector<LocEnAndPoss<D, N>> const &v, StatisticType stat) {
    switch (stat) {
    case mean:
        return GetMean(v);
    case stdDev:
        return GetStdDev(v);
    default:
        assert(false);
    }
}

// Helper function for Blocking, fills blocks-vectors with desired statistic
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>> EvalStatBlocks(std::vector<LocEnAndPoss<D, N>> const &energies,
                                               IntType numEnergies, IntType blockSize, IntType numOfBlocks,
                                               StatisticType stat) {
    std::vector<LocEnAndPoss<D, N>> blockStats;
    blockStats.reserve(static_cast<UIntType>(numOfBlocks));
    assert(numEnergies % blockSize == 0);

    LocEnAndPoss<D, N> blockLeps;

    IntType currentBlock = 0;
    std::generate_n(std::back_inserter(blockStats), numOfBlocks,
                    [&energies, &numEnergies, &blockSize, &currentBlock, &stat]() {
                        auto start = energies.begin() + currentBlock * blockSize;
                        auto end = start + blockSize;

                        blockLeps.localEn = GetStat(std::vector<LocEnAndPoss<D, N>>(start, end), stat);
                        ++currentBlock;
                        return blockLeps;
                    });
    return blockStats;
}

// Core helper method for Blocking
template <Dimension D, ParticNum N>
BlockingResult BlockingAnalysis(std::vector<LocEnAndPoss<D, N>> const &energies) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    // Check numEnergies is a power of 2 using bitwise AND between the number n and (n - 1)
    assert((numEnergies & (numEnergies - 1)) == 0);
    assert(numEnergies > 1);

    std::vector<IntType> blockSizes;
    std::vector<Energy> means;
    std::vector<Energy> stdDevs;

    UIntType reservedSize = static_cast<UIntType>(numEnergies / 2);
    blockSizes.reserve(reservedSize);
    means.reserve(reservedSize);
    stdDevs.reserve(reservedSize);

    for (IntType blockSize = 2; blockSize <= numEnergies / 2; blockSize *= 2) {
        IntType numOfBlocks = static_cast<IntType>(static_cast<FPType>(numEnergies) / blockSize);
        // Evaluate mean of each block
        std::vector<LocEnAndPoss<D, N>> blockMeans =
            EvalStatBlocks(energies, numEnergies, blockSize, numOfBlocks, mean);
        // Evaluate std. dev. of each block
        std::vector<LocEnAndPoss<D, N>> blockStdDev =
            EvalStatBlocks(energies, numEnergies, blockSize, numOfBlocks, stdDev);

        blockSizes.push_back(blockSize);

        // Statistics
        Energy meanOfMeans = GetStat(blockMeans, mean);
        means.push_back(meanOfMeans);
        Energy stdDev = GetStat(blockStdDev, mean);
        stdDevs.push_back(stdDev);
    }
    assert(blockSizes.size() == means.size());
    assert(blockSizes.size() == stdDevs.size());
    return BlockingResult{blockSizes, means, stdDevs};
}

template <Dimension D, ParticNum N>
VMCResult BlockingOut(std::vector<LocEnAndPoss<D, N>> const &energies) {
    BlockingResult blockingResult = BlockingAnalysis(energies);

    // Find maximum value of standard dev.
    auto maxIt = std::max_element(blockingResult.stdDevs.begin(), blockingResult.stdDevs.end());
    Energy bestStdDev = *maxIt;

    // Compute the index of the maximum std. dev. element // LF FIXME: take the maximum of last 3/4 elements?
    IntType index = static_cast<IntType>(std::distance(blockingResult.stdDevs.begin(), maxIt));
    Energy mean = blockingResult.means[static_cast<UIntType>(index)];
    return VMCResult{mean, bestStdDev};
}

// Helper function for Bootstrapping
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>> GenerateBootstrapSample(std::vector<LocEnAndPoss<D, N>> const &energies,
                                                        UIntType const &numEnergies, RandomGenerator &gen,
                                                        std::uniform_int_distribution<> &dist) {
    std::vector<LocEnAndPoss<D, N>> sample;

    // Resample with replacement
    std::generate_n(std::back_inserter(sample), numEnergies,
                    [&]() { return energies[static_cast<UIntType>(dist(gen))]; });
    return sample;
}

// Helper function for Bootstrapping, fills sample-vectors with desired statistic
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>>
FillBootstrapVec(std::vector<LocEnAndPoss<D, N>> const &energies, IntType numEnergies, IntType numSamples,
                 RandomGenerator &gen, std::uniform_int_distribution<> &uIntDis, StatisticType stat) {
    std::vector<Energy> bootstrapVector;

    std::generate_n(
        std::back_inserter(bootstrapVector), numSamples, [&energies, &numEnergies, &gen, &uIntDis, &stat]() {
            std::vector<LocEnAndPoss<D, N>> bootstrapSample =
                GenerateBootstrapSample_(energies, static_cast<UIntType>(numEnergies), gen, uIntDis);
            return GetStatValue(bootstrapSample, stat);
        });
    return bootstrapVector;
}

template <Dimension D, ParticNum N>
VMCResult BootstrapAnalysis(std::vector<LocEnAndPoss<D, N>> const &energies, IntType const &numSamples,
                            RandomGenerator &gen) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    assert(numEnergies > 1);
    std::uniform_int_distribution<> uIntDis(0, (numEnergies - 1));

    // Calculate mean of each sample vector and place into bootstrapMeans
    std::vector<LocEnAndPoss<D, N>> bootstrapMeans =
        FillBootstrapVec(energies, numEnergies, numSamples, gen, uIntDis, mean);
    // Calculate second moment (squared mean) of each sample vector and place into bootstrapMeans
    std::vector<LocEnAndPoss<D, N>> bootstrapStdDevs =
        FillBootstrapVec(energies, numEnergies, numSamples, gen, uIntDis, stdDev);

    // Calculate statistics of each means
    Energy meanOfMeans = GetStat(bootstrapMeans, mean);
    Energy stdDev = GetStat(bootstrapStdDevs, mean);
    return VMCResult{meanOfMeans, stdDev};
}

ConfInterval GetConfInt(Energy mean, Energy stdDev, IntType percentage) {
    ConfInterval confInterval;

    boost::math::normal_distribution<FPType> dist(mean.val, stdDev.val);
    FPType z = boost::math::quantile(dist, 1 - (1 - percentage / 100) / 2);

    confInterval.min = mean - stdDev * z;
    confInterval.max = mean + stdDev * z;
    return confInterval;
}

} // namespace vmcp

#endif
