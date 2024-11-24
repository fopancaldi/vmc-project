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
//!
//! Helper for 'GetStat'
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
//!
//! Helper for 'GetStat'
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
    case StatisticType::mean:
        return GetMean(v);
    case StatisticType::stdDev:
        return GetStdDev(v);
    default:
        assert(false);
    }
}

//! @brief Fills blocks-vectors with desired statistic
//! @param energies The energies and positions, where only the energies will be used
//! @param numEnergies The number of energies
//! @see BlockingAnalysis
//!
//! @param blockSize size of the blocks for which the desired statistic is evaluated
//! @param numOfBlocks number of blocks with size blockSize
//! @param stat statistic that will be evaluated for each block
//! @return A vector of energies (and positions) contaning the statistic for each block
//!
//! Helper for 'BlockingAnalysis'
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>> EvalStatBlocks(std::vector<LocEnAndPoss<D, N>> const &energies,
                                               IntType numEnergies, IntType blockSize, IntType numOfBlocks,
                                               StatisticType stat) {
    std::vector<LocEnAndPoss<D, N>> blockStats;
    blockStats.reserve(static_cast<UIntType>(numOfBlocks));
    assert(numEnergies % blockSize == 0);

    IntType currentBlock = 0;

    std::generate_n(std::back_inserter(blockStats), numOfBlocks,
                    [&energies, &numEnergies, &blockSize, &currentBlock, &stat]() {
                        auto start = energies.begin() + currentBlock * blockSize;
                        auto end = start + blockSize;

                        LocEnAndPoss<D, N> blockLeps;
                        blockLeps.localEn = GetStat(std::vector<LocEnAndPoss<D, N>>(start, end), stat);

                        ++currentBlock;
                        return blockLeps;
                    });
    return blockStats;
}

//! @brief Helper function for Bootstrapping
//! @param energies The energies and positions, where only the energies will be used
//! @param numEnergies The number of energies
//! @see BootstrapAnalysis
//!
//! @param numSamples The number of samples that will be generated
//! @param gen The random generator
//! @param dist The uniform integer distribution used to extract random elements from energies vector
//! @return A vector of energies (and positions) containing the generated samples
//!
//! Helper function for 'BootstrappingAnalysis'
template <Dimension D, ParticNum N>
std::vector<std::vector<LocEnAndPoss<D, N>>>
GenerateBootstrapSamples(std::vector<LocEnAndPoss<D, N>> const &energies, UIntType const numEnergies,
                         IntType const numSamples, RandomGenerator &gen,
                         std::uniform_int_distribution<> &dist) {
    std::vector<std::vector<LocEnAndPoss<D, N>>> bootstrapSamples;
    bootstrapSamples.reserve(static_cast<UIntType>(numSamples));

    // Resample with replacement
    std::generate_n(std::back_inserter(bootstrapSamples), numSamples,
                    [&energies, &numEnergies, &dist, &gen]() {
                        std::vector<LocEnAndPoss<D, N>> sample;
                        sample.reserve(numEnergies);

                        // Fill the current sample with random energies
                        std::generate_n(std::back_inserter(sample), numEnergies,
                                        [&]() { return energies[static_cast<UIntType>(dist(gen))]; });

                        return sample;
                    });
    return bootstrapSamples;
}

//! @brief Calculates the desired statistic of each sample vector
//! @param bootstrapSamples The energies and positions, where only the energies will be used
//! @param numSamples The number of samples that will be generated
//! @see GenerateBootstrapSamples
//!
//! @param stat statistic that will be evaluated for each block
//! @return A vector of energies (and positions) containing the calculated statistic for each sample
//!
//! Helper function for 'BootstrappingAnalysis'
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>>
FillBootstrapVec(std::vector<std::vector<LocEnAndPoss<D, N>>> const &bootstrapSamples, IntType numSamples,
                 StatisticType stat) {
    std::vector<LocEnAndPoss<D, N>> bootstrapVector;
    bootstrapVector.reserve(static_cast<UIntType>(numSamples));

    // Track the current position in bootstrapSamples
    UIntType currentSample = 0;

    std::generate_n(std::back_inserter(bootstrapVector), numSamples,
                    [&bootstrapSamples, &currentSample, &stat]() {
                        const auto &sample = bootstrapSamples[currentSample];
                        ++currentSample;

                        LocEnAndPoss<D, N> leps;
                        leps.localEn = GetStat(sample, stat);
                        return leps;
                    });
    return bootstrapVector;
}

//! @}

//! @defgroup core-functions Core functions
//! @brief The most important functions in the code
//!
//! The ones that actually do the work.
//! @{

//! @brief Core helper method for Blocking, divides dataset into multiple blocks with a certain block size,
//! then evaluates means of each block and takes the mean of means and standard deviation of means (for 
//! each block size)
//! @param energies The energies and positions, where only the energies will be used
//! @param numEnergies The number of energies
//! @see BlockingAnalysis
//!
//! @return Three vectors containing resepctively a list of block sizes, means and standard deviations
template <Dimension D, ParticNum N>
BlockingResult EvalBlocking(std::vector<LocEnAndPoss<D, N>> const &energies, IntType const &numEnergies) {
    std::vector<IntType> blockSizes;
    std::vector<Energy> means;
    std::vector<Energy> stdDevs;

    UIntType reservedSize = static_cast<UIntType>(std::log2(numEnergies) - 1);
    blockSizes.reserve(reservedSize);
    means.reserve(reservedSize);
    stdDevs.reserve(reservedSize);

    for (IntType blockSize = 2; blockSize <= numEnergies / 2; blockSize *= 2) {
        IntType numOfBlocks = static_cast<IntType>(static_cast<FPType>(numEnergies) / blockSize);
        // Evaluate mean of each block
        std::vector<LocEnAndPoss<D, N>> blockMeans =
            EvalStatBlocks(energies, numEnergies, blockSize, numOfBlocks, StatisticType::mean);

        blockSizes.push_back(blockSize);

        // Statistics
        Energy meanOfMeans = GetStat(blockMeans, StatisticType::mean);
        means.push_back(meanOfMeans);
        Energy stdDev = GetStat(blockMeans, StatisticType::stdDev);
        stdDevs.push_back(stdDev);
    }
    assert(blockSizes.size() == means.size());
    assert(blockSizes.size() == stdDevs.size());
    return BlockingResult{blockSizes, means, stdDevs};
}

//! @brief Takes the result of EvalBlocking and then looks for the plateau of standard deviation to get the
//! best estimate of the error.
//! @see EvalBlocking
//!
//! @param energies The energies and positions, where only the energies will be used
//! @return The mean corresponding to the best standard deviation and the best standard deviation
template <Dimension D, ParticNum N>
VMCResult BlockingAnalysis(std::vector<LocEnAndPoss<D, N>> const &energies) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    // Check numEnergies is a power of 2 using bitwise AND between the number n and (n - 1)
    assert((numEnergies & (numEnergies - 1)) == 0);
    assert(numEnergies > 1);

    BlockingResult blockingResult = EvalBlocking(energies, numEnergies);

    Energy const threshold = blockingResult.means[0] * FPType{0.05f};

    // Find the first pair of elements where the difference is below the threshold
    auto pltIt = std::adjacent_find(blockingResult.stdDevs.begin(), blockingResult.stdDevs.end(),
                                    [threshold](Energy a, Energy b) {
                                        return abs(b - a) < threshold; // Condition for plateau
                                    });
    assert(pltIt != blockingResult.stdDevs.end());
    Energy bestStdDev = *pltIt;

    // Compute the index of the plateau element for standard deviation
    IntType index = static_cast<IntType>(std::distance(blockingResult.stdDevs.begin(), pltIt) + 1);
    Energy mean = blockingResult.means[static_cast<UIntType>(index)];
    return VMCResult{mean, bestStdDev};
}

//! @brief Samples dataset with replacement multiple times, then evaluates mean and standard deviation of
//! each sample and takes the mean these values
//! @param energies The energies and positions, where only the energies will be used
//! @param numSamples The number of samples that will be generated
//! @param gen The random generator
//! @return The mean and standard deviation of bootstrapped samples
template <Dimension D, ParticNum N>
VMCResult BootstrapAnalysis(std::vector<LocEnAndPoss<D, N>> const &energies, IntType const &numSamples,
                            RandomGenerator &gen) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    assert(numEnergies > 1);
    std::uniform_int_distribution<> dist(0, numEnergies - 1);

    // Generate sample with replacement
    std::vector<std::vector<LocEnAndPoss<D, N>>> bootstrapSamples =
        GenerateBootstrapSamples(energies, static_cast<UIntType>(numEnergies), numSamples, gen, dist);

    // Calculate mean of each sample vector and place into bootstrapMeans
    std::vector<LocEnAndPoss<D, N>> bootstrapMeans =
        FillBootstrapVec(bootstrapSamples, numSamples, StatisticType::mean);
    // Calculate mean of each sample vector and place into bootstrapStdDevs
    std::vector<LocEnAndPoss<D, N>> bootstrapStdDevs =
        FillBootstrapVec(bootstrapSamples, numSamples, StatisticType::stdDev);

    // Calculate statistics of each means
    Energy meanOfMeans = GetStat(bootstrapMeans, StatisticType::mean);
    Energy stdDev = GetStat(bootstrapStdDevs, StatisticType::mean);
    return VMCResult{meanOfMeans, stdDev};
}

//! @}

//! @defgroup staistic-wrappers User functions
//! @brief The functions that are meant to be called by the user
//!
//! Are wrappers for the core functions.
//! @{

//! @brief Wrapper function called by the user, choose which statistical method to use
//! @param energies The energies and positions, where only the energies will be used
//! @param function The desired statistical method the user wants to apply to Monte Carlo data
//! @param numSamples The number of samples that will be generated
//! @param gen The random generator
//! @return The results (mean and standard deviation) of the desired statistical analysis
template <Dimension D, ParticNum N>
VMCResult Statistics(std::vector<LocEnAndPoss<D, N>> const &energies, StatFuncType function,
                     IntType const &numSamples, RandomGenerator &gen) {
    switch (function) {
    // Statistical analysis with Blocking
    case StatFuncType::blocking:
        return BlockingAnalysis(energies);

    // Statistical analysis with Bootstrapping
    case StatFuncType::bootstrap:
        return BootstrapAnalysis(energies, numSamples, gen);

    // Basic statistical analysis
    default:
        return MeanAndErr_(energies);
    }
}

//! @}

} // namespace vmcp

#endif
