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
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <ranges>
#include <vector>

namespace vmcp {

//! @defgroup algs-constants Constants
//! @brief Constants used in the algorithms and/or the helper functions
//!
//! Constants used in the algorithms.
//! They are named with the convention 'constantName_algorithmThatUsesIt'.
//! @{

//! @brief threshold for determining plateau in standrd deviation of blocking
constexpr Energy threshold_blockingAnalysis{0.05f};

//! @}

//! @defgroup helpers Helpers
//! @brief Help the core functions.
//! @{

//! @brief Calculates the mean
//! @param v The energies and positions, where only the energies will be averaged
//! @return The mean
//!
//! Helper for 'GetStat'
template <Dimension D, ParticNum N>
Energy Mean(std::vector<LocEnAndPoss<D, N>> const &v) {
    assert(v.size() > 1);
    auto const size = std::ssize(v);

    return std::accumulate(v.begin(), v.end(), Energy{0},
                             [](Energy e, LocEnAndPoss<D, N> leps) { return e + leps.localEn; }) /
             static_cast<FPType>(size);
};

//! @brief Calculates the error on the mean (by taking just one standard deviation)
//! @param v The energies and positions, where only the energies will be averaged
//! @return The standard deviation
//!
//! Helper for 'GetStat'
template <Dimension D, ParticNum N>
Energy StdDev(std::vector<LocEnAndPoss<D, N>> const &v) {
    assert(v.size() > 1);
    auto const size = std::ssize(v);

    Energy const mean = Mean(v);
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
Energy GetStat(std::vector<LocEnAndPoss<D, N>> const &v, Statistic stat) {
    Energy result;
    switch (stat) {
    case Statistic::mean:
        result = Mean(v);
        break;
    case Statistic::stdDev:
        result = StdDev(v);
        break;
    default:
        assert(false);
    }
    return result;
}

//! @brief Fills blocks-vectors with desired statistic
//! @param energies The energies and positions, where only the energies will be used
//! @param blockSize size of the blocks for which the desired statistic is evaluated
//! @param numBlocks number of blocks with size blockSize
//! @param stat statistic that will be evaluated for each block
//! @see BlockingAnalysis
//! @return A vector of energies (and positions) contaning the statistic for each block
//!
//! Helper for 'BlockingAnalysis'
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>> GetStatOfEachBlock(std::vector<LocEnAndPoss<D, N>> const &energies,
                                                   IntType blockSize, IntType numBlocks, Statistic stat) {
    assert(numBlocks > 0);
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    assert(numEnergies > 0);
    assert((numEnergies % blockSize) == 0);

    // The vector which will contain the statistic of each block
    std::vector<LocEnAndPoss<D, N>> blockStats;
    blockStats.reserve(static_cast<long unsigned int>(numBlocks));

    std::generate_n(std::back_inserter(blockStats), numBlocks,
                    [&energies, &numEnergies, &blockSize, &stat, currentBlock = IntType{0}]() mutable {
                        auto start = energies.begin() + currentBlock * blockSize;
                        auto end = start + blockSize;

                        LocEnAndPoss<D, N> blockLEPs;
                        blockLEPs.localEn = GetStat(std::vector<LocEnAndPoss<D, N>>(start, end), stat);
                        Position<D> fakePosition;
                        std::fill(fakePosition.begin(), fakePosition.end(),
                                  Coordinate{std::numeric_limits<FPType>::quiet_NaN()});
                        std::fill(blockLEPs.positions.begin(), blockLEPs.positions.end(), fakePosition);

                        ++currentBlock;
                        return blockLEPs;
                    });
    return blockStats;
}

//! @brief Helper function for Bootstrapping
//! @param energies The energies and positions, where only the energies will be used
//! @param boostrapSamples The number of samples that will be generated
//! @param gen The random generator
//! @return A vector of energies (and positions) containing the generated samples
//! @see BootstrapAnalysis
//!
//! Helper function for 'BootstrappingAnalysis'
template <Dimension D, ParticNum N>
std::vector<std::vector<LocEnAndPoss<D, N>>> BootstrapSamples(std::vector<LocEnAndPoss<D, N>> const &energies,
                                                              IntType boostrapSamples, RandomGenerator &gen) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    assert(numEnergies > 0);
    assert(boostrapSamples > 0);

    std::uniform_int_distribution<> dist(0, numEnergies - 1);

    std::vector<std::vector<LocEnAndPoss<D, N>>> bootstrapSamples;
    bootstrapSamples.reserve(static_cast<unsigned long int>(boostrapSamples));

    // Resample with replacement
    std::generate_n(std::back_inserter(bootstrapSamples), boostrapSamples,
                    [&energies, &numEnergies, &dist, &gen]() {
                        std::vector<LocEnAndPoss<D, N>> sample;
                        sample.reserve(static_cast<unsigned long int>(numEnergies));

                        // Fill the current sample with random energies
                        std::generate_n(std::back_inserter(sample), numEnergies, [&]() {
                            LocEnAndPoss<D, N> result = energies[static_cast<UIntType>(dist(gen))];
                            Position<D> fakePosition;
                            std::fill(fakePosition.begin(), fakePosition.end(),
                                      Coordinate{std::numeric_limits<FPType>::quiet_NaN()});
                            std::fill(result.positions.begin(), result.positions.end(), fakePosition);
                            return result;
                        });

                        return sample;
                    });
    return bootstrapSamples;
}

//! @brief Calculates the desired statistic of each sample vector
//! @param bootstrapSamples The energies and positions, where only the energies will be used
//! @param boostrapSamples The number of samples that will be generated
//! @param stat statistic that will be evaluated for each block
//! @return A vector of energies (and positions) containing the calculated statistic for each sample
//! @see BootstrapSamples
//!
//! Helper function for 'BootstrappingAnalysis'
template <Dimension D, ParticNum N>
std::vector<LocEnAndPoss<D, N>>
BootstrapLEPs(std::vector<std::vector<LocEnAndPoss<D, N>>> const &bootstrapSamples, IntType boostrapSamples,
              Statistic stat) {
    std::vector<LocEnAndPoss<D, N>> bootstrapVector;
    bootstrapVector.reserve(static_cast<long unsigned int>(boostrapSamples));

    std::generate_n(std::back_inserter(bootstrapVector), boostrapSamples,
                    [&bootstrapSamples, &stat, currentSample = UIntType{0u}]() mutable {
                        const auto &sample = bootstrapSamples[currentSample];
                        ++currentSample;

                        LocEnAndPoss<D, N> leps;
                        leps.localEn = GetStat(sample, stat);
                        Position<D> fakePosition;
                        std::fill(fakePosition.begin(), fakePosition.end(),
                                  Coordinate{std::numeric_limits<FPType>::quiet_NaN()});
                        std::fill(leps.positions.begin(), leps.positions.end(), fakePosition);
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
//! @see BlockingAnalysis
//!
//! @return Three vectors containing respectively a list of block sizes, means and standard deviations
template <Dimension D, ParticNum N>
BlockingResult EvalBlocking(std::vector<LocEnAndPoss<D, N>> const &energies) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    assert(numEnergies > 0);

    std::vector<IntType> blockSizes;
    std::vector<Energy> means;
    std::vector<Energy> stdDevs;

    unsigned long int const reservedSize = static_cast<unsigned long int>(std::log2(numEnergies) - 1);
    blockSizes.reserve(reservedSize);
    means.reserve(reservedSize);
    stdDevs.reserve(reservedSize);

    for (IntType blockSize = 2; blockSize <= numEnergies / 2; blockSize *= 2) {
        IntType numBlocks = static_cast<IntType>(static_cast<FPType>(numEnergies) / blockSize);

        std::vector<LocEnAndPoss<D, N>> blockMeans =
            GetStatOfEachBlock(energies, blockSize, numBlocks, Statistic::mean);
        blockSizes.push_back(blockSize);

        // Statistics
        Energy meanOfMeans = GetStat(blockMeans, Statistic::mean);
        means.push_back(meanOfMeans);
        Energy stdDevOfMeans = GetStat(blockMeans, Statistic::stdDev);
        stdDevs.push_back(stdDevOfMeans);
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
//! @return The best standard deviation
//!
//! EvalBlocking also returns "blockSizes" and "means". They are not used by BlockingAnalysis, they are
//! used for testing blocking method
//!
template <Dimension D, ParticNum N>
Energy BlockingAnalysis(std::vector<LocEnAndPoss<D, N>> const &energies) {
    IntType const numEnergies = static_cast<IntType>(std::ssize(energies));
    // Check that numEnergies is a power of 2 using bitwise AND between the number n and (n - 1)
    assert((numEnergies & (numEnergies - 1)) == 0);
    assert(numEnergies > 1);

    BlockingResult blockingResult = EvalBlocking(energies);

    // Find the first pair of elements where the difference is below the threshold
    auto pltIt =
        std::adjacent_find(blockingResult.stdDevs.begin(), blockingResult.stdDevs.end(),
                           [](Energy e1, Energy e2) { return abs(e2 - e1) < threshold_blockingAnalysis; });
    assert(pltIt != blockingResult.stdDevs.end());
    Energy bestStdDev = *pltIt;

    return bestStdDev;
}

//! @brief Samples dataset with replacement multiple times, then evaluates mean and standard deviation of
//! each sample and takes the mean these values
//! @param energies The energies and positions, where only the energies will be used
//! @param boostrapSamples The number of samples that will be generated
//! @param gen The random generator
//! @return The mean and standard deviation of bootstrapped samples
template <Dimension D, ParticNum N>
Energy BootstrapAnalysis(std::vector<LocEnAndPoss<D, N>> const &energies, IntType const &boostrapSamples,
                         RandomGenerator &gen) {
    // Generate sample with replacement
    std::vector<std::vector<LocEnAndPoss<D, N>>> bootstrapSamples =
        BootstrapSamples(energies, boostrapSamples, gen);

    // Calculate std. dev. of each sample vector and place into bootstrapStdDevs
    std::vector<LocEnAndPoss<D, N>> bootstrapStdDevs =
        BootstrapLEPs(bootstrapSamples, boostrapSamples, Statistic::stdDev);

    Energy stdDev = GetStat(bootstrapStdDevs, Statistic::mean);
    return stdDev;
}

//! @}

//! @defgroup user-functions User functions
//! @brief The functions that are meant to be called by the user
//!
//! Are wrappers for the core functions.
//! @{

//! @brief Wrapper function called by the user, choose which statistical method to use to calculate the
//! error on the average
//! @param energies The energies and positions, where only the energies will be used
//! @param function The desired statistical method the user wants to apply to Monte Carlo data
//! @param boostrapSamples The number of samples that will be generated
//! @param gen The random generator
//! @return The error on the average calculated with the desired statistical method
template <Dimension D, ParticNum N>
Energy ErrorOnAvg(std::vector<LocEnAndPoss<D, N>> const &energies, StatFuncType function,
                  IntType const &boostrapSamples, RandomGenerator &gen) {
    switch (function) {
    case StatFuncType::blocking:
        return BlockingAnalysis(energies);
        break;
    case StatFuncType::bootstrap:
        return BootstrapAnalysis(energies, boostrapSamples, gen);
        break;
    case StatFuncType::regular:
        return StdDev(energies);
    default:
        assert(false);
    }
}

//! @}

} // namespace vmcp

#endif
