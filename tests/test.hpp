//!
//! @file test.hpp
//! @brief Definition of constants and generic functions used in the tests
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!

#ifndef VMCP_TESTS_HPP
#define VMCP_TESTS_HPP

#include "types.hpp"

#include <execution>
#include <ranges>

// Chosen at random, but fixed to guarantee reproducibility of failed tests
constexpr vmcp::UIntType seed = 648265u;
const std::string logFilePath = "../artifacts/test-log.txt";
const std::string metrLogMes = "Metropolis algortihm";
const std::string impSampLogMes = "Importance sampling algorithm";
const std::string anDerLogMes = "analytical derivative";
const std::string numDerLogMes = "numerical derivative";

constexpr vmcp::IntType allowedStdDevs = 10;
// Maximum allowed discrepancy between the computed energy and the expected energy
constexpr vmcp::Energy vmcEnergyTolerance{1E-6f};
// Should have been constexpr, we just used const since otherwise the intellisense complains
// If the standard deviation is smaller than this, it is highly probable that numerical errors were
// non-negligible
const vmcp::Energy stdDevTolerance{std::numeric_limits<vmcp::FPType>::epsilon() * 1000};

constexpr vmcp::IntType iterations = 1 << 6;
// The denominator to obtain the number of iterations when the variational parameters are used
constexpr vmcp::IntType vpIterationsFactor = 1 << 4;
static_assert((iterations % vpIterationsFactor) == 0);

constexpr vmcp::IntType numEnergies = 1 << 9;
// The denominator to obtain the number of energies when the variational parameters are used
constexpr vmcp::IntType vpNumEnergiesFactor = 1 << 3;
static_assert((numEnergies % vpNumEnergiesFactor) == 0);

// The number of samples for bootsrapping technique of statistical analysis
constexpr vmcp::IntType bootstrapSamples = 10000;

// Denominator to obtain the derivative step from the length of the integration region
constexpr vmcp::FPType derivativeStepDenom = 100000;

// Number of randomly chosen points in the integration region where the potential is computed
constexpr vmcp::IntType points_peakSearch = 100;
//! @brief Minimal value of the wavefunction to accept the point as the new highest point of the potential
//! @see FindStartPoint
//!
//! Minimal value of the wavefunction to accept the point as the new peak of the potential.
//! Aviods situations where the wavefunction at the peak is 'nan', which breaks the update algorithms.
constexpr vmcp::FPType minWavef_peakSearch = 1e-6f;

//! @brief Finds a point where the potential is large
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param pot The potential
//! @param bounds The region in which the search for the peak will be done
//! @param numPoints How many points will be sampled in the search
//! @param gen The random generator
//! @return The positions of the peak
//!
//! Randomly chooses 'numPoints' points in the region and where the potential is largest, but the wavefunction
//! is not too small. The latter is done to avoid having the wavefunction be 'nan', which breaks the VMC
//! update algorithms.
template <vmcp::Dimension D, vmcp::ParticNum N, vmcp::VarParNum V, class Wavefunction, class Potential>
vmcp::Positions<D, N> FindPeak_(Wavefunction const &wavef, vmcp::VarParams<V> params, Potential const &pot,
                                vmcp::CoordBounds<D> bounds, vmcp::IntType numPoints,
                                vmcp::RandomGenerator &gen) {
    using namespace vmcp;

    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());
    assert(numPoints > 0);

    Position<D> center;
    std::transform(bounds.begin(), bounds.end(), center.begin(),
                   [](Bound<Coordinate> b) { return (b.upper + b.lower) / 2; });
    // FP TODO: Can you make it more elegant?
    Positions<D, N> result;
    std::fill(result.begin(), result.end(), center);
    std::uniform_real_distribution<FPType> unif(0, 1);
    std::mutex m;
    auto const indices = std::ranges::views::iota(0, numPoints);
    // FP TODO: Data race in unif(gen)
    // Maybe put another mutex? Deadlock risk?
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int) {
        Positions<D, N> newPoss;
        for (Position<D> &p : newPoss) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), [&unif, &gen](Bound<Coordinate> b) {
                return b.lower + (b.upper - b.lower) * unif(gen);
            });
        }
        // The requirement ... > minPsi avoids having wavef(...) = nan in the future, which breaks the update
        // algorithms
        {
            std::lock_guard<std::mutex> l(m);
            if ((pot(newPoss) > pot(result)) && (wavef(newPoss, params) > minWavef_peakSearch)) {
                result = newPoss;
            }
        }
    });
    return result;
}

#endif
