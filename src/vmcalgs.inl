//!
//! @file vmcalgs.inl
//! @brief Definition of the templated VMC algorithms
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the templated VMC algorithms declared in the header.
//! Among the helper functions used in said definitions, the templated ones are also defined here, while the
//! non-templated ones are in the .inl file.
//! @see vmcalgs.hpp
//!

// TODO: remove all those damn unserscores
// The functions that have a trailing underscore would, in a regular project, be (declared and) defined in a
// .cpp file, therefore making them unreachable for the user
// Since they are templated, they must be defined in a header

// FP TODO: For example in FindPeak_, is it really necessary to specify Position<D>?
// In general, study class template argument deduction
// Also I believe you are putting too many (), for example: in assert((i / 2) == 3), are the brackets
// necessary? Study

#ifndef VMCPROJECT_VMCALGS_INL
#define VMCPROJECT_VMCALGS_INL

#include "statistics.hpp"
#include "vmcalgs.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <ranges>

namespace vmcp {

//! @defgroup algs-constants Constants
//! @brief Constants used in the algorithms and/or the helper functions
//!
//! Constants used in the algorithms.
//! They are named with the convention 'constantName_algorithmThatUsesIt'.
//! @{

//! @brief Reduced Planck constant
constexpr FPType hbar = 1;
//! @brief Number of randomly chosen points in the integration region where the potential is computed
//! @see FindPeak
constexpr IntType points_peakSearch = 100;
//! @brief Minimal value of the wavefunction to accept the point as the new highest point of the potential
//! @see FindPeak
//!
//! Minimal value of the wavefunction to accept the point as the new peak of the potential.
//! Aviods situations where the wavefunction at the peak is 'nan', which breaks the update algorithms.
constexpr FPType minWavef_peakSearch = 1e-6f;
//! @brief Maximum number of iterations of the gradient descent
//! @see BestParams
//!
//! Maximum number of iterations of the gradient descent.
//! Should never be reached, unless the parameters are very large.
constexpr IntType maxLoops_gradDesc = 100000;
//! @brief The norm of the initial parameters divided by this gives the starting step for the gradient descent
constexpr IntType stepDenom_gradDesc = 100;
//! @brief When the gradient divided by the parameters' norm is smaller than this, stop the gradient descent
constexpr FPType stoppingThreshold_gradDesc = 1e-9f;
//! @brief Number of independent gradient descents carried out simultaneously
constexpr IntType numWalkers_gradDesc = 8;
//! @brief Fraction of the current energy that can be gained at most in a gradient descent move
constexpr FPType increaseFrac_gradDesc = 0.1f;
// FP TODO: Rename this one, and document
constexpr IntType stepDenom_vmcLEPs = 100;
//! @brief Number of updates after which the sampled local energies are uncorrelated
//! @see VMCLocEnAndPoss
constexpr IntType autocorrelationMoves_vmcLEPs = 100;
//! @brief Number of moves after which the system has forgot about its initial conditions
//! @see VMCLocEnAndPoss
constexpr IntType movesForgetICs_vmcLEPs = 10 * autocorrelationMoves_vmcLEPs;
//! @brief Optimal acceptance rate for the updates in the VMC algorithm
constexpr FPType targetAcceptRate_vmcLEPs = 0.5f;
// LF TODO: Rename this one, and document
// Also, is there a better criterion to choose this than to simply fix it (which is a bad idea since it might
// turn out to be too large for some programs)?
// I propose the following criterion: ask for 'step' as an input too (like MetropolisUpdate_ does), and choose
// deltaT such that on average the random part of the jump has length 'step'
constexpr FPType deltaT = 0.005f;

//! @}

//! @defgroup helpers Helpers
//! @brief Help the core functions.
//! @{

//! @brief Calculates the mean and its error (by taking just one standard deviation)
//! @param v The energies and positions, where only the energies will be averaged
//! @return The mean and its (!= the) standard deviation
template <Dimension D, ParticNum N>
VMCResult MeanAndErr_(std::vector<LocEnAndPoss<D, N>> const &v) {
    Energy const mean = Mean(v);
    Energy const stdDev = StdDev(v);
    return VMCResult{mean, stdDev};
}

//! @brief Moves one particle in a cardinal direction
//! @param poss The positions of the particles
//! @param d The index of the cardinal direction in which the particle will be moved
//! @param n The index of the particle that will be moved
//! @param delta How much the particle will be moved
//! @return The updated positions
template <Dimension D, ParticNum N>
Positions<D, N> MoveBy_(Positions<D, N> const &poss, Dimension d, ParticNum n, Coordinate delta) {
    assert(d < D);
    assert(n < N);
    Positions<D, N> result = poss;
    result[n][d] += delta;
    return result;
}

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
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Positions<D, N> FindPeak_(Wavefunction const &wavef, VarParams<V> params, Potential const &pot,
                          CoordBounds<D> bounds, IntType numPoints, RandomGenerator &gen) {
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

//! @defgroup update-algs Update algorithms
//! @brief The algorithms that move the particles during the simulations
//! @{

//! @defgroup update-algs-helpers UA Helpers
//! @brief Help the update algorithms
//! @{

// LF TODO: Document
// Computes the drift force by using its analytic expression
template <Dimension D, ParticNum N, VarParNum V, class FirstDerivative, class Wavefunction>
std::array<std::array<FPType, D>, N> DriftForceAnalytic_(Wavefunction const &wavef, Positions<D, N> poss,
                                                         VarParams<V> params,
                                                         Gradients<D, N, FirstDerivative> const &grads) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    std::array<std::array<FPType, D>, N> gradsVal;
    std::array<std::array<FPType, D>, N> result;
    for (ParticNum n = 0u; n != N; ++n) {
        std::transform(grads[n].begin(), grads[n].end(), gradsVal[n].begin(),
                       [&poss, params](FirstDerivative const &fd) { return fd(poss, params); });
        std::transform(gradsVal[n].begin(), gradsVal[n].end(), result[n].begin(),
                       [&wavef, &poss, params](FPType f) { return 2 * f / wavef(poss, params); });
    }

    return result;
}

// LF TODO: Document
// Computes the drift force by numerically estimating the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
std::array<std::array<FPType, D>, N> DriftForceNumeric_(Wavefunction const &wavef, VarParams<V> params,
                                                        FPType derivativeStep, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    std::array<FPType, D> driftForce;
    // FP TODO: Can you parallelize this?
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            driftForce[n][d] = 2 *
                               (wavef(MoveBy_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                                wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                               (derivativeStep * wavef(poss, params));
        }
    }
}

//! @}

//! @brief Attempts to update each position once by using the Metropolis algorithm
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param poss The current positions of the particles, will be modified if some updates succeed
//! @param step The step size of th jump
//! @param gen The random generator
//! @return The number of succesful updates
//!
//! Attempts to update the position of each particle once, sequentially.
//! An update consists in a random jump in each cardinal direction, after which the Metropolis question is
//! asked.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
IntType MetropolisUpdate_(Wavefunction const &wavef, VarParams<V> params, Positions<D, N> &poss, FPType step,
                          RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    IntType succesfulUpdates = 0;
    for (Position<D> &p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = wavef(poss, params);
        std::uniform_real_distribution<FPType> unif(0, 1);
        std::transform(p.begin(), p.end(), p.begin(), [&gen, &unif, step](Coordinate c) {
            // FP TODO: Convert step to Coordinate?
            return c + Coordinate{(unif(gen) - FPType{0.5f}) * step};
        });
        if (unif(gen) < std::pow(wavef(poss, params) / oldPsi, 2)) {
            ++succesfulUpdates;
        } else {
            p = oldPos;
        }
    }
    return succesfulUpdates;
}

// LF TODO: Document
// LF TODO: Tell that this functions applies formula at page 22 Jensen
// LF TODO: DriftForceNumeric_ is not used here!
//  Updates the wavefunction with the importance sampling algorithm and outputs the number of succesful
//  updates
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative>
IntType ImportanceSamplingUpdate_(Wavefunction const &wavef, VarParams<V> params,
                                  Gradients<D, N, FirstDerivative> const &grads, Masses<N> masses,
                                  Positions<D, N> &poss, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    std::array<FPType, N> diffConsts;
    std::transform(masses.begin(), masses.end(), diffConsts.begin(),
                   [](Mass m) { return hbar * hbar / (2 * m.val); });

    IntType successfulUpdates = 0;
    for (ParticNum n = 0u; n != N; ++n) {
        Position<D> &p = poss[n];
        Position const oldPos = p;
        FPType const oldPsi = wavef(poss, params);
        std::array<std::array<FPType, D>, N> const oldDriftForce =
            DriftForceAnalytic_<D, N, V>(wavef, poss, params, grads);
        std::normal_distribution<FPType> normal(0, diffConsts[n] * deltaT);
        for (Dimension d = 0u; d != D; ++d) {
            p[d].val = oldPos[d].val + diffConsts[n] * deltaT * (oldDriftForce[n][d] + normal(gen));
        }
        FPType const newPsi = wavef(poss, params);

        FPType forwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            forwardExponent -=
                std::pow(p[d].val - oldPos[d].val - diffConsts[n] * deltaT * oldDriftForce[n][d], 2) /
                (4 * diffConsts[n] * deltaT);
        }
        FPType const forwardProb = std::exp(forwardExponent);

        std::array<std::array<FPType, D>, N> const newDriftForce =
            DriftForceAnalytic_<D, N, V>(wavef, poss, params, grads);
        FPType backwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            backwardExponent -=
                std::pow(oldPos[d].val - p[d].val - diffConsts[n] * deltaT * newDriftForce[n][d], 2) /
                (4 * diffConsts[n] * deltaT);
        }
        FPType const backwardProb = std::exp(backwardExponent);

        FPType const acceptanceRatio = (newPsi * newPsi * backwardProb) / (oldPsi * oldPsi * forwardProb);
        std::uniform_real_distribution<FPType> unif(0, 1);
        if (unif(gen) < acceptanceRatio) {
            ++successfulUpdates;
        } else {
            p = oldPos;
        }
    }
    return successfulUpdates;
}

//! @}

//! @addtogroup helpers
//! @{

//! @brief Computes the local energy by using the analytic formula for the derivative of the wavefunction
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param lapls The laplacians, one for each particle
//! @param masses The masses of the particles
//! @param pot The potential
//! @param poss The positions of the particles
//! @return The local energy
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Laplacian, class Potential>
Energy LocalEnergyAnalytic_(Wavefunction const &wavef, VarParams<V> params,
                            Laplacians<N, Laplacian> const &lapls, Masses<N> masses, Potential const &pot,
                            Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, Laplacian>());
    static_assert(IsPotential<D, N, Potential>());

    FPType weightedLaplSum = 0;
    // FP TODO: Bad indices, bad!
    for (ParticNum n = 0u; n != N; ++n) {
        weightedLaplSum += lapls[n](poss, params) / masses[n].val;
    }
    return Energy{-(hbar * hbar / 2) * (weightedLaplSum / wavef(poss, params)) + pot(poss)};
}

//! @brief Computes the local energy by numerically estimating the derivative of the wavefunction
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param derivativeStep The step used is the numerical estimation of the derivative
//! @param masses The masses of the particles
//! @param pot The potential
//! @param poss The positions of the particles
//! @return The local energy
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Energy LocalEnergyNumeric_(Wavefunction const &wavef, VarParams<V> params, FPType derivativeStep,
                           Masses<N> masses, Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());

    Energy result{pot(poss)};
    // FP TODO: Can you parallelize this?
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            result.val += -std::pow(hbar, 2) / (2 * masses[n].val) *
                          (wavef(MoveBy_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                           2 * wavef(poss, params) +
                           wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                          std::pow(derivativeStep, 2);
        }
    }
    return result;
}

//! @brief Computes the mean energy by using the reweighting method, after moving one parameter in a cardinal
//! direction
//! @param wavef The wavefunction
//! @param oldParams The variational parameters
//! @param oldLEPs The local energies to be reweighted, and the positions of the particles when each one was
//! computed
//! @param step How much one parameter should be moved
//! @return The mean energy after reweighting
//!
//! Used to compute the gradient of the VMC energy in parameter space
//! @see VMCRBestParams
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
std::array<Energy, D> ReweightedEnergies_(Wavefunction const &wavef, VarParams<V> oldParams,
                                          std::vector<LocEnAndPoss<N, D>> oldLEPs, FPType step) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    // FP TODO: can you use std::transform here?
    std::array<Energy, V> result;
    for (UIntType v = 0u; v != V; ++v) {
        VarParams<V> newParams = oldParams;
        newParams[v].val += step;
        std::vector<Energy> reweightedLocalEnergies(oldLEPs.size());
        std::transform(
            std::execution::par_unseq, oldLEPs.begin(), oldLEPs.end(), reweightedLocalEnergies.begin(),
            [&wavef, newParams, oldParams](LocEnAndPoss<D, N> const &lep) {
                return Energy{std::pow(wavef(lep.positions, newParams) / wavef(lep.positions, oldParams), 2) *
                              lep.localEn.val};
            });

        FPType const numerator =
            std::accumulate(reweightedLocalEnergies.begin(), reweightedLocalEnergies.end(), FPType{0},
                            [](FPType f, Energy e) { return f + e.val; });
        FPType const denominator = std::accumulate(
            oldLEPs.begin(), oldLEPs.end(), FPType{0},
            [&wavef, newParams, oldParams](FPType f, LocEnAndPoss<D, N> const &lep) {
                return f + std::pow(wavef(lep.positions, newParams) / wavef(lep.positions, oldParams), 2);
            });

        result[v] = Energy{numerator / denominator};
    }
    return result;
}

//! @}

//! @}

//! @defgroup core-functions Core functions
//! @brief The most important functions in the code
//!
//! The ones that actually do the work.
//! @{

// TODO: Whether ... is ...?
// Is this the correct logical construct?
// FP TODO: rename numberEnergies -> numEnergies

//! @brief Computes the energies that will be averaged to obtain the estimate of the energy
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param useAnalytical Whether the local energy must be computed by using the analytical expression of the
//! laplacians
//! @param useImpSamp Whether to use importance sampling as the update algorithm (the alternative is
//! Metropolis)
//! @param grads The gradients of the particles (unused if 'useAnalytical == false' or 'useImpSamp == false')
//! @param lapls The laplacians of the particles (unused if 'useAnalytical == false')
//! @param derivativeStep The step used is the numerical estimation of the derivative (unused if
//! 'useAnalytical == true')
//! @param masses The masses of the particles
//! @param pot The potential
//! @param bounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Is the most important function of the library, since is the one which actually does the work.
//! Starts from a point where the potential is sufficiently large, to quickly forget about the initial
//! conditions. Does some updates to move away from the peak, then starts computing the local energies. In
//! between two evaluations of the local energy, some updates are done to avoid correlations. Adjusts the step
//! size on the fly to best match the target acceptance rate.
//! Depending on 'useAnalytical' and 'useImpSamp', some parameters are unused.
//! To avoid having the user supply some parameters he does not care about, wrappers that only ask for the
//! necessary ones are provided.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative, class Laplacian,
          class Potential>
std::vector<LocEnAndPoss<D, N>>
VMCLocEnAndPoss_(Wavefunction const &wavef, VarParams<V> params, bool useAnalytical, bool useImpSamp,
                 Gradients<D, N, FirstDerivative> const &grads, Laplacians<N, Laplacian> const &lapls,
                 FPType derivativeStep, Masses<N> masses, Potential const &pot, CoordBounds<D> bounds,
                 IntType numberEnergies, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());
    static_assert(IsWavefunctionDerivative<D, N, V, Laplacian>());
    static_assert(IsPotential<D, N, Potential>());
    assert(numberEnergies > 0);

    // Find a good starting point, in the sense that it's easy to move away from
    Positions<D, N> const peak = FindPeak_<D, N>(wavef, params, pot, bounds, points_peakSearch, gen);

    // Choose the initial step
    Bound const smallestBound =
        *(std::min_element(bounds.begin(), bounds.end(), [](Bound<Coordinate> b1, Bound<Coordinate> b2) {
            return b1.Length().val < b2.Length().val;
        }));
    FPType step = smallestBound.Length().val / stepDenom_vmcLEPs;

    // Save the energies to be averaged
    Positions<D, N> poss = peak;
    std::function<Energy()> localEnergy;
    if (useAnalytical) {
        localEnergy = std::function<Energy()>{
            [&]() { return LocalEnergyAnalytic_<D, N>(wavef, params, lapls, masses, pot, poss); }};
    } else {
        localEnergy = std::function<Energy()>{
            [&]() { return LocalEnergyNumeric_<D, N>(wavef, params, derivativeStep, masses, pot, poss); }};
    }
    std::function<IntType()> update;
    if (useImpSamp) {
        update = std::function<IntType()>{
            [&]() { return ImportanceSamplingUpdate_<D, N>(wavef, params, grads, masses, poss, gen); }};
    } else {
        update = std::function<IntType()>{
            [&]() { return MetropolisUpdate_<D, N>(wavef, params, poss, step, gen); }};
    }

    std::vector<LocEnAndPoss<D, N>> result;
    result.reserve(static_cast<long unsigned int>(numberEnergies));
    // Move away from the peak, in order to forget the dependence on the initial conditions
    for (IntType i = 0; i != movesForgetICs_vmcLEPs; ++i) {
        update();
    }
    for (IntType i = 0; i != numberEnergies; ++i) {
        IntType succesfulUpdates = 0;
        for (IntType j = 0; j != autocorrelationMoves_vmcLEPs; ++j) {
            succesfulUpdates += update();
        }
        result.emplace_back(localEnergy(), poss);

        // Adjust the step size
        // Call car = current acc. rate, tar = target acc. rate
        // Add (car - tar)/tar to step, since it increases step if too many moves were accepted and decreases
        // it if too few were accepted
        FPType currentAcceptRate = succesfulUpdates * FPType{1} / (autocorrelationMoves_vmcLEPs * N);
        step *= currentAcceptRate / targetAcceptRate_vmcLEPs;
    }

    return result;
}

// FP TODO: step should be a var. param.

//! @brief Computes the energy with error, with the parameters that minimize the former
//! @param initialParams The initial variational parameters
//! @param wavef The wavefunction
//! @param lepsCalc A function that takes as input the variational parameters and returns the local energies
//! and the positions of the particles when each one was computed
//! @return The energy with error
//!
//! Does gradient descent starting from the given parameters.
//! Stops when the proposed step is too small compared to the current parameters.
//! Computes the gradient by using reweighting.
//! After having computed the gradient, if the proposed step would increase the energy too much, proposes a
//! new step of half the length, and repeats.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class LocEnAndPossCalculator>
VMCResult VMCRBestParams_(VarParams<V> initialParams, Wavefunction const &wavef,
                          LocEnAndPossCalculator const &lepsCalc) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(
        std::is_invocable_r_v<std::vector<LocEnAndPoss<D, N>>, LocEnAndPossCalculator, VarParams<V>>);
    static_assert(V != UIntType{0u});

    // Use a gradient descent algorithm with termination condition: stop if the next step is too small
    // compared to the current parameters
    VMCResult result;
    VarParams<V> currentParams = initialParams;
    FPType const initialParamsNorm = std::sqrt(
        std::inner_product(initialParams.begin(), initialParams.end(), initialParams.begin(), FPType{0},
                           std::plus<>(), [](VarParam v1, VarParam v2) { return v1.val * v2.val; }));
    FPType gradientStep = initialParamsNorm / stepDenom_gradDesc;

    for (IntType i = 0; i != maxLoops_gradDesc; ++i) {
        std::vector<LocEnAndPoss<D, N>> const currentEnAndPoss = lepsCalc(currentParams);
        VMCResult const currentVMCR = MeanAndErr_(currentEnAndPoss);
        result = currentVMCR;

        // Compute the gradient by using reweighting
        std::array<Energy, V> energiesIncreasedParam =
            ReweightedEnergies_<D, N, V>(wavef, currentParams, currentEnAndPoss, gradientStep);
        std::array<Energy, V> energiesDecreasedParam =
            ReweightedEnergies_<D, N, V>(wavef, currentParams, currentEnAndPoss, -gradientStep);
        std::array<FPType, V> gradient;
        for (VarParNum v = 0u; v != V; ++v) {
            FPType const component =
                (energiesIncreasedParam[v].val - energiesDecreasedParam[v].val) / (2 * gradientStep);
            assert(!std::isnan(component));
            gradient[v] = component;
        }

        // Set as next step used to compute the gradient the current gradient norm, which is also the size of
        // the step if that step is accepted
        FPType const oldParamsNorm =
            std::sqrt(std::accumulate(currentParams.begin(), currentParams.end(), FPType{0},
                                      [](FPType f, VarParam v) { return f + v.val * v.val; }));
        gradientStep =
            std::sqrt(std::inner_product(gradient.begin(), gradient.end(), gradient.begin(), FPType{0}));
        if ((gradientStep / oldParamsNorm) < stoppingThreshold_gradDesc) {
            break;
        }

        // In order to move, try smaller and smaller steps until the new parameters produce a VMC energy not
        // much larger than the current one
        FPType gradMultiplier = 2;
        VarParams<V> newParams;
        VMCResult newVMCR;
        do {
            gradMultiplier /= 2;
            for (VarParNum v = 0u; v != V; ++v) {
                newParams[v].val = currentParams[v].val - gradMultiplier * gradient[v];
                newVMCR = MeanAndErr_(lepsCalc(newParams));
            }

        } while (newVMCR.energy.val >
                 (currentVMCR.energy.val + increaseFrac_gradDesc * std::abs(currentVMCR.energy.val)));
        currentParams = newParams;

        // The gradient descent should end in a reasonable time
        assert((i + 1) != maxLoops_gradDesc);
    }

    return result;
}

//! @brief Computes the energy with error, with the parameters that minimize the former
//! @param bounds The interval in which the best parameters should be found
//! @param wavef The wavefunction
//! @param lepsCalc A function that takes as input the variational parameters and returns the local energies
//! and the positions of the particles when each one was computed
//! @param numWalkers The number of independent gradient descents carried out
//! @param gen The random generator
//! @return The energy with error
//!
//! Carries out 'numWalkers' gradient descents in parallel, and at the end chooses the lowest energy obtained.
//! The starting parameters of the walkers are chosen randomly inside 'bounds'
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class LocEnAndPossCalculator>
VMCResult VMCRBestParams_(ParamBounds<V> bounds, Wavefunction const &wavef,
                          LocEnAndPossCalculator const &lepsCalc, IntType numWalkers, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(
        std::is_invocable_r_v<std::vector<LocEnAndPoss<D, N>>, LocEnAndPossCalculator, VarParams<V>>);
    assert(numWalkers > IntType{0});

    if constexpr (V == VarParNum{0u}) {
        VarParams<0u> const fakeParams{};
        return MeanAndErr_(lepsCalc(fakeParams));
    } else {
        // FP TODO: Data race unif(gen)
        std::uniform_real_distribution<FPType> unif(0, 1);
        std::vector<VMCResult> vmcResults(static_cast<long unsigned int>(numWalkers));
        std::generate_n(std::execution::par_unseq, vmcResults.begin(), numWalkers_gradDesc, [&]() {
            VarParams<V> initialParams;
            for (VarParNum v = 0u; v != V; ++v) {
                initialParams[v] = bounds[v].lower + bounds[v].Length() * unif(gen);
            }
            return VMCRBestParams_<D, N, V>(initialParams, wavef, lepsCalc);
        });

        return *std::min_element(vmcResults.begin(), vmcResults.end(), [](VMCResult vmcr1, VMCResult vmcr2) {
            return vmcr1.energy < vmcr2.energy;
        });
    }
}

//! @}

//! @defgroup user-functions User functions
//! @brief The functions that are meant to be called by the user
//!
//! Are wrappers for the core functions.
//! @{

//! @brief Computes the energies that will be averaged by using the analytical formula for the derivative and
//! the Metropolis algorithm
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param lapls The laplacians of the particles
//! @param masses The masses of the particles
//! @param pot The potential
//! @param bounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Wrapper for the true 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Laplacian, class Potential>
std::vector<LocEnAndPoss<D, N>> VMCLocEnAndPoss(Wavefunction const &wavef, VarParams<V> params,
                                                Laplacians<N, Laplacian> const &lapls, Masses<N> masses,
                                                Potential const &pot, CoordBounds<D> bounds,
                                                IntType numberEnergies, RandomGenerator &gen) {
    struct FakeDeriv {
        FPType operator()(Positions<D, N> const &, VarParams<V>) const {
            assert(false);
            return 0;
        }
    };
    Gradients<D, N, FakeDeriv> fakeGrads;
    FPType const fakeStep = std::numeric_limits<FPType>::quiet_NaN();
    return VMCLocEnAndPoss_<D, N, V>(wavef, params, true, false, fakeGrads, lapls, fakeStep, masses, pot,
                                     bounds, numberEnergies, gen);
}

//! @brief Computes the energy with error, by using the analytical formula for the derivative and the
//! Metropolis algorithm, after finding the best parameter
//! @param wavef The wavefunction
//! @param parBounds The interval in which the best parameters should be found
//! @param lapls The laplacians of the particles
//! @param masses The masses of the particles
//! @param pot The potential
//! @param coorBounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Wrapper for 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Laplacian, class Potential>
VMCResult VMCEnergy(Wavefunction const &wavef, ParamBounds<V> parBounds,
                    Laplacians<N, Laplacian> const &lapls, Masses<N> masses, Potential const &pot,
                    CoordBounds<D> coorBounds, IntType numberEnergies, RandomGenerator &gen) {
    auto const enPossCalculator{[&](VarParams<V> vps) {
        return VMCLocEnAndPoss<D, N, V>(wavef, vps, lapls, masses, pot, coorBounds, numberEnergies, gen);
    }};
    return VMCRBestParams_<D, N, V>(parBounds, wavef, enPossCalculator, numWalkers_gradDesc, gen);
}

//! @brief Computes the energies that will be averaged by using the analytical formula for the derivative and
//! the importance sampling algorithm
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param grads The gradients of the particles
//! @param lapls The laplacians of the particles
//! @param masses The masses of the particles
//! @param pot The potential
//! @param bounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Wrapper for the true 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative, class Laplacian,
          class Potential>
std::vector<LocEnAndPoss<D, N>>
VMCLocEnAndPoss(Wavefunction const &wavef, VarParams<V> params, Gradients<D, N, FirstDerivative> const &grads,
                Laplacians<N, Laplacian> const &lapls, Masses<N> masses, Potential const &pot,
                CoordBounds<D> bounds, IntType numberEnergies, RandomGenerator &gen) {
    FPType const fakeStep = std::numeric_limits<FPType>::quiet_NaN();
    return VMCLocEnAndPoss_<D, N, V>(wavef, params, true, false, grads, lapls, fakeStep, masses, pot, bounds,
                                     numberEnergies, gen);
}

//! @brief Computes the energy with error, by using the analytical formula for the derivative and the
//! importance sampling algorithm, after finding the best parameter
//! @param wavef The wavefunction
//! @param parBounds The interval in which the best parameters should be found
//! @param grads The gradients of the particles
//! @param lapls The laplacians of the particles
//! @param masses The masses of the particles
//! @param pot The potential
//! @param coorBounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Wrapper for 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative, class Laplacian,
          class Potential>
VMCResult VMCEnergy(Wavefunction const &wavef, ParamBounds<V> parBounds,
                    Gradients<D, N, FirstDerivative> const &grads, Laplacians<N, Laplacian> const &lapls,
                    Masses<N> masses, Potential const &pot, CoordBounds<D> coorBounds, IntType numberEnergies,
                    RandomGenerator &gen) {
    auto const enPossCalculator{[&](VarParams<V> vps) {
        return VMCLocEnAndPoss<D, N, V>(wavef, vps, grads, lapls, masses, pot, coorBounds, numberEnergies,
                                        gen);
    }};
    return VMCRBestParams_<D, N, V>(parBounds, wavef, enPossCalculator, numWalkers_gradDesc, gen);
}

//! @brief Computes the energies that will be averaged by numerically estimating the derivative and using
//! either the Metropolis or the importance sampling algorithm
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param useImpSamp Whether to use importance sampling as the update algorithm (the alternative is
//! Metropolis)
//! @param derivativeStep The step used is the numerical estimation of the derivative
//! @param masses The masses of the particles
//! @param pot The potential
//! @param bounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Wrapper for the true 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<LocEnAndPoss<D, N>> VMCLocEnAndPoss(Wavefunction const &wavef, VarParams<V> params,
                                                bool useImpSamp, FPType derivativeStep, Masses<N> masses,
                                                Potential const &pot, CoordBounds<D> bounds,
                                                IntType numberEnergies, RandomGenerator &gen) {
    struct FakeDeriv {
        FPType operator()(Positions<D, N> const &, VarParams<V>) const {
            assert(false);
            return FPType{0};
        }
    };
    Gradients<D, N, FakeDeriv> fakeGrads;
    std::array<FakeDeriv, N> fakeLapls;
    return VMCLocEnAndPoss_<D, N, V>(wavef, params, false, useImpSamp, fakeGrads, fakeLapls, derivativeStep,
                                     masses, pot, bounds, numberEnergies, gen);
}

//! @brief Computes the energy with error, by numerically estimating the derivative and using either the
//! Metropolis or the importance sampling algorithm, after finding the best parameter
//! @param wavef The wavefunction
//! @param parBounds The interval in which the best parameters should be found
//! @param useImpSamp Whether to use importance sampling as the update algorithm (the alternative is
//! Metropolis)
//! @param masses The masses of the particles
//! @param pot The potential
//! @param coorBounds The integration region
//! @param numberEnergies The number of energies to compute
//! @param gen The random generator
//! @return The computed local energies, and the positions of the particles when each local energy was
//! computed
//!
//! Wrapper for 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult VMCEnergy(Wavefunction const &wavef, ParamBounds<V> parBounds, bool useImpSamp,
                    FPType derivativeStep, Masses<N> masses, Potential const &pot, CoordBounds<D> coorBounds,
                    IntType numberEnergies, RandomGenerator &gen) {
    auto const enPossCalculator{[&](VarParams<V> vps) {
        return VMCLocEnAndPoss<D, N, V>(wavef, vps, useImpSamp, derivativeStep, masses, pot, coorBounds,
                                        numberEnergies, gen);
    }};
    return VMCRBestParams_<D, N, V>(parBounds, wavef, enPossCalculator, numWalkers_gradDesc, gen);
}

//! @}

} // namespace vmcp

#endif
