//!
//! @file vmchelpers.inl
//! @brief Definition of the helpers of the core functions
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Separated to improve readability
//! @see vmcalgs.inl
//!

// TODO: remove all those damn unserscores
// The functions that have a trailing underscore would, in a regular project, be (declared and) defined in a
// .cpp file, therefore making them unreachable for the user
// Since they are templated, they must be defined in a header

// FP TODO: For example in FindPeak_, is it really necessary to specify Position<D>?
// In general, study class template argument deduction
// Also I believe you are putting too many (), for example: in assert((i / 2) == 3), are the brackets
// necessary? Study

#ifndef VMCPROJECT_VMCHELPERS_INL
#define VMCPROJECT_VMCHELPERS_INL

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

//! @defgroup core-helpers Core helpers
//! @brief Help the core functions
//! @{

//! @defgroup update-algs Update algorithms
//! @brief The algorithms that move the particles during the simulations
//! @{

//! @defgroup update-algs-helpers Update helpers
//! @brief Help the update algorithms
//! @{

//! @brief Computes the drift force by using its analytic expression
//! @param wavef The wavefunction
//! @param poss The current positions of the particles
//! @param params The variational parameters
//! @param grads The gradients of the wavefunction (one for each particle)
//! @return The drift force evaluated analytically
template <Dimension D, ParticNum N, VarParNum V, class FirstDerivative, class Wavefunction>
std::array<std::array<FPType, D>, N> DriftForceAnalytic_(Wavefunction const &wavef, Positions<D, N> poss,
                                                         VarParams<V> params,
                                                         Gradients<D, N, FirstDerivative> const &grads) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    std::array<std::array<FPType, D>, N> result;

    std::transform(std::execution::par_unseq, grads.begin(), grads.end(), result.begin(),
                   [&wavef, &poss, params](Gradient<D, FirstDerivative> const &g) {
                       std::array<FPType, D> result_;
                       std::transform(g.begin(), g.end(), result_.begin(),
                                      [&wavef, &poss, params](FirstDerivative const &fd) {
                                          return 2 * fd(poss, params) / wavef(poss, params);
                                      });
                       return result_;
                   });

    return result;
}

//! @brief Computes the drift force by numerically estimating the derivative of the wavefunction
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param step The step size of the jump in the numeric estimate of the derivative
//! @param poss The current positions of the particles
//! @return The drift force evaluated numerically
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
std::array<std::array<FPType, D>, N> DriftForceNumeric_(Wavefunction const &wavef, Positions<D, N> poss,
                                                        VarParams<V> params, FPType step) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    std::array<std::array<FPType, D>, N> result;
    auto const indices = std::ranges::views::iota(VarParNum{0u}, N);
    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(),
        [&wavef, &poss, params, step, &result](ParticNum n) {
            for (Dimension d = 0u; d != D; ++d) {
                // Numerical derivative correct up to O(step^9)
                result[n][d] =
                    2 *
                    (FPType{1} / 280 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-4 * step}), params) +
                     FPType{-4} / 105 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-3 * step}), params) +
                     FPType{1} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-2 * step}), params) +
                     FPType{-4} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-step}), params) +
                     FPType{4} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{step}), params) +
                     FPType{-1} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{2 * step}), params) +
                     FPType{4} / 105 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{3 * step}), params) +
                     FPType{-1} / 280 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{4 * step}), params)) /
                    (step * wavef(poss, params));
            }
        });

    return result;
}

//! @}

//! @brief Attempts to update each position once by using the Metropolis algorithm
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param poss The current positions of the particles, will be modified if some updates succeed
//! @param step The step size of the jump
//! @param gen The random generator
//! @return The number of successful updates
//!
//! Attempts to update the position of each particle once, sequentially.
//! An update consists in a random jump in each cardinal direction, after which the Metropolis question is
//! asked.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
IntType MetropolisUpdate_(Wavefunction const &wavef, VarParams<V> params, Positions<D, N> &poss, FPType step,
                          RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    // std::cout << "wavef: " << wavef(poss, params) << '\n';
    //  std::cout << "pos[0]: " << poss[0][0].val << '\n';
    assert(wavef(poss, params) > 1e-12);

    IntType succesfulUpdates = 0;
    for (Position<D> &p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = wavef(poss, params);
        std::uniform_real_distribution<FPType> unif(0, 1);
        std::transform(p.begin(), p.end(), p.begin(), [&gen, &unif, step](Coordinate c) {
            // FP TODO: Convert step to Coordinate?
            return c + Coordinate{(unif(gen) - FPType{0.5f}) * step};
        });
        std::cout << "frac: " << wavef(poss, params) / oldPsi << '\n';
        if (unif(gen) < std::pow(wavef(poss, params) / oldPsi, 2)) {
            ++succesfulUpdates;
        } else {
            p = oldPos;
        }
    }
    return succesfulUpdates;
}

//! @brief Attempts to update each position once by using the Importance Sampling algorithm
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param useAnalytical Whether the drift force must be computed by using the analytical expression of the
//! gardients
//! @param derivativeStep The step used is the numerical estimation of the drift force derivatives (unused if
//! 'useAnalytical == true')
//! @param grads The gradients of the wavefunction (one for each particle)
//! @param masses The masses of the particles
//! @param poss The current positions of the particles, will be modified if some updates succeed
//! @param step The average length of the random part of the jump in changing positions
//! @param gen The random generator
//! @return The number of successful updates
//!
//! This function applies formulas in the end of section 1.4.3 of Nuclear Many-body Physics -
//! a Computational Approach - Monte Carlo methods, Morten Hjorth-Jensen.
//! It attempts to update the position of each particle once, sequentially.
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative>
IntType ImportanceSamplingUpdate_(Wavefunction const &wavef, VarParams<V> params, bool useAnalytical,
                                  FPType derivativeStep, Gradients<D, N, FirstDerivative> const &grads,
                                  Masses<N> masses, Positions<D, N> &poss, RandomGenerator &gen) {
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

        std::array<std::array<FPType, D>, N> oldDriftForce;
        if (useAnalytical) {
            oldDriftForce = DriftForceAnalytic_<D, N, V>(wavef, poss, params, grads);
        } else {
            oldDriftForce = DriftForceNumeric_<D, N, V>(wavef, poss, params, derivativeStep);
        }

        // Jensen in his notes, section 1.4.3, suggests a value between 0.001 and 0.01
        FPType const timeStep = 0.005;

        /* // Variance is choosen such that the average length of the random part of the jump equals step
         std::normal_distribution<FPType> normal(
             0, std::pow(step / (4 * std::sqrt(std::numbers::pi_v<FPType>)), 1.f / 3));*/
        std::normal_distribution<FPType> normal(0, 1);

        for (Dimension d = 0u; d != D; ++d) {
            p[d].val = oldPos[d].val + diffConsts[n] * timeStep * oldDriftForce[n][d] +
                       normal(gen) * std::sqrt(timeStep);
        }

        FPType const newPsi = wavef(poss, params);

        FPType forwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            forwardExponent -=
                std::pow(p[d].val - oldPos[d].val - diffConsts[n] * timeStep * oldDriftForce[n][d], 2) /
                (4 * diffConsts[n] * timeStep);
        }
        FPType const forwardProb = std::exp(forwardExponent);

        std::array<std::array<FPType, D>, N> newDriftForce;
        if (useAnalytical) {
            newDriftForce = DriftForceAnalytic_<D, N, V>(wavef, poss, params, grads);
        } else {
            newDriftForce = DriftForceNumeric_<D, N, V>(wavef, poss, params, derivativeStep);
        }
        FPType backwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            backwardExponent -=
                std::pow(oldPos[d].val - p[d].val - diffConsts[n] * timeStep * newDriftForce[n][d], 2) /
                (4 * diffConsts[n] * timeStep);
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

//! @addtogroup core-helpers
//! @{

//! @defgroup energy-calc Local energy calculators
//! @brief The algorithms that calculate the local energy
//! @{

//! @defgroup energy-calc-helpers Local energy helpers
//! @brief The algorithms that calculate the local energy
//! @{

//! @brief Moves one particle in a cardinal direction
//! @param poss The positions of the particles
//! @param d The index of the cardinal direction in which the particle will be moved
//! @param n The index of the particle that will be moved
//! @param delta How much the particle will be moved
//! @return The updated positions
//!
//! Helper for 'LocalEnergyNumeric' and 'DriftForceNumeric'
template <Dimension D, ParticNum N>
Positions<D, N> MoveBy_(Positions<D, N> const &poss, Dimension d, ParticNum n, Coordinate delta) {
    assert(d < D);
    assert(n < N);
    Positions<D, N> result = poss;
    result[n][d] += delta;
    return result;
}

//! @}

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

    FPType const weightedLaplSum =
        std::inner_product(lapls.begin(), lapls.end(), masses.begin(), FPType{0}, std::plus<>(),
                           [&poss, params](Laplacian const &l, Mass m) { return l(poss, params) / m.val; });
    return Energy{-hbar * hbar * weightedLaplSum / (2 * wavef(poss, params)) + pot(poss)};
}

//! @brief Computes the local energy by numerically estimating the derivative of the wavefunction
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param step The step used is the numerical estimation of the derivative
//! @param masses The masses of the particles
//! @param pot The potential
//! @param poss The positions of the particles
//! @return The local energy
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Energy LocalEnergyNumeric_(Wavefunction const &wavef, VarParams<V> params, FPType step, Masses<N> masses,
                           Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());

    Energy result{pot(poss)};
    std::mutex m;
    auto const indices = std::ranges::views::iota(VarParNum{0u}, N);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](ParticNum n) {
        for (Dimension d = 0u; d != D; ++d) {
            // Numerical derivative correct up to O(step^9)
            Energy const temp =
                Energy{-hbar * hbar / (2 * masses[n].val) *
                       (FPType{-1} / 560 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-4 * step}), params) +
                        FPType{8} / 315 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-3 * step}), params) +
                        FPType{-1} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-2 * step}), params) +
                        FPType{8} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-step}), params) +
                        FPType{-205} / 72 * wavef(poss, params) +
                        FPType{8} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{step}), params) +
                        FPType{-1} / 5 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{2 * step}), params) +
                        FPType{8} / 315 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{3 * step}), params) +
                        FPType{-1} / 560 * wavef(MoveBy_<D, N>(poss, d, n, Coordinate{4 * step}), params)) /
                       (std::pow(step, 2) * wavef(poss, params))};

            {
                std::lock_guard<std::mutex> l(m);
                result += temp;
            }
        }
    });

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
std::array<Energy, V> ReweightedEnergies_(Wavefunction const &wavef, VarParams<V> oldParams,
                                          std::vector<LocEnAndPoss<D, N>> oldLEPs, FPType step) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    // FP TODO: step -> VarParam?
    std::array<Energy, V> result;
    std::generate_n(result.begin(), V, [&, v = VarParNum{0u}]() mutable {
        VarParams<V> newParams = oldParams;
        newParams[v] += VarParam{step};
        std::vector<Energy> reweightedLocEns(oldLEPs.size());
        std::transform(
            std::execution::par_unseq, oldLEPs.begin(), oldLEPs.end(), reweightedLocEns.begin(),
            [&wavef, newParams, oldParams](LocEnAndPoss<D, N> const &lep) {
                return Energy{std::pow(wavef(lep.positions, newParams) / wavef(lep.positions, oldParams), 2) *
                              lep.localEn.val};
            });
        std::vector<FPType> denomAddends(oldLEPs.size());
        std::transform(std::execution::par_unseq, oldLEPs.begin(), oldLEPs.end(), denomAddends.begin(),
                       [&wavef, newParams, oldParams](LocEnAndPoss<D, N> const &lep) {
                           return std::pow(wavef(lep.positions, newParams) / wavef(lep.positions, oldParams),
                                           2);
                       });

        Energy const num = std::reduce(std::execution::par_unseq, reweightedLocEns.begin(),
                                       reweightedLocEns.end(), Energy{0}, std::plus<>());
        FPType const den = std::reduce(std::execution::par_unseq, denomAddends.begin(), denomAddends.end(),
                                       FPType{0}, std::plus<>());

        ++v;
        return num / den;
    });
    return result;
}

//! @}

//! @}

//! @}

} // namespace vmcp

#endif
