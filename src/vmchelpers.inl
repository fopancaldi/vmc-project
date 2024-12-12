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

//! @brief Finds a point where the potential is large
//! @param wavef The wavefunction
//! @param params The variational parameters
//! @param pot The potential
//! @param bounds The region in which the search for the starting point will be done
//! @param numPoints How many points will be sampled in the search
//! @param gen The random generator
//! @return The positions of the starting point
//!
//! Randomly chooses points in the region and where the potential is largest, but the wavefunction
//! is not too small. The latter is done to avoid having the wavefunction be 'nan', which breaks the VMC
//! update algorithms
//! Helper for 'VMCLocEnAndPoss'

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
Positions<D, N> FindStartPoint_(Wavefunction const &wavef, VarParams<V> params, CoordBounds<D> bounds,
                                RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    Positions<D, N> result;
    std::normal_distribution<FPType> norm(FPType{0.5f}, FPType{0.25f});

    // Initialize positions randomly within bounds
    do {
        for (Position<D> &p : result) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), [&norm, &gen](Bound<Coordinate> b) {
                Coordinate resultC = b.lower + b.Length() * norm(gen);
                while ((resultC < b.lower) || (resultC > b.upper)) {
                    resultC = b.lower + (b.upper - b.lower) * norm(gen);
                }
                return resultC;
            });
        }
    } while (wavef(result, params) < minWavef_peakSearch);

    FPType const paramsNorm =
        std::sqrt(std::inner_product(params.begin(), params.end(), params.begin(), FPType{0}, std::plus<>(),
                                     [](VarParam v1, VarParam v2) { return v1.val * v2.val; }));
    FPType gradientStep = paramsNorm / stepDenom_gradDesc;
    do {
        // Gradient descent
        for (int i = 0; i < maxLoops_gradDesc; ++i) {
            // The gradient descent should end in a reasonable time
            assert((i + 1) != maxLoops_gradDesc);

            Positions<D, N> gradient;
            for (auto &g : gradient) {
                g.fill(Coordinate{0}); // Initialize gradient to zero
            }

            // Compute gradient numerically for each particle in each dimension
            for (ParticNum n = 0; n < N; ++n) {
                for (Dimension d = 0; d < D; ++d) {
                    Positions<D, N> perturbedForward = result;
                    Positions<D, N> perturbedBackward = result;

                    perturbedForward[n][d].val += gradientStep;
                    perturbedBackward[n][d].val -= gradientStep;

                    FPType wavefForward = wavef(perturbedForward, params);
                    FPType wavefBackward = wavef(perturbedBackward, params);

                    // Central difference to estimate gradient
                    gradient[n][d].val = (wavefForward - wavefBackward) / (2 * gradientStep);
                }
            }

            // Apply gradient to update positions
            FPType gradNorm{0};
            for (ParticNum n = 0; n < N; ++n) {
                for (Dimension d = 0; d < D; ++d) {
                    gradNorm += gradient[n][d].val * gradient[n][d].val;
                    result[n][d].val += gradientStep * gradient[n][d].val;
                }
            }

            // Normalize step size and enforce bounds
            if (std::sqrt(gradNorm) < stoppingThreshold_gradDesc) {
                break;
            }
            for (Position<D> &p : result) {
                std::transform(bounds.begin(), bounds.end(), p.begin(), p.begin(),
                               [](Bound<Coordinate> b, Coordinate c) {
                                   return std::clamp<Coordinate>(c, b.lower, b.upper);
                               });
            }
        }
    } while (wavef(result, params) > minWavef_peakSearch);

    std::cout << "\nFinished start point search using gradient descent\n";
    return result;
}
/*template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
Positions<D, N> FindStartPoint_(Wavefunction const &wavef, VarParams<V> params, CoordBounds<D> bounds,
                                RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    Positions<D, N> result;
    std::normal_distribution<FPType> norm(FPType{0.5f}, FPType{0.25f});

    do {
        for (Position<D> &p : result) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), [&norm, &gen](Bound<Coordinate> b) {
                Coordinate resultC = b.lower + b.Length() * norm(gen);
                // LF FP TODO: define <= , >= operators for coordinate
                while ((resultC.val < b.lower.val) || (resultC.val > b.upper.val)) {
                    resultC = b.lower + (b.upper - b.lower) * norm(gen);
                }
                return resultC;
            });
        }

    } while (wavef(result, params) < minWavef_peakSearch);

    std::uniform_real_distribution<FPType> unif(0, 1);

    do {
        for (Position<D> &p : result) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), p.begin(),
                           [&unif, &gen](Bound<Coordinate> b, Coordinate c) {
                               // LF TODO: Replace with grad descent
                               return c + unif(gen) * b.Length() / 1000;
                           });
        }

    } while (wavef(result, params) > minWavef_peakSearch);

    std::cout << "\nFinished start point search \n";
    return result;
}*/

//! @defgroup update-algs Update algorithms
//! @brief The algorithms that move the particles during the simulations
//! @{

//! @defgroup update-algs-helpers Update helpers
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
std::array<std::array<FPType, D>, N> DriftForceNumeric_(Wavefunction const &wavef, Positions<D, N> poss,
                                                        VarParams<V> params, FPType derivativeStep) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    std::array<std::array<FPType, D>, N> driftForce;
    // FP TODO: Can you parallelize this?
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            driftForce[n][d] = 2 *
                               (wavef(MoveBy_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                                wavef(MoveBy_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                               (derivativeStep * wavef(poss, params));
        }
    }
    return driftForce;
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

        std::array<std::array<FPType, D>, N> newDriftForce;
        if (useAnalytical) {
            newDriftForce = DriftForceAnalytic_<D, N, V>(wavef, poss, params, grads);
        } else {
            newDriftForce = DriftForceNumeric_<D, N, V>(wavef, poss, params, derivativeStep);
        }
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
std::array<Energy, V> ReweightedEnergies_(Wavefunction const &wavef, VarParams<V> oldParams,
                                          std::vector<LocEnAndPoss<D, N>> oldLEPs, FPType step) {
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

//! @}

} // namespace vmcp

#endif
