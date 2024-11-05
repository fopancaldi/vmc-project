//
//
// Contains the definition of the templates declared in vmcalgs.hpp
// This file is supposed to be #included at the end of vmcalgs.hpp and nowhere else
// It is just a way to improve the readability of vmcalgs.hpp
//

#ifndef VMCPROJECT_VMCALGS_INL
#define VMCPROJECT_VMCALGS_INL

#include "vmcalgs.hpp"

#include <algorithm>
#include <cassert>

namespace vmcp {

// Constants
constexpr FPType hbar = 1; // Natural units
constexpr IntType numSamplesForPeakSearch = 100;
constexpr FPType deltaT = 0.005;
// TODO: Rename this one
constexpr IntType boundSteps = 100;
constexpr IntType thermalizationMoves = 100;
// TODO: Rename this one, also find if it is really 10 * thermalization moves
constexpr IntType movesForgetICs = 10 * thermalizationMoves;
// constexpr IntType vmcMoves = 10;
constexpr FPType minPsi = 1e-6f;

// Should be hidden from the user
// Computes the (approximate) peak of the potential
// In practice, choose some points in the integration domain and see where the potential is largest
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Positions<D, N> FindPeak_(Wavefunction const &psi, VarParams<V> params, Potential const &pot,
                          Bounds<D> bounds, IntType points, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());

    Position<D> center;
    std::transform(bounds.begin(), bounds.end(), center.begin(),
                   [](Bound b) { return Coordinate{(b.upper + b.lower) / 2}; });
    Positions<D, N> result;
    std::fill(result.begin(), result.end(), center);
    std::uniform_real_distribution<FPType> unif(0, 1);
    for (IntType i = 0; i != points; ++i) {
        Positions<D, N> newPoss;
        // TODO: Is it really necessary to specify Position<D>?
        for (Position<D> &p : newPoss) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), [&unif, &gen](Bound b) {
                return Coordinate{b.lower + unif(gen) * (b.upper - b.lower)};
            });
        }
        // TODO: Explain better
        // The requirement ... > minPsi avoids having psi(...) = nan, which breaks the Metropolis updates
        if ((pot(newPoss) > pot(result)) && (psi(newPoss, params) > minPsi)) {
            result = newPoss;
        }
    }
    return result;
}

// Should be hidden from the user
// Updates the wavefunction with the Metropolis algorithms and outputs the number of succesful updates
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
UIntType MetropolisUpdate_(Wavefunction const &psi, VarParams<V> params, Positions<D, N> &poss, FPType step,
                           RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    UIntType succesfulUpdates = 0u;
    for (Position<D> &p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        std::uniform_real_distribution<FPType> unifDist(0, 1);
        std::transform(p.begin(), p.end(), p.begin(), [&gen, &unifDist, step](Coordinate c) {
            return Coordinate{c.val + (unifDist(gen) - FPType{0.5f}) * step};
        });
        if (unifDist(gen) > std::pow(psi(poss, params) / oldPsi, 2)) {
            p = oldPos;

        } else {
            ++succesfulUpdates;
        }
    }
    return succesfulUpdates;
}

// Should be hidden from the user
// Computes the drift force using its analytic expression
template <Dimension D, ParticNum N, VarParNum V, class FirstDerivative, class Wavefunction>
std::array<FPType, D> DriftForceAnalytic_(Wavefunction const &psi, Positions<D, N> poss, VarParams<V> params,
                                          std::array<FirstDerivative, D> const &grad) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    std::array<FPType, D> gradVal;
    std::transform(grad.begin(), grad.end(), gradVal.begin(),
                   [&poss, params](FirstDerivative const &fd) { return fd(poss, params); });
    std::array<FPType, D> result;
    std::transform(gradVal.begin(), gradVal.end(), result.begin(),
                   [&psi, &poss, params](FPType f) { return 2 * f / psi(poss, params); });
    return result;
}

// Should be hidden from the user
// Computes the position of the particles when the n-th one is moved by delta along the d-th direction (with
// both n and d starting from 0)
template <Dimension D, ParticNum N>
Positions<D, N> PerturbPos_(Positions<D, N> const &poss, Dimension d, ParticNum n, Coordinate delta) {
    assert(d < D);
    assert(n < N);

    Positions<D, N> result = poss;
    // TODO: Maybe define operator+= for coordinates?
    result[n][d].val += delta.val;
    return result;
}

// Should be hidden from the user
// Computes the drift force by numerically estimating the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
std::array<FPType, D> DriftForceNumeric_(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep,
                                         Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    std::array<FPType, D> driftForce;
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            driftForce[d] = 2 *
                            (psi(PerturbPos_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                             psi(PerturbPos_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                            (derivativeStep * psi(poss, params));
        }
    }
}
// TODO: Tell that this functions applies formula at page 22 Jensen
//  Updates the wavefunction with the Importance Sampling algorithms and outputs the number of succesful
//  updates
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative>
IntType ImportanceSamplingUpdate_(Wavefunction const &psi, VarParams<V> params,
                                  std::array<FirstDerivative, D> const &grad, Mass mass,
                                  Positions<D, N> &poss, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    FPType const diffusionConst = hbar * hbar / (2 * mass.val);

    IntType successfulUpdates = 0u;
    for (Position<D> &p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        std::array<FPType, D> const oldDriftForce = DriftForceAnalytic_<D, N, V>(psi, poss, params, grad);
        std::normal_distribution<FPType> normalDist(0.f, 1.f);
        for (Dimension d = 0u; d != D; ++d) {
            p[d].val = oldDriftForce[d] * deltaT + normalDist(gen) * std::sqrt(deltaT);
        }
        FPType const newPsi = psi(poss, params);
        std::array<FPType, D> newDriftForce = DriftForceAnalytic_<D, N, V>(psi, poss, params, grad);
        FPType forwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            forwardExponent -=
                std::pow(p[d].val - oldPos[d].val - diffusionConst * deltaT * oldDriftForce[d], 2) /
                (4 * diffusionConst * deltaT);
        }
        FPType const forwardProb = std::exp(forwardExponent);
        FPType backwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            backwardExponent -=
                std::pow(oldPos[d].val - p[d].val - diffusionConst * deltaT * newDriftForce[d], 2) /
                (4 * diffusionConst * deltaT);
        }
        FPType const backwardProb = std::exp(backwardExponent);
        FPType const acceptanceRatio = (newPsi * newPsi * backwardProb) / (oldPsi * oldPsi * forwardProb);
        std::uniform_real_distribution<FPType> unifDist(0.f, 1.f);
        if (unifDist(gen) < acceptanceRatio) {
            ++successfulUpdates;
        } else {
            p = oldPos;
        }
    }
    return successfulUpdates;
}

// Should be hidden from the user
// Computes the local energy with the analytic formula for the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class SecondDerivative, class Potential>
Energy LocalEnergyAnalytic_(Wavefunction const &psi, VarParams<V> params, SecondDerivative const &secondDer,
                            Mass mass, Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, SecondDerivative>());
    static_assert(IsPotential<D, N, Potential>());

    return Energy{-(hbar * hbar / (2 * mass.val)) * (secondDer(poss, params) / psi(poss, params)) +
                  pot(poss)};
}

// Should be hidden from the user
// Computes the local energy by numerically estimating the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Energy LocalEnergyNumeric_(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep, Mass mass,
                           Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());

    Energy result{pot(poss)};
    // TODO: Maybe redo this without the indices? Open to suggestions
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            result.val += -std::pow(hbar, 2) / (2 * mass.val) *
                          (psi(PerturbPos_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                           2 * psi(poss, params) +
                           psi(PerturbPos_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                          std::pow(derivativeStep, 2);
        }
    }
    return result;
}

// Should be hidden from the user
// Computes the energies that will be averaged to obtain the estimate of the GS energy of the VMC algorithm
// by either using the analytical formula for the derivative or estimating it numerically
// Use the wrappers to select which method should be used
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
std::vector<Energy> WrappedVMCEnergies_(Wavefunction const &psi, VarParams<V> params, bool useAnalitycal,
                                        bool useImpSamp, std::array<FirstDerivative, D> const &grad,
                                        SecondDerivative const &secondDer, FPType derivativeStep, Mass mass,
                                        Potential const &pot, Bounds<D> bounds, int numberEnergies,
                                        RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());
    static_assert(IsWavefunctionDerivative<D, N, V, SecondDerivative>());
    static_assert(IsPotential<D, N, Potential>());
    assert(numberEnergies >= 0);

    std::vector<Energy> result;

    // Step 1: Find a good starting point, in the sense that it's easy to move away from
    Positions<D, N> const peak = FindPeak_<D, N>(psi, params, pot, bounds, numSamplesForPeakSearch, gen);

    // Step 2: Choose the step
    Bound const smallestBound = *(std::min_element(
        bounds.begin(), bounds.end(), [](Bound b1, Bound b2) { return b1.Length() < b2.Length(); }));
    FPType const step = smallestBound.Length() / boundSteps;

    // Step 3: Save the energies to be averaged
    Positions<D, N> poss = peak;
    // Move away from the peak, in order to forget the dependence on the initial conditions
    auto computeEnergy = [&]() {
        if (useAnalitycal) {
            result.push_back(LocalEnergyAnalytic_<D, N>(psi, params, secondDer, mass, pot, poss));
        } else {
            result.push_back(LocalEnergyNumeric_<D, N>(psi, params, derivativeStep, mass, pot, poss));
        }
    };
    auto importanceSamplingUpdate = [&]() {
        ImportanceSamplingUpdate_<D, N>(psi, params, grad, mass, poss, gen);
    };
    auto metropolisUpdate = [&]() { MetropolisUpdate_<D, N>(psi, params, poss, step, gen); };
    auto updateAndComputeEn = [&](auto &updateFunction) {
        // Initial moves to forget initial conditions
        for (int i = 0; i != movesForgetICs; ++i) {
            updateFunction();
        }
        // Main loop for calculating energies
        for (int i = 0; i != numberEnergies; ++i) {
            for (int j = 0; j != thermalizationMoves; ++j) {
                updateFunction();
            }
            computeEnergy();
        }
    };
    if (useImpSamp) {
        updateAndComputeEn(importanceSamplingUpdate);
    } else {
        updateAndComputeEn(metropolisUpdate);
    }

    // TODO: Step 4: Adjust the step size

    return result;
}

VMCResult AvgAndVar_(std::vector<Energy> const &v) {
    assert(v.size() > 1);
    // TODO: Maybe define operator+ for the energies?
    Energy const cumul = std::accumulate(v.begin(), v.end(), Energy{0},
                                         [](Energy e1, Energy e2) { return Energy{e1.val + e2.val}; });
    EnVariance const cumulSq =
        std::accumulate(v.begin(), v.end(), EnVariance{0},
                        [](EnVariance ev, Energy e) { return EnVariance{ev.val + e.val * e.val}; });
    auto const size = v.size();
    return VMCResult{cumul.val / size, (cumulSq.val / size - std::pow(cumul.val / size, 2)) / (size - 1)};
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &psi, VarParams<V> params, bool useImpSamp,
                                std::array<FirstDerivative, D> const &grad, SecondDerivative const &secondDer,
                                Mass mass, Potential const &pot, Bounds<D> bounds, int numberEnergies,
                                RandomGenerator &gen) {
    return WrappedVMCEnergies_<D, N, V>(psi, params, true, useImpSamp, grad, secondDer, 0, mass, pot, bounds,
                                        numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> params, bool useImpSamp,
                    std::array<FirstDerivative, D> const &grad, SecondDerivative const &secondDer, Mass mass,
                    Potential const &pot, Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    return AvgAndVar_(VMCEnergies<D, N, V>(psi, params, useImpSamp, grad, secondDer, mass, pot, bounds,
                                           numberEnergies, gen));
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &psi, VarParams<V> params, bool useImpSamp,
                                FPType derivativeStep, Mass mass, Potential const &pot, Bounds<D> bounds,
                                int numberEnergies, RandomGenerator &gen) {
    return WrappedVMCEnergies_<D, N, V>(
        psi, params, false, useImpSamp, [](Positions<D, N>, VarParams<V>) { return FPType{0}; },
        derivativeStep, mass, pot, bounds, numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> params, bool useImpSamp, FPType derivativeStep,
                    Mass mass, Potential const &pot, Bounds<D> bounds, int numberEnergies,
                    RandomGenerator &gen) {
    return AvgAndVar_(VMCEnergies<D, N, V>(psi, params, useImpSamp, derivativeStep, mass, pot, bounds,
                                           numberEnergies, gen));
}

} // namespace vmcp

#endif
