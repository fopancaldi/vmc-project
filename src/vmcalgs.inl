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
// TODO: Rename this one
constexpr IntType boundSteps = 100;
constexpr IntType thermalizationMoves = 100;
// TODO: Rename this one, also find if it is really 10 * thermalization moves
constexpr IntType movesForgetICs = 10 * thermalizationMoves;
// constexpr IntType vmcMoves = 10;

// Should be hidden from the user
// Computes the (approximate) peak of the potential
// In practice, choose some points in the integration domain and see where the potential is largest
template <Dimension D, ParticNum N, class Potential>
Positions<D, N> FindPeak_(Potential pot, Bounds<D> bounds, IntType points, RandomGenerator &gen) {
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
        for (Position<D> p : newPoss) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), [&unif, &gen](Bound b) {
                return Coordinate{b.lower + unif(gen) * (b.upper - b.lower)};
            });
        }
        if (pot(newPoss) > pot(result)) {
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

    UIntType successfulUpdates;
    for (Position<D> p : poss) {
        Position<D> const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        // Using float avoids narrowing, whatever FPType is
        std::uniform_real_distribution<FPType> unifDist(-0.5f, 0.5f);
        std::transform(p.begin(), p.end(), p.begin(), [&gen, &unifDist, step](Coordinate c) {
            return Coordinate{c.val + unifDist(gen) * step};
        });
        if (unifDist(gen) > std::pow(psi(poss, params) / oldPsi, 2)) {
            p = oldPos;

        } else {
            ++successfulUpdates;
        }
    }
    return successfulUpdates;
}

// Should be hidden from the user
// Computes the drift force using its analytic expression
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
void DriftForceAnalytic_(Wavefunction const &psi, VarParams<V> params, Positions<D, N> poss,
                         Position<D> &driftForce) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    FPType alpha = params[0];
    for (int i = 0; i < D; ++i) {
        driftForce[i].val = -4 * alpha * poss[i].val;
    }
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
void DriftForceNumeric_(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep,
                        Positions<D, N> poss, Position<D> &driftForce) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            driftForce[d] = 2 *
                            (psi(PerturbPos_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                             psi(PerturbPos_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                            (derivativeStep * psi(poss, params));
        }
    }
}

// Updates the wavefunction with the Importance Sampling algorithms and outputs the number of succesful
// updates
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
UIntType ImportanceSamplingUpdate_(Wavefunction const &psi, VarParams<V> params, Positions<D, N> &poss,
                                   FPType step, FPType deltaT, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    UIntType successfulUpdates;
    for (Position<D> p : poss) {
        Position<D> const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        Position<D> driftForce;
        DriftForceAnalytic_(psi, poss, params, driftForce);
        std::normal_distribution<FPType> normalDist(0.0, 1.0); // Gaussian noise
        std::transform(p.begin(), p.end(), driftForce.begin(), p.begin(),
                       [&gen, &normalDist, step, deltaT](Coordinate c, Coordinate f) {
                           return Coordinate{c.val + f.val * deltaT + normalDist(gen) * std::sqrt(deltaT)};
                       });
        FPType const newPsi = psi(poss, params);
        Position<D> newDriftForce;
        DriftForceAnalytic_(psi, poss, params, newDriftForce);
        FPType const forwardProb =
            std::exp(-0.5 * deltaT *
                     std::inner_product(driftForce.begin(), driftForce.end(), driftForce.begin(), 0.0));
        FPType const backwardProb = std::exp(
            -0.5 * deltaT *
            std::inner_product(newDriftForce.begin(), newDriftForce.end(), newDriftForce.begin(), 0.0));
        FPType const acceptanceRatio = (newPsi * newPsi * backwardProb) / (oldPsi * oldPsi * forwardProb);
        std::uniform_real_distribution<FPType> unifDist(0.0f, 1.0f);
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
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class KinEnergy, class Potential>
Energy LocalEnergyAnalytic_(Wavefunction const &psi, VarParams<V> params, KinEnergy const &kin,
                            Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsKinEnergy<D, N, V, KinEnergy>());
    static_assert(IsPotential<D, N, Potential>());

    return Energy{kin(poss, params) / psi(poss, params) + pot(poss)};
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
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class KinEnergy, class Potential>
std::vector<Energy> WrappedVMCEnergies_(Wavefunction const &psi, VarParams<V> params, bool useAnalitycal,
                                        KinEnergy const &kin, FPType derivativeStep, Mass mass,
                                        Potential const &pot, Bounds<D> bounds, int numberEnergies,
                                        RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsKinEnergy<D, N, V, KinEnergy>());
    static_assert(IsPotential<D, N, Potential>());
    assert(numberEnergies >= 0);

    std::vector<Energy> result;

    // Step 1: Find a good starting point, in the sense that it's easy to move away from
    Positions<D, N> const peak = FindPeak_<D, N>(pot, bounds, numSamplesForPeakSearch, gen);

    // Step 2: Choose the step
    Bound const smallestBound = *(std::min_element(
        bounds.begin(), bounds.end(), [](Bound b1, Bound b2) { return b1.Length() < b2.Length(); }));
    FPType const step = smallestBound.Length() / boundSteps;

    // Step 3: Save the energies to be averaged
    Positions<D, N> poss = peak;
    // Move away from the peak, in order to forget the dependence on the initial conditions
    for (int i = 0; i != movesForgetICs; ++i) {
        MetropolisUpdate_<D, N>(psi, params, poss, step, gen);
    }
    for (int i = 0; i != numberEnergies; ++i) {
        for (int j = 0; j != thermalizationMoves; ++j) {
            MetropolisUpdate_<D, N>(psi, params, poss, step, gen);
        }
        if (useAnalitycal) {
            result.push_back(LocalEnergyAnalytic_<D, N>(psi, params, kin, pot, poss));
        } else {
            result.push_back(LocalEnergyNumeric_<D, N>(psi, params, derivativeStep, mass, pot, poss));
        }
    }

    // TODO: Step 4: Adjust the step size

    return result;
}

VMCResult AvgAndVar_(std::vector<Energy> const &v) {
    assert(v.size() > 1);
    // TODO: Maybe define operator+ for the energies?
    Energy cumul = std::accumulate(v.begin(), v.end(), Energy(0),
                                   [](Energy e1, Energy e2) { return Energy{e1.val + e2.val}; });
    EnVariance cumulSq = std::accumulate(v.begin(), v.end(), EnVariance(0), [](EnVariance ev, Energy e) {
        return EnVariance{ev.val + e.val * e.val};
    });
    return VMCResult{cumul.val / v.size(), cumulSq.val / (v.size() - 1)};
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class KinEnergy, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &psi, VarParams<V> params, KinEnergy const &kin,
                                Potential const &pot, Bounds<D> bounds, int numberEnergies,
                                RandomGenerator &gen) {
    return WrappedVMCEnergies_<D, N, V>(psi, params, true, kin, 0, Mass{0}, pot, bounds, numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class KinEnergy, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> params, KinEnergy const &kin, Potential const &pot,
                    Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    return AvgAndVar_(VMCEnergies<D, N, V>(psi, params, kin, pot, bounds, numberEnergies, gen));
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep,
                                Mass mass, Potential const &pot, Bounds<D> bounds, int numberEnergies,
                                RandomGenerator &gen) {
    return WrappedVMCEnergies_<D, N, V>(
        psi, params, false, [](Positions<D, N>, VarParams<V>) { return FPType{0}; }, derivativeStep, mass,
        pot, bounds, numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep, Mass mass,
                    Potential const &pot, Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    return AvgAndVar_(
        VMCEnergies<D, N, V>(psi, params, derivativeStep, mass, pot, bounds, numberEnergies, gen));
}

} // namespace vmcp

#endif
