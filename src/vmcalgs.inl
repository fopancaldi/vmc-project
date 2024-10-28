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

// Constants used in VMCEnergies
constexpr IntType pointsSearchPeak = 100;
// TODO: Rename this one
constexpr IntType boundSteps = 100;
constexpr IntType thermalizationMoves = 100;
// TODO: Rename this one, also find if it is really 10 * thermalization moves
constexpr IntType movesForgetICs = 10 * thermalizationMoves;
constexpr IntType vmcMoves = 10;

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
        for (Position p : newPoss) {
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
template <Dimension D, ParticNum N, class Wavefunction>
UIntType MetropolisUpdate_(Wavefunction const &psi, VarParams const &params, Positions<D, N> &poss,
                           FPType step, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, Wavefunction>());

    UIntType succesfulUpdates;
    for (Position p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        // Using float avoids narrowing, whatever FPType is
        std::uniform_real_distribution<FPType> unifDist(-0.5f, 0.5f);
        std::transform(p.begin(), p.end(), p.begin(), [&gen, &unifDist, step](Coordinate c) {
            return Coordinate{c.val + unifDist(gen) * step};
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
// Computes the local energy with the analytic formula for the derivative of the wavefunction
template <Dimension D, ParticNum N, class Wavefunction, class KinEnergy, class Potential>
Energy LocalEnergyAnalytic_(Wavefunction const &psi, VarParams const &params, KinEnergy const &kin,
                            Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, Wavefunction>());
    static_assert(IsKinEnergy<D, N, KinEnergy>());
    static_assert(IsPotential<D, N, Potential>());
    return Energy{kin(poss, params) / psi(poss, params) + pot(poss)};
}

// Computes the energies that will be averaged to obtain the estimate of the GS energy of the VMC algorithm
template <Dimension D, ParticNum N, class Wavefunction, class KinEnergy, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &psi, VarParams const &params, KinEnergy const &kin,
                                Potential const &pot, Bounds<D> bounds, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, Wavefunction>());
    static_assert(IsKinEnergy<D, N, KinEnergy>());
    static_assert(IsPotential<D, N, Potential>());

    std::vector<Energy> result;

    // Step 1: Find a good starting point, it the sense that it easy to move away from
    Positions<D, N> const peak = FindPeak_<D, N>(pot, bounds, pointsSearchPeak, gen);

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
    for (int i = 0; i != vmcMoves; ++i) {
        for (int j = 0; j != thermalizationMoves; ++j) {
            MetropolisUpdate_<D, N>(psi, params, poss, step, gen);
        }
        result.push_back(LocalEnergyAnalytic_<D, N>(psi, params, kin, pot, poss));
    }

    // TODO: Step 4: Adjust the step size

    return result;
}

} // namespace vmcp

#endif
