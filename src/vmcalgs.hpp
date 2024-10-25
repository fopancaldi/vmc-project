// Variational Monte Carlo algorithm definitions

#ifndef VMCPROJECT_ALGS_HPP
#define VMCPROJECT_ALGS_HPP

#include "types.hpp"

// TODO: Deal with this when you do the .inl
#include <iostream>

// TODO: Find a way to avoid having to always specify the template arguments (usually D, N), it's embarassing

namespace vmcp {

int Test(int);

// TODO: Remove from the header and replace with 2 wrapper functions
// Computes the energy using the VMC algorithm
// Might be renamed if the name is misleading
template <Dimension D, ParticNum N>
std::vector<FPType> VMCIntegral(Wavefunction<D, N> const &psi, VarParams const &params,
                                Bounds<D> const &bounds, Potential<D, N> const &pot,
                                KinEnergy<D, N> const &kin, RandomGenerator &gen);

/////////////////////////////////////////////////////////

template <Dimension D, ParticNum N>
void MetropolisUpdate(Wavefunction<D, N> psi, VarParams const &params, Positions<D, N> &poss, FPType step,
                      RandomGenerator &gen) {
    for (Position<D> p : poss) {
        Position<D> oldPos = p;
        FPType oldPsi = psi(poss, params);
        // TODO: Maybe this works even if I remove (0, 1)?
        std::uniform_real_distribution<FPType> unifDist(-0.5f, 0.5f);
        // p += (unifDist(gen) - FPType{0.5f}) * step;
        std::transform(p.begin(), p.end(), p.begin(),
                       [&gen, &unifDist, step](FPType f) { return f + unifDist(gen) * step; });

        // If the metropolis question is negative, reject the move
        if (unifDist(gen) > std::pow(psi(poss, params) / oldPsi, 2)) {
            p = oldPos;
        }
    }
}

template <Dimension D, ParticNum N>
FPType LocalEnergyAnalytic(Wavefunction<D, N> const &psi, VarParams const &params, Positions<D, N> poss,
                           KinEnergy<D, N> const &kin, Potential<D, N> const &pot) {
    return kin(poss, params) / psi(poss, params) + pot(poss);
}

// TODO: Change name
template <Dimension D, ParticNum N>
std::vector<FPType> VMCIntegral(Wavefunction<D, N> const &psi, VarParams const &params,
                                Bounds<D> const &bounds, Potential<D, N> const &pot,
                                KinEnergy<D, N> const &kin, RandomGenerator &gen) {
    std::vector<FPType> result;
    // Step 1: find a good starting point
    // In pratcice, take numberLoops points in the integration domain and see where the potential is largest
    Position<D> center;
    std::transform(bounds.begin(), bounds.end(), center.begin(),
                   [](Bound b) { return (b.upper + b.lower) / 2; });
    Positions<D, N> start;
    std::fill(start.begin(), start.end(), center);
    // TODO: Maybe this works even if I remove (0, 1)?
    std::uniform_real_distribution<FPType> unifDist(0, 1);
    for (int i = 0; i != numberLoops; ++i) {
        Positions<D, N> newPos;
        std::for_each(newPos.begin(), newPos.end(), [&](Position<D> p) {
            std::transform(bounds.begin(), bounds.end(), p.begin(),
                           [&](Bound b) { return b.lower + unifDist(gen) * (b.upper - b.lower); });
        });
        if (pot(newPos) > pot(start)) {
            start = newPos;
        }
    }

    // Step 2: choose the step
    Bound const smallestBound = *(std::min_element(bounds.begin(), bounds.end(),
                                                   [](Bound b1, Bound b2) { return b1.Size() < b2.Size(); }));
    FPType const step = smallestBound.Size() / boundSteps;

    // Step 3: save the energies to be averaged
    // TODO: MAGIC CONSTANT ALERT
    // This step is done to quickly (since we start from a very high energy) forget about the initial
    // conditions
    Positions<D, N> poss = start;
    // I read somewhere that the correct number of moves to forget the ICs is 10 * thermalization moves?
    for (int i = 0; i != 10 * thermalizationMoves; ++i) {
        MetropolisUpdate<D, N>(psi, params, poss, step, gen);
    }
    // Cmpute the energies
    for (int i = 0; i != vmcMoves; ++i) {
        for (int j = 0; j != thermalizationMoves; ++j) {
            MetropolisUpdate<D, N>(psi, params, poss, step, gen);
        }
        result.push_back(LocalEnergyAnalytic<D, N>(psi, params, poss, kin, pot));
        std::cout << i << '\n';
    }

    // TODO: Step 4: Adjust the step size

    return result;
}

} // namespace vmcp

#endif
