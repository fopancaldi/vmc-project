#include "vmcalgs.hpp"

#include <algorithm>

namespace vmcp {

template <Dimension D, ParticNum N>
VMCResult VMCIntegral(Wavefunction<D, N> const &psi, VarParams const &params, Bounds<D> const &bounds,
                      Potential<D> const &pot, int derivativeAlg, RandomGenerator &rndGen) {
    // Step 1: find a good starting point
    // In pratcice, take numberLoops points in the integration domain and see where the potential is largest
    Position<D> center;
    std::transform(bounds.begin(), center.begin(), [](Bound b) { return (b.upper + b.lower) / 2; });
    Positions<D, N> startingPos;
    std::fill(startingPos.begin(), startingPos.end(), center);
    std::uniform_real_distribution intDist(0, 1);
    for (int i{0}; i != numberLoops; ++i) {
        Positions<D, N> newPos;
        std::for_each(newPos.begin(), newPos.end(), [&](Position p) {
            std::transform(bounds.begin(), bounds.end(), p.begin(),
                           [&](Bound b) { return b.lower + intDist(rndGen) * (b.upper - b.lower); });
        });
        if (pot(newPos) > pot(startingPos)) {
            startingPos = newPos;
        }
    }

    // Step 2: compute the energy

    return VMCResult{0, 0};
}

} // namespace vmcp
