// Variational Monte Carlo algorithm definitions

#ifndef VMCPROJECT_VMCALGS_HPP
#define VMCPROJECT_VMCALGS_HPP

#include "types.hpp"

namespace vmcp {

// Computes the energy using the VMC algorithm
// Might be renamed if the name is misleading
template <Dimension D, ParticNum N>
VMCResult VMCIntegral(Wavefunction const &, VarParams const &, Bounds const&, Potential const &, int,
                      RandomGenerator &);

} // namespace vmcp

#endif
