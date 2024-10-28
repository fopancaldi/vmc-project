// Variational Monte Carlo algorithm definitions

#ifndef VMCPROJECT_VMCALGS_HPP
#define VMCPROJECT_VMCALGS_HPP

#include "types.hpp"

// TODO: Find a way to avoid having to always specify the template arguments (usually D, N), it's embarassing

namespace vmcp {

// TODO: Remove from the header and replace with 2 wrapper functions
// Computes the energies using the VMC algorithm
// Might be renamed if the name is misleading
template <Dimension D, ParticNum N, class Wavefunction, class KinEnergy, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &, VarParams const &, KinEnergy const &, Potential const &,
                                Bounds<D>, RandomGenerator &);

} // namespace vmcp

// Implementation of templates is in this file
// It is separated to improve readability
#include "vmcalgs.inl"

#endif
