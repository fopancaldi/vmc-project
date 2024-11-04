// Variational Monte Carlo algorithm definitions

#ifndef VMCPROJECT_VMCALGS_HPP
#define VMCPROJECT_VMCALGS_HPP

#include "types.hpp"

// TODO: Find a way to avoid having to always specify the template arguments (usually D, N), it's embarassing

namespace vmcp {

// Computes the energies using the VMC algorithm and the analytical formula for the derivative
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &, VarParams<V>, bool,
                                std::array<FirstDerivative, D> const &, SecondDerivative const &, Mass mass,
                                Potential const &, Bounds<D>, int, RandomGenerator &);

// Computes energy and variance using the VMC algorithm and the analytical formula for the derivative
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
VMCResult VMCEnergy(Wavefunction const &, VarParams<V>, bool, std::array<FirstDerivative, D> const &,
                    SecondDerivative const &, Mass mass, Potential const &, Bounds<D>, int,
                    RandomGenerator &);

// Computes the energies using the VMC algorithm and estimating the derivative numerically
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<Energy> VMCEnergies(Wavefunction const &, VarParams<V>, bool, FPType, Mass, Potential const &,
                                Bounds<D>, int, RandomGenerator &);

// Computes energy and variance using the VMC algorithm and estimating the derivative numerically
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult VMCEnergy(Wavefunction const &, VarParams<V>, bool, FPType, Mass, Potential const &, Bounds<D>, int,
                    RandomGenerator &);

} // namespace vmcp

// Implementation of templates is in this file
// It is separated to improve readability
#include "vmcalgs.inl"

#endif
