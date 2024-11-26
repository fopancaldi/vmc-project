//!
//! @file vmcalgs.hpp
//! @brief Declaration of the VMC algorithms
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the declarations of the VMC algorithms that are meant to be called by the user.
//! Does NOT contain the descritpions of said algorithms.
//! To improve readability, the implementation of the templated functions is in the .inl file.
//! @see vmcalgs.inl
//!

#ifndef VMCPROJECT_VMCALGS_HPP
#define VMCPROJECT_VMCALGS_HPP

#include "types.hpp"

// TODO: Find a way to avoid having to always specify the template arguments (usually D, N), it's embarassing

namespace vmcp {

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Laplacian, class Potential>
std::vector<LocEnAndPoss<D, N>>
VMCLocEnAndPoss(Wavefunction const &, VarParams<V>, Laplacians<N, Laplacian> const &, Masses<N>,
                Potential const &, CoordBounds<D>, IntType, RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Laplacian, class Potential>
VMCResult<V> VMCEnergy(Wavefunction const &, ParamBounds<V>, Laplacians<N, Laplacian> const &, Masses<N>,
                    Potential const &, CoordBounds<D>, IntType, RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative, class Laplacian,
          class Potential>
std::vector<LocEnAndPoss<D, N>>
VMCLocEnAndPoss(Wavefunction const &, VarParams<V>, Gradients<D, N, FirstDerivative> const &,
                Laplacians<N, Laplacian> const &, Masses<N>, Potential const &, CoordBounds<D>, IntType,
                RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative, class Laplacian,
          class Potential>
VMCResult<V> VMCEnergy(Wavefunction const &, ParamBounds<V>, Gradients<D, N, FirstDerivative> const &,
                    Laplacians<N, Laplacian> const &, Masses<N>, Potential const &, CoordBounds<D>, IntType,
                    RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<LocEnAndPoss<D, N>> VMCLocEnAndPoss(Wavefunction const &, VarParams<V>, bool, FPType, Masses<N>,
                                                Potential const &, CoordBounds<D>, IntType,
                                                RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult<V> VMCEnergy(Wavefunction const &, ParamBounds<V>, bool, FPType, Masses<N>, Potential const &,
                    CoordBounds<D>, IntType, RandomGenerator &);

} // namespace vmcp

#include "vmcalgs.inl"

#endif
