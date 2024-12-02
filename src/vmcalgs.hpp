//!
//! @file vmcalgs.hpp
//! @brief Declaration of the user functions and the constants
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! To improve readability, the definitions of the functions is in the related .inl file
//! Despite possibly cluttering the header, the constants are here for easiness of access (and avoiding double
//! inclusions or things like that) by the .inl files
//! @see vmcalgs.inl
//!

#ifndef VMCPROJECT_VMCALGS_HPP
#define VMCPROJECT_VMCALGS_HPP

#include "types.hpp"

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
                       Laplacians<N, Laplacian> const &, Masses<N>, Potential const &, CoordBounds<D>,
                       IntType, RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<LocEnAndPoss<D, N>> VMCLocEnAndPoss(Wavefunction const &, VarParams<V>, bool, FPType, Masses<N>,
                                                Potential const &, CoordBounds<D>, IntType,
                                                RandomGenerator &);

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult<V> VMCEnergy(Wavefunction const &, ParamBounds<V>, bool, FPType, Masses<N>, Potential const &,
                       CoordBounds<D>, IntType, RandomGenerator &);

//! @defgroup algs-constants Constants
//! @brief Constants used in the algorithms and/or the helper functions
//!
//! Constants used in the algorithms.
//! They are named with the convention 'constantName_algorithmThatUsesIt'.
//! @{

//! @brief Reduced Planck constant
constexpr FPType hbar = 1;
//! @brief Number of randomly chosen points in the integration region where the potential is computed
//! @see FindPeak
constexpr IntType points_peakSearch = 100;
//! @brief Minimal value of the wavefunction to accept the point as the new highest point of the potential
//! @see FindPeak
//!
//! Minimal value of the wavefunction to accept the point as the new peak of the potential.
//! Aviods situations where the wavefunction at the peak is 'nan', which breaks the update algorithms.
constexpr FPType minWavef_peakSearch = 1e-6f;
//! @brief Maximum number of iterations of the gradient descent
//! @see BestParams
//!
//! Maximum number of iterations of the gradient descent.
//! Should never be reached, unless the parameters are very large.
constexpr IntType maxLoops_gradDesc = 100000;
//! @brief The norm of the initial parameters divided by this gives the starting step for the gradient descent
constexpr IntType stepDenom_gradDesc = 100;
//! @brief When the gradient divided by the parameters' norm is smaller than this, stop the gradient descent
constexpr FPType stoppingThreshold_gradDesc = 1e-9f;
//! @brief Number of independent gradient descents carried out simultaneously
constexpr IntType numWalkers_gradDesc = 1;
//! @brief Fraction of the current energy that can be gained at most in a gradient descent move
constexpr FPType increaseFrac_gradDesc = 0.1f;
// FP TODO: Rename this one, and document
constexpr IntType stepDenom_vmcLEPs = 100;
//! @brief Number of updates after which the sampled local energies are uncorrelated
//! @see VMCLocEnAndPoss
constexpr IntType autocorrelationMoves_vmcLEPs = 100;
//! @brief Number of moves after which the system has forgot about its initial conditions
//! @see VMCLocEnAndPoss
constexpr IntType movesForgetICs_vmcLEPs = 10 * autocorrelationMoves_vmcLEPs;
//! @brief Optimal acceptance rate for the updates in the VMC algorithm
constexpr FPType targetAcceptRate_vmcLEPs = 0.5f;
// LF TODO: Rename this one, and document
// Also, is there a better criterion to choose this than to simply fix it (which is a bad idea since it might
// turn out to be too large for some programs)?
// I propose the following criterion: ask for 'step' as an input too (like MetropolisUpdate_ does), and choose
// deltaT such that on average the random part of the jump has length 'step'
constexpr FPType deltaT = 0.005f;

//! @}

} // namespace vmcp

#include "vmcalgs.inl"

#endif
