//!
//! @file types.hpp
//! @brief Type definition header
//! @author Lorenzo Fabbri
//! @author Francesco Orso Pancaldi
//!
//! Definitions of the types used in the program.
//! Some of them are simple wrapper structs, and are used to ensure that the arithmetic operations have
//! physical sense.
//! For example, the code forbids adding up an object of type 'Mass' with one of type 'Energy'.
//!

#ifndef VMCPROJECT_TYPES_HPP
#define VMCPROJECT_TYPES_HPP

// TODO: Use somewhere the Jackson-Freebeerg kinetic energy

#include <array>
#include <atomic>
#include <cassert>
#include <random>
#include <type_traits>

namespace vmcp {

//! @defgroup Structure types
//! Give a consistent structure to the program and are used to define the other ones.
//! @{

//! @brief Floating point type
//!
//! Can be adjusted to improve either precision or execution time.
//! Only C++ floating point types are allowed
//! 'long double' cannot be used due to a bug in the C++ 'atomic' library.
//! See https://stackoverflow.com/questions/60559650/why-does-stdatomiclong-double-block-indefinitely-in-c14
using FPType = double;
static_assert(std::is_floating_point_v<FPType>);

//! @brief Signed integer type
//!
//! The type to use when an integer is needed, even if that integer is guaranteed to be non-negative.
using IntType = int;
static_assert(std::is_integral_v<IntType>);
static_assert(std::is_signed_v<IntType>);

//! @brief Unsigned integer type
//!
//! Should be used instead of IntType only when unsigned integers are explicitly required (for example, in a
//! std::array).
using UIntType = long unsigned int;
static_assert(std::is_integral_v<UIntType>);
static_assert(std::is_unsigned_v<UIntType>);

//! @brief Random generator type
//!
//! Used instead of default_random_engine since that one is implementation-defined.
using RandomGenerator = std::mt19937;

//! @}

//! @defgroup Lexicon types
//! FIXME: What the heck is this descrptiion?
//! Give meaning to the structure types by using names that clarify their properties.
//! Also include wrapper structs.
//! @{

//! @brief Number of space dimensions
using Dimension = UIntType;

//! @brief Number of particles
using ParticNum = UIntType;

//! @brief Number of variational parameters
using VarParNum = UIntType;

//! @brief Position of one particle in one dimension
struct Coordinate {
    FPType val;
    Coordinate &operator+=(Coordinate);
    Coordinate &operator-=(Coordinate);
    Coordinate &operator*=(FPType);
    Coordinate &operator/=(FPType);
};

//! @brief Position of one particle in D dimensions
template <Dimension D>
using Position = std::array<Coordinate, D>;

//! @brief Position of N particles in D dimensions
template <Dimension D, ParticNum N>
using Positions = std::array<Position<D>, N>;

//! @brief Variational parameter
struct VarParam {
    FPType val;
    VarParam &operator+=(VarParam);
    VarParam &operator-=(VarParam);
    VarParam &operator*=(FPType);
    VarParam &operator/=(FPType);
};

//! @brief Set of V variational parameters
template <VarParNum V>
using VarParams = std::array<VarParam, V>;

//! @brief Mass of one particle
struct Mass {
    FPType val;
    Mass &operator+=(Mass);
    Mass &operator-=(Mass);
    Mass &operator*=(FPType);
    Mass &operator/=(FPType);
};

//! @brief Masses on N particles
template <ParticNum N>
using Masses = std::array<Mass, N>;

//! @brief Energy of the system
struct Energy {
    FPType val;
};

//! @brief Variance on the average of the energies
//!
//! Not to be confused with the variance on the energy.
//! Has 'units of measurement' Energy^2
struct EnVariance {
    FPType val;
};

//! @brief Average of the energy with error
struct VMCResult {
    Energy energy;
    EnVariance variance;
};

//! @brief Local energy and the positions of the particles when it was computed
template <Dimension D, ParticNum N>
struct LocEnAndPoss {
    Energy localEn;
    Positions<D, N> positions;
};

//! @brief One-dimensional interval
//!
//! Requires the templated type to be a class with public member 'val'.
template <typename T>
struct Bound {
    T lower;
    T upper;
    Bound(T lower_, T upper_) : lower{lower_}, upper{upper_} { assert(upper.val >= lower.val); }
    T Length() const { return upper - lower; }
};

//! @brief D-dimensional coordinate interval
template <Dimension D>
using CoordBounds = std::array<Bound<Coordinate>, D>;

//! @brief Intervals for V variational parameters
template <VarParNum V>
using ParamBounds = std::array<Bound<VarParam>, V>;

//! @}

//! @defgroup Function properties
//! Functions used to guarantee properties (for now, just the signature) of the functions passed to the VMC
//! algorithms.
//! @{

//! @brief Checks the signature of the function
template <Dimension D, ParticNum N, VarParNum V, class Function>
constexpr bool IsWavefunction() {
    return std::is_invocable_r_v<FPType, Function, Positions<D, N> const &, VarParams<V>>;
}

//! @brief Checks the signature of the function
template <Dimension D, ParticNum N, VarParNum V, class Function>
constexpr bool IsWavefunctionDerivative() {
    return IsWavefunction<D, N, V, Function>();
}

//! @brief Checks the signature of the function
template <Dimension D, ParticNum N, class Function>
constexpr bool IsPotential() {
    return std::is_invocable_r_v<FPType, Function, Positions<D, N> const &>;
}

//! @}

//! @addtogroup Lexicon types
//! @{

//! @brief Gradient for one particle in D dimensions
template <Dimension D, class FirstDerivative>
using Gradient = std::array<FirstDerivative, D>;

//! @brief Gradients for N particles in D dimensions
template <Dimension D, ParticNum N, class FirstDerivative>
using Gradients = std::array<Gradient<D, FirstDerivative>, N>;

//! @brief Laplacians for N particles
template <ParticNum N, class Laplacian>
using Laplacians = std::array<Laplacian, N>;

//! @}

} // namespace vmcp

// Some implementations are in this file
// It is separated to improve readability
#include "types.inl"

#endif
