// Type definitions and constants

#ifndef VMCPROJECT_TYPES_HPP
#define VMCPROJECT_TYPES_HPP

// TODO: Use somewhere the Jackson-Freebeerg kinetic energy

#include <array>
#include <atomic>
#include <cassert>
#include <random>
#include <type_traits>

namespace vmcp {

// 'Structure' types
//
// Floating point type, can be adjusted to improve precision or compilation time
// 'long double' cannot be used here, see
// https://stackoverflow.com/questions/60559650/why-does-stdatomiclong-double-block-indefinitely-in-c14
using FPType = double;
// Unsigned integer type
// Only to be used for arrays, sizes etc.
// When it necessary to count something, use signed integers
using UIntType = long unsigned int;
// Signed integer type
using IntType = int;
// Used instead of default_random_engine since that one is implementation-defined
using RandomGenerator = std::mt19937;

// 'Program lexicon' types
//
// Dimension of the problem, usually it is 1, 2, 3
using Dimension = UIntType;
// Number of particles
using ParticNum = UIntType;
// Number of variational parameters
using VarParNum = UIntType;

// 'Wrapper' structs (so that it is impossible to add two objects of different types)
//
// Position of the particles (in D dimensions)
struct Coordinate {
    FPType val;
    Coordinate &operator+=(Coordinate);
    Coordinate &operator-=(Coordinate);
    Coordinate &operator*=(FPType);
    Coordinate &operator/=(FPType);
};
template <Dimension D>
using Position = std::array<Coordinate, D>;
template <Dimension D, ParticNum N>
using Positions = std::array<Position<D>, N>;
// Variational parameters
struct VarParam {
    FPType val;
    VarParam &operator+=(VarParam);
    VarParam &operator-=(VarParam);
    VarParam &operator*=(FPType);
    VarParam &operator/=(FPType);
};
template <VarParNum V>
using VarParams = std::array<VarParam, V>;
// Mass
struct Mass {
    FPType val;
    Mass &operator+=(Mass);
    Mass &operator-=(Mass);
    Mass &operator*=(FPType);
    Mass &operator/=(FPType);
};
template<ParticNum N>
using Masses = std::array<Mass, N>;
// Variational Monte Carlo algorithm result
struct Energy {
    FPType val;
};
struct EnVariance {
    FPType val;
};
struct VMCResult {
    Energy energy;
    EnVariance variance;
};
template <Dimension D, ParticNum N>
struct LocEnAndPoss {
    Energy energy;
    Positions<D, N> positions;
};
// One-dimensional interval
template <typename T>
struct Bound {
    T lower;
    T upper;
    Bound(T lower_, T upper_) : lower{lower_}, upper{upper_} { assert(upper.val >= lower.val); }
    T Length() const { return upper - lower; }
};
template <Dimension D>
using CoordBounds = std::array<Bound<Coordinate>, D>;
template <VarParNum V>
using ParamBounds = std::array<Bound<VarParam>, V>;

// Derivatives
template <Dimension D, class FirstDerivative>
using Gradient = std::array<FirstDerivative, D>;
template <Dimension D, ParticNum N, class FirstDerivative>
using Gradients = std::array<Gradient<D, FirstDerivative>, N>;
template <ParticNum N, class Laplacian>
using Laplacians = std::array<Laplacian, N>;
// To (only) be used in a static assertion
template <Dimension D, ParticNum N, VarParNum V, class Function>
constexpr bool IsWavefunction() {
    return std::is_invocable_r_v<FPType, Function, Positions<D, N> const &, VarParams<V>>;
}
template <Dimension D, ParticNum N, VarParNum V, class Function>
constexpr bool IsWavefunctionDerivative() {
    return IsWavefunction<D, N, V, Function>();
}
template <Dimension D, ParticNum N, class Function>
constexpr bool IsPotential() {
    return std::is_invocable_r_v<FPType, Function, Positions<D, N> const &>;
}

} // namespace vmcp

// Some implementations are in this file
// It is separated to improve readability
#include "types.inl"

#endif
