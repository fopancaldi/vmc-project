//!
//! @file types.hpp
//! @brief Type definition header
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Definitions of the types used in the program.
//! Some of them are simple wrapper structs, and are used to ensure that the arithmetic operations have
//! physical sense.
//! For example, the code forbids adding up an object of type 'Mass' with one of type 'Energy'.
//!

#ifndef VMCPROJECT_TYPES_HPP
#define VMCPROJECT_TYPES_HPP

#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <random>
#include <type_traits>

namespace vmcp {

//! @defgroup struct-types Structure types
//! @brief Type aliases, used to define the other types
//!
//! These types give a consistent structure to the program and are used to define the other ones.
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
using RandomGenerator = std::default_random_engine;

//! @}

//! @defgroup lexic-types Lexicon types
//! @brief Types that have a clear physical meaning
//!
//! Types that have a clear physical meaning.
//! Also includes wrapper structs.
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
    Coordinate &operator+=(Coordinate other) {
        val += other.val;
        return *this;
    }
    Coordinate &operator-=(Coordinate other) {
        val -= other.val;
        return *this;
    }
    Coordinate &operator*=(FPType other) {
        val *= other;
        return *this;
    }
    Coordinate &operator/=(FPType other) {
        val /= other;
        return *this;
    }
};
inline Coordinate operator+(Coordinate lhs, Coordinate rhs) { return lhs += rhs; }
inline Coordinate operator-(Coordinate lhs, Coordinate rhs) { return lhs -= rhs; }
inline Coordinate operator*(Coordinate lhs, FPType rhs) { return lhs *= rhs; }
inline Coordinate operator/(Coordinate lhs, FPType rhs) { return lhs /= rhs; }
//! @brief Position of one particle in D dimensions
template <Dimension D>
using Position = std::array<Coordinate, D>;
//! @brief Position of N particles in D dimensions
template <Dimension D, ParticNum N>
using Positions = std::array<Position<D>, N>;
//! @brief Variational parameter
struct VarParam {
    FPType val;
    VarParam &operator+=(VarParam other) {
        val += other.val;
        return *this;
    }
    VarParam &operator-=(VarParam other) {
        val -= other.val;
        return *this;
    }
    VarParam &operator*=(FPType other) {
        val *= other;
        return *this;
    }
    VarParam &operator/=(FPType other) {
        val /= other;
        return *this;
    }
};
inline VarParam operator+(VarParam lhs, VarParam rhs) { return lhs += rhs; }
inline VarParam operator-(VarParam lhs, VarParam rhs) { return lhs -= rhs; }
inline VarParam operator*(VarParam lhs, FPType rhs) { return lhs *= rhs; }
inline VarParam operator*(FPType lhs, VarParam rhs) { return rhs * lhs; }
inline VarParam operator/(VarParam lhs, FPType rhs) { return lhs /= rhs; }
//! @brief Set of V variational parameters
template <VarParNum V>
using VarParams = std::array<VarParam, V>;
//! @brief Mass of one particle
struct Mass {
    FPType val;
    Mass &operator+=(Mass other) {
        val += other.val;
        return *this;
    }
    Mass &operator-=(Mass other) {
        val -= other.val;
        return *this;
    }
    Mass &operator*=(FPType other) {
        val *= other;
        return *this;
    }
    Mass &operator/=(FPType other) {
        val /= other;
        return *this;
    }
};
inline Mass operator+(Mass lhs, Mass rhs) { return lhs += rhs; }
inline Mass operator-(Mass lhs, Mass rhs) { return lhs -= rhs; }
inline Mass operator*(Mass lhs, FPType rhs) { return lhs *= rhs; }
inline Mass operator*(FPType lhs, Mass rhs) { return rhs * lhs; }
inline Mass operator/(Mass lhs, FPType rhs) { return lhs /= rhs; }
//! @brief Masses on N particles
template <ParticNum N>
using Masses = std::array<Mass, N>;
//! @brief Energy of the system
struct Energy {
    FPType val;
    Energy &operator+=(Energy other) {
        val += other.val;
        return *this;
    }
    Energy &operator-=(Energy other) {
        val -= other.val;
        return *this;
    }
    Energy &operator*=(FPType other) {
        val *= other;
        return *this;
    }
    Energy &operator/=(FPType other) {
        val /= other;
        return *this;
    }
    friend std::ostream &operator<<(std::ostream &os, Energy e) { return os << e.val; }
};
inline Energy operator+(Energy lhs, Energy rhs) { return lhs += rhs; }
inline Energy operator-(Energy lhs, Energy rhs) { return lhs -= rhs; }
inline Energy operator*(Energy lhs, FPType rhs) { return lhs *= rhs; }
inline Energy operator*(FPType lhs, Energy rhs) { return rhs * lhs; }
inline Energy operator/(Energy lhs, FPType rhs) { return lhs /= rhs; }
inline bool operator<(Energy lhs, Energy rhs) { return lhs.val < rhs.val; }
inline bool operator>(Energy lhs, Energy rhs) { return lhs.val > rhs.val; }
inline Energy max(Energy lhs, Energy rhs) { return lhs > rhs ? lhs : rhs; }
inline Energy abs(Energy e) { return Energy{std::abs(e.val)}; };
//! @brief Square of the energy of the system
struct EnSquared {
    FPType val;
    EnSquared &operator+=(EnSquared other) {
        val += other.val;
        return *this;
    }
    EnSquared &operator-=(EnSquared other) {
        val -= other.val;
        return *this;
    }
    EnSquared &operator*=(FPType other) {
        val *= other;
        return *this;
    }
    EnSquared &operator/=(FPType other) {
        val /= other;
        return *this;
    }
};
inline EnSquared operator+(EnSquared lhs, EnSquared rhs) { return lhs += rhs; }
inline EnSquared operator-(EnSquared lhs, EnSquared rhs) { return lhs -= rhs; }
inline EnSquared operator*(EnSquared lhs, FPType rhs) { return lhs *= rhs; }
inline EnSquared operator*(FPType lhs, EnSquared rhs) { return rhs * lhs; }
inline EnSquared operator/(EnSquared lhs, FPType rhs) { return lhs /= rhs; }
inline EnSquared operator*(Energy lhs, Energy rhs) { return EnSquared{lhs.val * rhs.val}; }
inline Energy sqrt(EnSquared es) { return Energy{std::sqrt(es.val)}; }
// LF TODO: Remove when you do not need this anymore (so when you are sure that all statistical methods that
// before returned a 'PartialVMCResult' now retyurn an 'Energy')
//! @brief Average of the energy and its error
struct PartialVMCResult {
    Energy energy;
    Energy stdDev;
};
//! @brief Average of the energy and its error, and the best variational parameters
template <VarParNum V>
struct VMCResult {
    Energy energy;
    Energy stdDev;
    VarParams<V> bestParams;
};
//! @brief Local energy and the positions of the particles when it was computed
template <Dimension D, ParticNum N>
struct LocEnAndPoss {
    Energy localEn;
    Positions<D, N> positions;
};
//! @brief One-dimensional interval
//!
//! Requires the templated type to be a class (or struct) with public member 'val'.
template <typename T>
struct Bound {
    T lower;
    T upper;
    Bound() : lower{}, upper{} {}
    Bound(T lower_, T upper_) : lower{lower_}, upper{upper_} { assert(upper.val >= lower.val); }
    T Length() const { return upper - lower; }
};
//! @brief D-dimensional coordinate interval
template <Dimension D>
using CoordBounds = std::array<Bound<Coordinate>, D>;
//! @brief Intervals for V variational parameters
template <VarParNum V>
using ParamBounds = std::array<Bound<VarParam>, V>;
// LF TODO: Would rather have BlockingResult as a vector of structs, where each struct contains one size, one
// mean, one stdDev
//! @brief Statistical analysis results
struct BlockingResult {
    std::vector<IntType> sizes;
    std::vector<Energy> means;
    std::vector<Energy> stdDevs;
};
struct ConfInterval {
    Energy min;
    Energy max;
};
//! @brief Statistical tags
enum class Statistic { mean, stdDev };
//! @brief Statistical analysis fuctions
enum class StatFuncType { regular, blocking, bootstrap };

//! @}

//! @defgroup func-properties Function properties
//! @brief Functions that check the signature of a function
//!
//! Functions used to guarantee properties (for now, just the signature) of the functions passed to the VMC
//! algorithms.
//! @{

//! @brief Checks the signature of the function
//! @return Whether the function has the correct signature
//!
//! Checks if Function takes the positions of N particles in D dimension and V variational parameters, and
//! returns a real number.
template <Dimension D, ParticNum N, VarParNum V, class Function>
constexpr bool IsWavefunction() {
    return std::is_invocable_r_v<FPType, Function, Positions<D, N> const &, VarParams<V>>;
}
//! @brief Checks the signature of the function
//! @return Whether the function has the correct signature
//!
//! Checks if Function takes the positions of N particles in D dimension and V variational
//! parameters, and returns a real number.
template <Dimension D, ParticNum N, VarParNum V, class Function>
constexpr bool IsWavefunctionDerivative() {
    return IsWavefunction<D, N, V, Function>();
}
//! @brief Checks the signature of the function
//! @return Whether the function has the correct signature
//!
//! Checks if Function takes the positions of N particles in D dimension, and returns a real number.
template <Dimension D, ParticNum N, class Function>
constexpr bool IsPotential() {
    return std::is_invocable_r_v<FPType, Function, Positions<D, N> const &>;
}

//! @}

//! @addtogroup lexic-types
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

#endif
