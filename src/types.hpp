// Type definitions and constants

// TODO: Replace these types with structs

#ifndef VMCPROJECT_TYPES_HPP
#define VMCPROJECT_TYPES_HPP

#include <array>
#include <cassert>
#include <functional>
#include <random>
#include <vector>

namespace vmcp {

// 'Structure' types

// Floating point type, can be adjusted to improve precision or compilation time
using FPType = long double;
// Unsigned integer type
using UIntType = unsigned int;
// Signed integer type
using IntType = int;

// 'Program lexicon' types

// Dimension of the problem, usually it is 1, 2, 3
using Dimension = UIntType;
// Spatial coordinate
using Coordinate = FPType;
// Number of particles
using ParticNum = UIntType;
// Position of one particle
template <Dimension D>
using Position = std::array<Coordinate, D>;
// Positions of an many particles
template <Dimension D, ParticNum N>
using Positions = std::array<Position<D>, N>;
// Variational parameters
using VarParams = std::vector<FPType>;
// First argument is the positions, second argument is the variational parameters
template <Dimension D, ParticNum N>
using Wavefunction = std::function<FPType(Positions<D, N> const &, VarParams)>;
// Potential to be put in a Hamiltonian
template <Dimension D, ParticNum N>
using KinEnergy = std::function<FPType(Positions<D, N> const &, VarParams)>;
template <Dimension D, ParticNum N>
using Potential = std::function<FPType(Positions<D, N> const &)>;
struct VMCResult {
    FPType energy;
    FPType variance;
};
struct Bound {
    FPType lower;
    FPType upper;
    FPType Size() const {
        assert(upper >= lower);
        return upper - lower;
    }
};
template <Dimension D>
using Bounds = std::array<Bound, D>;
// default_random_engine is implementation-defined, so I thought it might be more dangerous
using RandomGenerator = std::mt19937;

// Constants
constexpr FPType pi = 3.1415926535897932384626433;
constexpr FPType e = 2.71828182845904523536;

// TODO: Avoid magic numbers or something
// These one are used in VMCIntegral
constexpr IntType numberLoops = 100;
constexpr UIntType initialLatticeStep = 100;
constexpr IntType boundSteps = 100;
constexpr IntType thermalizationMoves = 100;
constexpr IntType vmcMoves = 10;

} // namespace vmcp

#endif

// TODO: Use somewhere the Jackson-Freebeerg kinetic energy
