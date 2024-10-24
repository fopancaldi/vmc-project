// Type definitions and constants

#ifndef VMCPROJECT_TYPES_HPP
#define VMCPROJECT_TYPES_HPP

#include <array>
#include <functional>
#include <random>
#include <vector>

namespace vmcp {

constexpr unsigned int initialLatticeStep = 100;

// TODO: Avoid magic numbers or something
// This one is used in VMCIntegral
constexpr int numberLoops = 100;

// Floating point type, can be adjusted to improve precision or compilation time
using FPType = long double;
// Unsigned integer type
using UIntType = unsigned int;
// Signed integer type
using IntType = int;

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
using Wavefunction = std::function<FPType(Positions<D, N> const&, VarParams)>;

// Potential to be put in a Hamiltonian
template <Dimension D>
using Potential = std::function<FPType(Positions<D>)>;

struct VMCResult {
    FPType energy;
    FPType variance;
};

struct Bound {
    FPType lower;
    FPType upper;
};

// This may be heresy
// Is there a better way?
template <Dimension D>
using Bounds = std::array<Bound, D>;

// default_random_engine is implementation-defined, so I thought it might be more dangerous
using RandomGenerator = std::mt19937;

} // namespace vmcp

#endif
