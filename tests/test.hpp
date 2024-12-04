//!
//! @file test.hpp
//! @brief Definition of constants and generic functions used in the tests
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!

#ifndef VMCP_TESTS_HPP
#define VMCP_TESTS_HPP

#include "types.hpp"

// Chosen at random, but fixed to guarantee reproducibility of failed tests
constexpr vmcp::UIntType seed = 6482658u;
const std::string logFilePath = "../artifacts/test-log.txt";

constexpr vmcp::IntType allowedStdDevs = 10;
// Maximum allowed discrepancy between the computed energy and the expected energy
constexpr vmcp::Energy vmcEnergyTolerance{0.5f};
// Should have been constexpr, we just used const since otherwise the intellisense complains
// If the standard deviation is smaller than this, it is highly probable that numerical errors were
// non-negligible
const vmcp::Energy stdDevTolerance{std::numeric_limits<vmcp::FPType>::epsilon() * 100};

constexpr vmcp::IntType iterations = 1 << 6;
// The denominator to obtain the number of iterations when the variational parameters are used
constexpr vmcp::IntType vpIterationsFactor = 1 << 4;
static_assert((iterations % vpIterationsFactor) == 0);

constexpr vmcp::IntType numEnergies = 1 << 9;
// The denominator to obtain the number of energies when the variational parameters are used
constexpr vmcp::IntType vpNumEnergiesFactor = 1 << 3;
static_assert((numEnergies % vpNumEnergiesFactor) == 0);

// LF TODO: This is unused for now!
// constexpr bool testImpSamp = true;

constexpr vmcp::FPType minParamFactor = 0.33f;
constexpr vmcp::FPType maxParamFactor = 3;
constexpr vmcp::VarParam maxParDiff{20};
// Computes an interval for a variational parameter which is fairly large but allows the gradient descent to
// converge in a reasonable time
inline vmcp::Bound<vmcp::VarParam> NiceBound(vmcp::VarParam param, vmcp::FPType lowFactor,
                                             vmcp::FPType highFactor, vmcp::VarParam maxDiff) {
    vmcp::VarParam const low{std::max(param.val * lowFactor, param.val - maxDiff.val)};
    vmcp::VarParam const high{std::min(param.val * highFactor, param.val + maxDiff.val)};
    return vmcp::Bound<vmcp::VarParam>{low, high};
}

#endif
