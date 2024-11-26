//!
//! @file test.hpp
//! @brief Definition of constants and generic functions used in the tests
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!

#ifndef VMCP_TESTS_HPP
#define VMCP_TESTS_HPP

#include "types.hpp"

// Chosen at random, but fixed to guarantee reproducibility of failed tests
constexpr vmcp::UIntType seed = 648265u;
constexpr vmcp::IntType iterations = 64;
// FP TODO: Rename this
constexpr vmcp::IntType allowedStdDevs = 25;
// If the standard deviation is smaller than this, it is highly probable that numerical errors were
// non-negligible
// FP TODO: If you declare this as constexpr, intellisense complains but the program compiles
// So should this be constexpr?
const vmcp::Energy stdDevTolerance{std::numeric_limits<vmcp::FPType>::epsilon() * 100};
// FP TODO: Rename
constexpr vmcp::IntType varParamsFactor = 32;
// FP TODO: One pair of brackets can probably be removed here
// Learn the priority of the operations and adjust the rest of the code too
static_assert((iterations % varParamsFactor) == 0);
// LF TODO: This is unused for now!
// constexpr bool testImpSamp = true;
// Rules out situations where both the VMC energy and the variance are extremely large, so the test
// succeds
constexpr vmcp::Energy vmcEnergyTolerance{0.5f};
// FP TODO: Explain
constexpr vmcp::FPType minParamFactor = 0.33f;
constexpr vmcp::FPType maxParamFactor = 3;
constexpr vmcp::VarParam maxParDiff{20};
constexpr vmcp::IntType numEnergies = 1 << 10;
const std::string logFileName = "../artifacts/test-log.txt";

// Computes an interval for a variational parameter which is fairly large but allows the gradient descent to
// converge in a reasonable time
inline vmcp::Bound<vmcp::VarParam> NiceBound(vmcp::VarParam param, vmcp::FPType lowFactor,
                                             vmcp::FPType upFactor, vmcp::VarParam maxDiff) {
    vmcp::VarParam const low{std::max(param.val * lowFactor, param.val - maxDiff.val)};
    vmcp::VarParam const up{std::min(param.val * upFactor, param.val + maxDiff.val)};
    return vmcp::Bound<vmcp::VarParam>{low, up};
}

#endif
