//!
//! @file statistics.hpp
//! @brief Statistical methods header
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the declarations of statistical analysis algorithms that are meant to be called by the user.
//! Does NOT contain the descritpions of said algorithms.
//! To improve readability, the implementation of the templated functions is in the .inl file.
//! @see statistics.inl
//!

#ifndef VMCPROJECT_STATISTICS_HPP
#define VMCPROJECT_STATISTICS_HPP

#include "types.hpp"

namespace vmcp {

template <Dimension D, ParticNum N>
Energy Statistics(std::vector<LocEnAndPoss<D, N>> const &, StatFuncType, IntType const &, RandomGenerator &);

ConfInterval GetConfInt(Energy, Energy, FPType);

} // namespace vmcp

#include "statistics.inl"

#endif
