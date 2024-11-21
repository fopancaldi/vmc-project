//!
//! @file vmcalgs.cpp
//! @brief Definition of the non-templated helper functions for the VMC algorithms
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//! @see vmcalgs.hpp
//! @see vmcalgs.inl
//!

#include "vmcalgs.inl"

namespace vmcp {

//! @brief Calculates the average and its variance
//! @param v The energies to be averaged
//! @return The average and its variance
VMCResult AvgAndVar_(std::vector<Energy> const &v) {
    assert(v.size() > 1);
    auto const size = std::ssize(v);
    Energy const avg{std::accumulate(v.begin(), v.end(), Energy{0},
                                     [](Energy e1, Energy e2) { return Energy{e1.val + e2.val}; })
                         .val /
                     static_cast<FPType>(size)};
    EnVariance const var{std::accumulate(v.begin(), v.end(), EnVariance{0},
                                         [avg](EnVariance ev, Energy e) {
                                             return EnVariance{ev.val + std::pow(e.val - avg.val, 2)};
                                         })
                             .val /
                         static_cast<FPType>(size * (size - 1))};
    return VMCResult{avg, var};
}

} // namespace vmcp
