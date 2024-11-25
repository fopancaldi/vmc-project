//!
//! @file statistics.cpp
//! @brief Definition of the non-templated statistics algorithms
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the non-templated statistics algorithms declared in the header.
//! @see statistics.hpp
//!

#include "statistics.hpp"

#include <boost/math/distributions/normal.hpp>

namespace vmcp {

//! @defgroup user-functions User functions
//! @brief The functions that are meant to be called by the user
//!
//! @{

//! @brief Calculate the confidence interval of a mean given the correspondent std. dev. for a certain
//! confidence level
//! @param mean The mean
//! @param stdDev The standard deviation
//! @param confLevel The confidence level, supplied by user
//! @return The confidence interval

ConfInterval GetConfInt(Energy mean, Energy stdDev, FPType confLevel) {
    ConfInterval confInterval;
    boost::math::normal dist(mean.val, stdDev.val);

    FPType probability = 1 - (1 - confLevel / 100) / 2;
    FPType z = boost::math::quantile(dist, probability);

    confInterval.min = mean - stdDev * z;
    confInterval.max = mean + stdDev * z;
    return confInterval;
}

//! @}

} // namespace vmcp
