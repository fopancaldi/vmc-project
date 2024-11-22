// Statistical analysis algorithms definitions

#ifndef VMCPROJECT_STATISTICS_HPP
#define VMCPROJECT_STATISTICS_HPP

#include "types.hpp"

namespace vmcp {

// Divides dataset into multiple blocks with fixed block size, then evaluate means of each block and
// carries on a statistical analysis of this vectors (one for each block size) of means
VMCResult BlockingOut(std::vector<Energy> const &energies);

// Samples dataset with replacement multiple times, then evaluates mean of each sample and
// carries on a statistical analysis of this vector of means
VMCResult BootstrapAnalysis(std::vector<Energy> const &energies, IntType const numSamples,
                            RandomGenerator &gen);

ConfInterval GetConfInt(FPType mean, FPType stdDev);

} // namespace vmcp

#endif
