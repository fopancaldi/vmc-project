#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "test.hpp"
#include "vmcp.hpp"

TEST_CASE("Testing Statistics") {
    constexpr vmcp::FPType statisticsTolerance = 0.1f;

    constexpr vmcp::FPType gaussianMean = 0.f;
    constexpr vmcp::FPType gaussianStdDev = 1.f;
    constexpr vmcp::IntType numPoints = 1 << 10;

    std::normal_distribution<> dist(gaussianMean, gaussianStdDev);

    vmcp::RandomGenerator gen{seed};
    vmcp::IntType const numSamples = 10000;

    // Generate gaussian data points
    std::vector<vmcp::LocEnAndPoss<1, 1>> data(numPoints);
    for (vmcp::UIntType i = 0; i < numPoints; ++i) {
        data[i].localEn = vmcp::Energy{dist(gen)};
    }

    SUBCASE("Testing EvalBlocking") {
        std::vector<vmcp::LocEnAndPoss<1, 1>> testEnergies = {
            {vmcp::Energy{1.}, 0.}, {vmcp::Energy{2.}, 0.}, {vmcp::Energy{3.}, 0.}, {vmcp::Energy{4.}, 0.}};
        vmcp::BlockingResult blockingResults = EvalBlocking(testEnergies, numEnergies);
        CHECK(blockingResults.means[0].val == 2.5);
        CHECK(std::abs(blockingResults.stdDevs[0].val - 1.) < statisticsTolerance);
    }

    SUBCASE("Testing BlockingOut") {
        vmcp::PartialVMCResult const blockingRes =
            Statistics(data, vmcp::StatFuncType::blocking, numSamples, gen);
        CHECK(std::abs(blockingRes.energy.val) < statisticsTolerance);
        CHECK(std::abs(blockingRes.stdDev.val) < statisticsTolerance);
    }

    SUBCASE("Testing BootstrapAnalysis") {
        SUBCASE("Testing bootstrap with a small data vector") {
            std::vector<vmcp::LocEnAndPoss<1, 1>> testEnergies = {{vmcp::Energy{1.f}, 0.},
                                                                  {vmcp::Energy{2.f}, 0.},
                                                                  {vmcp::Energy{3.f}, 0.},
                                                                  {vmcp::Energy{4.f}, 0.},
                                                                  {vmcp::Energy{5.f}, 0.}};
            vmcp::PartialVMCResult bootstrapResults = BootstrapAnalysis(testEnergies, numSamples, gen);
            CHECK(std::abs(bootstrapResults.energy.val - 3.) < statisticsTolerance);
            CHECK(std::abs(bootstrapResults.stdDev.val - std::sqrt(1. / 2.)) < statisticsTolerance);
        }

        SUBCASE("Testing bootstrap with Gaussian data") {
            vmcp::PartialVMCResult const bootstrapRes =
                Statistics(data, vmcp::StatFuncType::bootstrap, numSamples, gen);
            CHECK(std::abs(bootstrapRes.energy.val) < statisticsTolerance);
            CHECK(std::abs(bootstrapRes.stdDev.val) < statisticsTolerance);
        }
    }
}
