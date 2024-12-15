#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "test.hpp"
#include "vmcp.hpp"

TEST_CASE("Testing Statistics") {
    constexpr vmcp::FPType statisticsTolerance = 0.1f;

    constexpr vmcp::FPType gaussianMean = 0;
    constexpr vmcp::FPType gaussianStdDev = 1;
    constexpr vmcp::IntType numPoints = 1 << 10;

    std::normal_distribution<> dist(gaussianMean, gaussianStdDev);

    vmcp::RandomGenerator gen{seed};

    // Generate gaussian data points
    std::vector<vmcp::LocEnAndPoss<1, 1>> data(numPoints);
    for (vmcp::UIntType i = 0; i < numPoints; ++i) {
        data[i].localEn = vmcp::Energy{dist(gen)};
    }

    SUBCASE("Testing EvalBlocking") {
        std::vector<vmcp::LocEnAndPoss<1, 1>> testEnergies = {
            {vmcp::Energy{1}, 0}, {vmcp::Energy{2}, 0}, {vmcp::Energy{3}, 0}, {vmcp::Energy{4}, 0}};
        vmcp::BlockingResult blockingResults = EvalBlocking(testEnergies, 4);
        CHECK(blockingResults.means[0].val == vmcp::FPType{2.5f});
        CHECK(std::abs(blockingResults.stdDevs[0].val - 1) < statisticsTolerance);
    }

    SUBCASE("Testing BlockingOut") {
        vmcp::Energy const blockingStdDev =
            Statistics(data, vmcp::StatFuncType::blocking, bootstrapSamples, gen);
        CHECK(std::abs(blockingStdDev.val) < statisticsTolerance);
    }

    SUBCASE("Testing BootstrapAnalysis") {
        SUBCASE("Testing bootstrap with a small data vector") {
            std::vector<vmcp::LocEnAndPoss<1, 1>> testEnergies = {{vmcp::Energy{1}, 0},
                                                                  {vmcp::Energy{2}, 0},
                                                                  {vmcp::Energy{3}, 0},
                                                                  {vmcp::Energy{4}, 0},
                                                                  {vmcp::Energy{5}, 0}};
            vmcp::Energy bootstrapStdDev = BootstrapAnalysis(testEnergies, bootstrapSamples, gen);
            CHECK(std::abs(bootstrapStdDev.val - std::sqrt(vmcp::FPType{1} / 2)) < statisticsTolerance);
        }

        SUBCASE("Testing bootstrap with Gaussian data") {
            vmcp::Energy const bootstrapStdDev =
                Statistics(data, vmcp::StatFuncType::bootstrap, bootstrapSamples, gen);
            CHECK(std::abs(bootstrapStdDev.val) < statisticsTolerance);
        }
    }
}
