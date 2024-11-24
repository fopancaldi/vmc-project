#include "src/vmcp.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace vmcp;

// The various features of main can be toggled here
constexpr std::array features = {false, true};

int main() {
    RandomGenerator gen{(std::random_device())()};

    // Feature 1:
    // Various tests with the harmonic oscillator
    if constexpr (features[0]) {
        // It is not normalized, but it doesn't matter
        auto const wavefHO{[](Positions<1, 1> x, VarParams<1> alpha) {
            return std::pow(std::numbers::e_v<FPType>, -alpha[0].val * x[0][0].val * x[0][0].val / 2);
        }};
        auto const potHO{[](Positions<1, 1> x) { return x[0][0].val * x[0][0].val; }};
        auto const secondDerHO{[&wavefHO](Positions<1, 1> x, VarParams<1> alpha) {
            return (std::pow(alpha[0].val * x[0][0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
        }};

        int const numberEnergies = 100;
        CoordBounds<1> const coorBounds = {Bound{Coordinate{-100}, Coordinate{100}}};
        Mass const mass{0.5f};

        for (VarParam alpha{0.1f}; alpha.val <= 2; alpha.val += FPType{0.05f}) {
            VMCResult const vmcr = MeanAndErr_(
                VMCLocEnAndPoss<1, 1, 1>(wavefHO, VarParams<1>{alpha}, std::array{secondDerHO},
                                         std::array{mass}, potHO, coorBounds, numberEnergies, gen));
            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr.energy.val << " +/- " << vmcr.stdDev.val
                      << '\n';
        }

        ParamBounds<1> alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

        VMCResult const vmcrBest =
            VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, std::array{secondDerHO}, std::array{mass}, potHO,
                               coorBounds, numberEnergies, gen);
        std::cout << "Energy with the best alpha:\n"
                  << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                  << " +/- " << vmcrBest.stdDev.val << '\n';
    }

    // Feature 2:
    // Various tests with the harmonic oscillator
    if constexpr (features[1]) {
        // It is not normalized, but it doesn't matter
        auto const wavefHO{[](Positions<1, 1> x, VarParams<1> alpha) {
            return std::pow(std::numbers::e_v<FPType>, -alpha[0].val * x[0][0].val * x[0][0].val / 2);
        }};
        auto const potHO{[](Positions<1, 1> x) { return x[0][0].val * x[0][0].val; }};
        auto const secondDerHO{[&wavefHO](Positions<1, 1> x, VarParams<1> alpha) {
            return (std::pow(alpha[0].val * x[0][0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
        }};

        IntType const numberEnergies = 1 << 10;
        CoordBounds<1> const coorBounds = {Bound{Coordinate{-100}, Coordinate{100}}};
        Mass const mass{0.5f};

        // Number of samples for bootsrapping
        IntType const numSamples = 1000;

        // Level of confidence for confidence interval
        FPType const confLevel = 95.f;

        for (VarParam alpha{0.1f}; alpha.val <= 2; alpha.val += FPType{0.05f}) {
            std::vector<LocEnAndPoss<1, 1>> const locEnPos =
                VMCLocEnAndPoss<1, 1, 1>(wavefHO, VarParams<1>{alpha}, std::array{secondDerHO},
                                         std::array{mass}, potHO, coorBounds, numberEnergies, gen);

            VMCResult const vmcr1 = Statistics(locEnPos, StatFuncType::blocking, numSamples, gen);
            VMCResult const vmcr2 = Statistics(locEnPos, StatFuncType::bootstrap, numSamples, gen);

            ConfInterval confInt1 = GetConfInt(vmcr1.energy, vmcr1.stdDev, confLevel);
            ConfInterval confInt2 = GetConfInt(vmcr2.energy, vmcr2.stdDev, confLevel);

            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr1.energy.val << " +/- "
                      << vmcr1.stdDev.val << "\tconf. interval of " << confLevel
                      << "%: " << std::setprecision(3) << confInt1.min.val << " -- " << confInt1.max.val
                      << '\n';
            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr2.energy.val << " +/- "
                      << vmcr2.stdDev.val << "\tconf. interval of " << confLevel
                      << "%: " << std::setprecision(3) << confInt2.min.val << " -- " << confInt2.max.val
                      << '\n';
        }

        ParamBounds<1> alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

        VMCResult const vmcrBest =
            VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, std::array{secondDerHO}, std::array{mass}, potHO,
                               coorBounds, numberEnergies, gen);
        std::cout << "Energy with the best alpha:\n"
                  << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                  << " +/- " << vmcrBest.stdDev.val << '\n';
    }
}
