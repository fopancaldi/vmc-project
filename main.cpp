#include "src/vmcp.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace vmcp;

// The various features of main can be toggled here
constexpr std::array features = {true};

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
}
