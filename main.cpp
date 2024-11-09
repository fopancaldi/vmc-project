#include "src/vmcp.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace vmcp;

// The various features of main can be toggled here
constexpr std::array features = {true};

int main() {
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
        Bounds<1> const bounds = {Bound{-100, 100}};
        RandomGenerator gen{(std::random_device())()};
        VarParams<1> const initialAlpha{0.5f};
        Mass const mass{0.5f};

        for (FPType alphaVal = 0.1f; alphaVal <= 2; alphaVal += FPType{0.05f}) {
            VMCResult const vmcr = AvgAndVar_(Energies_(VMCEnAndPoss<1, 1, 1>(
                wavefHO, VarParams<1>{alphaVal}, secondDerHO, mass, potHO, bounds, numberEnergies, gen)));
            std::cout << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                      << vmcr.energy.val << " +/- " << std::sqrt(vmcr.variance.val) << '\n';
        }
        VMCResult const vmcrBest =
            VMCEnergy<1, 1, 1>(wavefHO, initialAlpha, secondDerHO, mass, potHO, bounds, numberEnergies, gen);
        std::cout << "Energy with the best alpha:\n"
                  << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                  << " +/- " << std::sqrt(vmcrBest.variance.val) << '\n';
    }
}
