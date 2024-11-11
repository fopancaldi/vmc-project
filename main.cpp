#include "src/vmcp.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace vmcp;

// The various features of main can be toggled here
constexpr std::array features = {true, false};

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
        CoordBounds<1> const coorBounds = {Bound{Coordinate{-100}, Coordinate{100}}};
        RandomGenerator gen{(std::random_device())()};
        Mass const mass{0.5f};

        for (VarParam alpha{0.1f}; alpha.val <= 2; alpha.val += FPType{0.05f}) {
            VMCResult const vmcr = AvgAndVar_(LocalEnergies_(VMCLocEnAndPoss<1, 1, 1>(
                wavefHO, VarParams<1>{alpha}, secondDerHO, mass, potHO, coorBounds, numberEnergies, gen)));
            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr.energy.val << " +/- "
                      << std::sqrt(vmcr.variance.val) << '\n';
        }

        ParamBounds<1> alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

        VMCResult const vmcrBest = VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, secondDerHO, mass, potHO,
                                                      coorBounds, numberEnergies, gen);
        std::cout << "Energy with the best alpha:\n"
                  << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                  << " +/- " << std::sqrt(vmcrBest.variance.val) << '\n';
    }

    // Feature 2
    // Just bugfixing
    if constexpr (features[1]) {
        vmcp::IntType const numberEnergies = 100;
        vmcp::CoordBounds<1> const bounds = {vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}}};
        vmcp::RandomGenerator rndGen{1};
        struct PotHO {
            vmcp::Mass m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x) const {
                return x[0][0].val * x[0][0].val * (m.val * omega * omega / 2);
            }
        };
        Mass mass{5.8f};
        FPType omega{7.4f};
        vmcp::VarParams<1> const initialParam{vmcp::VarParam{5 * mass.val * omega / vmcp::hbar}};
        auto const wavefHO{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
            return std::exp(-alpha[0].val * x[0][0].val * x[0][0].val);
        }};
        auto const secondDerHO{[&wavefHO](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
            // FP TODO: Constructing a new object might be expensive
            return (std::pow(x[0][0].val * alpha[0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
        }};
        PotHO potHO{mass, omega};
        vmcp::VMCResult const vmcr = vmcp::VMCEnergy<1, 1, 1>(wavefHO, initialParam, secondDerHO, mass, potHO,
                                                              bounds, numberEnergies, rndGen);

        std::cout << vmcr.energy.val << '\t' << std::sqrt(vmcr.variance.val) << '\n';
    }
}
