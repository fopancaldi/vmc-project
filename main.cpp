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
            VMCResult const vmcr = AvgAndVar_(LocalEnergies_(
                VMCLocEnAndPoss<1, 1, 1>(wavefHO, VarParams<1>{alpha}, std::array{secondDerHO},
                                         std::array{mass}, potHO, coorBounds, numberEnergies, gen)));
            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr.energy.val << " +/- "
                      << std::sqrt(vmcr.variance.val) << '\n';
        }

        ParamBounds<1> alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

        VMCResult const vmcrBest =
            VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, std::array{secondDerHO}, std::array{mass}, potHO,
                               coorBounds, numberEnergies, gen);
        std::cout << "Energy with the best alpha:\n"
                  << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                  << " +/- " << std::sqrt(vmcrBest.variance.val) << '\n';
    }

    // Feature 2
    // Just bugfixing
    if constexpr (features[1]) {
        constexpr vmcp::UIntType seed = 648265u;
        constexpr vmcp::IntType iterations = 64;
        constexpr vmcp::FPType minParamFactor = 0.8f;
        constexpr vmcp::FPType maxParamFactor = 1.5f;

        vmcp::IntType const numberEnergies = 1 << 7;
        vmcp::CoordBounds<1> const coordBound = {vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Mass const m_{7.4f};
        vmcp::FPType const omega_ = 4.2f;
        struct PotHO {
            vmcp::Mass m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x) const {
                return x[0][0].val * x[0][0].val * (m.val * omega * omega / 2);
            }
        };

        PotHO potHO{m_, omega_};
        auto const wavefHO{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
            return std::exp(-alpha[0].val * x[0][0].val * x[0][0].val);
        }};
        std::array laplHO{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
            return (std::pow(x[0][0].val * alpha[0].val, 2) - alpha[0].val) *
                   std::exp(-alpha[0].val * x[0][0].val * x[0][0].val);
        }};

        vmcp::VarParam bestParam{m_.val * omega_ / vmcp::hbar};
        vmcp::ParamBounds<1> const parBound{
            vmcp::Bound{bestParam * minParamFactor, bestParam * maxParamFactor}};
        vmcp::Energy const expectedEn{vmcp::hbar * omega_ / 2};
        vmcp::VMCResult const vmcr = vmcp::VMCEnergy<1, 1, 1>(wavefHO, parBound, laplHO, std::array{m_},
                                                              potHO, coordBound, numberEnergies, rndGen);
    }
}
