#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "vmcp.hpp"

#include <functional>
#include <numbers>
#include <tuple>

constexpr vmcp::UIntType seed = 64826;
// TODO: Rename this
constexpr vmcp::FPType varianceTolerance = 1e-9f;

// TODO: Rename if WrappedVMCEnergies is renamed
TEST_CASE("Testing WrappedVMCEnergies_") {
    SUBCASE("1D harmonic oscillator") {
        struct WavefHO {
            vmcp::FPType m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                return std::pow(std::numbers::e_v<vmcp::FPType>,
                                -x[0][0].val * x[0][0].val * (m * omega / (2 * vmcp::hbar)));
            }
        };
        struct PotHO {
            vmcp::FPType m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x) const {
                return x[0][0].val * x[0][0].val * (m * omega * omega / 2);
            }
        };
        struct GradHO {
            vmcp::FPType m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                return (-x[0][0].val * (m * omega / (vmcp::hbar))) *
                       WavefHO{m, omega}(x, vmcp::VarParams<0>{});
            }
        };
        struct SecondDerHO {
            vmcp::FPType m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                return (std::pow((-x[0][0].val * (m * omega / (vmcp::hbar))), 2) +
                        (m * omega / (vmcp::hbar))) *
                       WavefHO{m, omega}(x, vmcp::VarParams<0>{});
            }
        };

        vmcp::IntType const numberEnergies = 100;
        vmcp::Bounds<1> const bounds = {vmcp::Bound{-100, 100}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::FPType const mInit = 0.1f;
        vmcp::FPType const omegaInit = 0.1f;
        vmcp::FPType const mStep = 0.2f;
        vmcp::FPType const omegaStep = 0.2f;
        vmcp::IntType const mIterations = 20;
        vmcp::IntType const omegaIterations = 20;
        vmcp::Mass const mass{1.f};
        WavefHO wavefHO{mInit, omegaInit};
        PotHO potHO{mInit, omegaInit};
        GradHO gradHO{mInit, omegaInit};
        SecondDerHO secondDerHO{mInit, omegaInit};
        std::array<GradHO, 1> gradHOArr = {gradHO};

        for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
            wavefHO.m = m_;
            potHO.m = m_;
            gradHO.m = m_;
            secondDerHO.m = m_;
            for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                 ++j, omega_ += omegaStep) {
                wavefHO.omega = omega_;
                potHO.omega = omega_;
                gradHO.omega = omega_;
                secondDerHO.omega = omega_;

                vmcp::Energy expectedEn{vmcp::hbar * omega_ / 2};
                vmcp::VMCResult vmcr =
                    vmcp::VMCEnergy<1, 1>(wavefHO, vmcp::VarParams<0>{}, false, gradHOArr, secondDerHO,
                                          //                    mass,
                                          potHO, bounds, numberEnergies, rndGen);

                // If due to numerical errors the variance is very small, use varianceTolerance instead
                // If the variance is negative and larger in absolute value than the tolerance, the check
                // will fail
                if (std::abs(vmcr.variance.val) < varianceTolerance) {
                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) < varianceTolerance);
                } else {
                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) < vmcr.variance.val);
                }
            }
        }
    }

    SUBCASE("1D potential box") {
        // l = length of the box
        struct WavefBox {
            vmcp::FPType m;
            vmcp::FPType l;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                return std::cos(std::numbers::pi_v<vmcp::FPType> * x[0][0].val / l);
            }
        };
        struct GradBox {
            vmcp::FPType m;
            vmcp::FPType l;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                return -(std::numbers::pi_v<vmcp::FPType> / l) *
                       std::sin(std::numbers::pi_v<vmcp::FPType> * x[0][0].val / l);
            }
        };
        struct SecondDerBox {
            vmcp::FPType m;
            vmcp::FPType l;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                return -std::pow((std::numbers::pi_v<vmcp::FPType> / l), 2) *
                       WavefBox{m, l}(x, vmcp::VarParams<0>{});
            }
        };

        vmcp::IntType const numberEnergies = 100;
        vmcp::RandomGenerator rndGen{seed};
        vmcp::FPType const mInit = 0.1f;
        vmcp::FPType const lInit = 0.5f;
        vmcp::FPType const mStep = 0.2f;
        vmcp::FPType const lStep = 0.2f;
        vmcp::IntType const mIterations = 20;
        vmcp::IntType const lIterations = 20;
        vmcp::Mass const mass = vmcp::Mass{1.f};
        WavefBox wavefBox{mInit, lInit};
        GradBox gradBox{mInit, lInit};
        SecondDerBox secondDerBox{mInit, lInit};
        std::array<GradBox, 1> gradBoxArr = {gradBox};

        for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
            wavefBox.m = m_;
            gradBox.m = m_;
            secondDerBox.m = m_;
            for (auto [j, l_] = std::tuple{vmcp::IntType{0}, lInit}; j != lIterations; ++j, l_ += lStep) {
                wavefBox.l = l_;
                gradBox.l = l_;
                secondDerBox.l = l_;

                vmcp::Energy expectedEn{1 / (2 * m_) *
                                        std::pow(vmcp::hbar * std::numbers::pi_v<vmcp::FPType> / l_, 2)};
                vmcp::VMCResult vmcr = vmcp::VMCEnergy<1, 1>(
                    wavefBox, vmcp::VarParams<0>{}, false, gradBoxArr, secondDerBox, mass,
                    [](vmcp::Positions<1, 1>) { return vmcp::FPType{0}; },
                    vmcp::Bounds<1>{vmcp::Bound{-l_ / 2, l_ / 2}}, numberEnergies, rndGen);

                // If due to numerical errors the variance is very small, use varianceTolerance instead
                // If the variance is negative and larger in absolute value than the tolerance, the check
                // will fail
                if (std::abs(vmcr.variance.val) < varianceTolerance) {
                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) < varianceTolerance);
                } else {
                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) < vmcr.variance.val);
                }
            }
        }
    }

    // TODO:
    SUBCASE("1D triangular well potential") {}

    // TODO:
    SUBCASE("1D radial Schroedinger eq") {}
}

/* FAKE TEST BEGINS HERE */

using namespace vmcp;

// This is just to test the algorithms before committing
// Will be removed later
// When using this test, it is suggested to comment out the others
// The lowest energy should be at alpha = 1, and at that value the variance should be really small (1e-6
// or smaller) For other values, the variance should be much larger (at least 1e-3), and the energy may be
// < 1, but within the error

TEST_CASE("main.cpp-like test case, will be removed") {
    auto wavefHO = [](Positions<1, 1> x, VarParams<1> alpha) {
        return std::pow(std::numbers::e_v<FPType>, -alpha[0].val * x[0][0].val * x[0][0].val / 2);
    };
    auto potHO = [](Positions<1, 1> x) { return x[0][0].val * x[0][0].val; };
    auto gradHO = [&wavefHO](Positions<1, 1> x, VarParams<1> alpha) {
        return -alpha[0].val * wavefHO(x, alpha);
    };
    std::array gradHOArr = {gradHO};
    auto secondDerHO = [&wavefHO](Positions<1, 1> x, VarParams<1> alpha) {
        return (std::pow(alpha[0].val * x[0][0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
    };

    vmcp::Mass mass = vmcp::Mass{1.f};
    int numberEnergies = 100;
    Bounds<1> bounds = {Bound{-100, 100}};
    RandomGenerator gen{(std::random_device())()};

    for (FPType alpha = 0.1f; alpha <= 2; alpha += FPType{0.05f}) {
        // Use gradHO directly
        VMCResult vmcr = VMCEnergy<1, 1, 1>(wavefHO, VarParams<1>{alpha}, true, gradHOArr, secondDerHO, mass,
                                            potHO, bounds, numberEnergies, gen);
        std::cout << "alpha: " << std::setprecision(3) << alpha << "\t\tenergy: " << std::setprecision(5)
                  << vmcr.energy.val << " +/- " << std::sqrt(vmcr.variance.val) << '\n';
    }
}
