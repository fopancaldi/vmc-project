#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "vmcp.hpp"

#include <numbers>
#include <tuple>

// FP TODO: A lot of duplicated for loops

constexpr vmcp::UIntType seed = 64826;
constexpr vmcp::IntType iterations = 40;
// FP TODO: Rename this
constexpr vmcp::IntType allowedStdDevs = 5;
// FP TODO: Explain, and maybe rename
constexpr vmcp::FPType stdDevTolerance = 1e-9f;
// FP TODO: Rename
constexpr vmcp::IntType varParamsFactor = 8;
// FP TODO: One pair of brackets can probably be removed here
// Learn th epriority of the operations and adjust the resto of the code too
static_assert((iterations % varParamsFactor) == 0);

// FP TODO: Rename if WrappedVMCEnergies is renamed
TEST_CASE("Testing WrappedVMCEnergies_") {
    SUBCASE("1D harmonic oscillator") {
        vmcp::IntType const numberEnergies = 100;
        vmcp::Bounds<1> const bounds = {vmcp::Bound{-100, 100}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Mass const mInit{1.f};
        vmcp::FPType const omegaInit = 1.f;
        vmcp::FPType const mStep = 0.2f;
        vmcp::FPType const omegaStep = 0.2f;
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const omegaIterations = iterations;
        struct PotHO {
            vmcp::Mass m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x) const {
                return x[0][0].val * x[0][0].val * (m.val * omega * omega / 2);
            }
        };

        SUBCASE("No variational parameters") {
            struct WavefHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    return std::exp(-x[0][0].val * x[0][0].val * (m.val * omega / (2 * vmcp::hbar)));
                }
            };
            struct SecondDerHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    // FP TODO: Constructing a new object might be expensive
                    return (std::pow(x[0][0].val * m.val * omega / vmcp::hbar, 2) -
                            (m.val * omega / vmcp::hbar)) *
                           WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };

            WavefHO wavefHO{mInit.val, omegaInit};
            PotHO potHO{mInit.val, omegaInit};
            SecondDerHO secondDerHO{mInit, omegaInit};

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_.val += mStep) {
                wavefHO.m = m_;
                potHO.m = m_;
                secondDerHO.m = m_;
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                     ++j, omega_ += omegaStep) {
                    wavefHO.omega = omega_;
                    potHO.omega = omega_;
                    secondDerHO.omega = omega_;

                    vmcp::Energy expectedEn{vmcp::hbar * omega_ / 2};
                    vmcp::VMCResult vmcr =
                        vmcp::VMCEnergy<1, 1, 0>(wavefHO, vmcp::VarParams<0>{}, secondDerHO, m_, potHO,
                                                 bounds, numberEnergies, rndGen);

                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) <
                          std::max((allowedStdDevs * std::sqrt(vmcr.variance.val)), stdDevTolerance));
                }
            }
        }

        SUBCASE("One variational parameter") {
            auto const wavefHO{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
                return std::exp(-alpha[0].val * x[0][0].val * x[0][0].val);
            }};
            auto const secondDerHO{[&wavefHO](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
                // FP TODO: Constructing a new object might be expensive
                return (std::pow(x[0][0].val * alpha[0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
            }};
            PotHO potHO{mInit.val, omegaInit};
            // Ensures that the initial parameter is sufficiently far from the best parameter, which is m *
            // omega / hbar
            vmcp::IntType const initialParamFactor = 5;

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations / varParamsFactor;
                 ++i, m_.val += mStep * varParamsFactor) {
                potHO.m = m_;
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit};
                     j != omegaIterations / varParamsFactor; ++j, omega_ += omegaStep * varParamsFactor) {
                    potHO.omega = omega_;

                    vmcp::VarParams<1> const initialParam{
                        vmcp::VarParam{initialParamFactor * m_.val * omega_ / vmcp::hbar}};
                    vmcp::Energy expectedEn{vmcp::hbar * omega_ / 2};
                    vmcp::VMCResult vmcr = vmcp::VMCEnergy<1, 1, 1>(wavefHO, initialParam, secondDerHO, m_,
                                                                    potHO, bounds, numberEnergies, rndGen);

                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) <
                          std::max((allowedStdDevs * std::sqrt(vmcr.variance.val)), stdDevTolerance));
                }
            }
        }
    }

    SUBCASE("1D potential box") {
        // l = length of the box
        vmcp::IntType const numberEnergies = 100;
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Mass const mInit{0.1f};
        vmcp::FPType const lInit = 0.5f;
        vmcp::FPType const mStep = 0.2f;
        vmcp::FPType const lStep = 0.2f;
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const lIterations = iterations;
        auto const potBox{[](vmcp::Positions<1, 1>) { return vmcp::FPType{0}; }};

        SUBCASE("No variational parameters") {
            struct WavefBox {
                vmcp::FPType l;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    return std::cos(std::numbers::pi_v<vmcp::FPType> * x[0][0].val / l);
                }
            };
            struct SecondDerBox {
                vmcp::Mass m;
                vmcp::FPType l;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    // FP TODO: Constructing a new object might be expensive
                    return -std::pow(std::numbers::pi_v<vmcp::FPType> / l, 2) *
                           WavefBox{l}(x, vmcp::VarParams<0>{});
                }
            };

            WavefBox wavefBox{lInit};
            SecondDerBox secondDerBox{mInit, lInit};

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_.val += mStep) {
                secondDerBox.m = m_;
                for (auto [j, l_] = std::tuple{vmcp::IntType{0}, lInit}; j != lIterations; ++j, l_ += lStep) {
                    wavefBox.l = l_;
                    secondDerBox.l = l_;

                    vmcp::Bounds<1> bound{vmcp::Bound{-l_ / 2, l_ / 2}};
                    vmcp::Energy expectedEn{1 / (2 * m_.val) *
                                            std::pow(vmcp::hbar * std::numbers::pi_v<vmcp::FPType> / l_, 2)};
                    vmcp::VMCResult vmcr =
                        vmcp::VMCEnergy<1, 1, 0>(wavefBox, vmcp::VarParams<0>{}, secondDerBox, m_, potBox,
                                                 bound, numberEnergies, rndGen);

                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) <
                          std::max((allowedStdDevs * std::sqrt(vmcr.variance.val)), stdDevTolerance));
                }
            }
        }

        /* // FP TODO: Find a nice trial wavefunction
        SUBCASE("One variational parameter") {
            auto const wavefBox{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
                return std::cos(alpha[0].val * x[0][0].val);
            }};
            auto const secondDerBox{[&wavefBox](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
                // FP TODO: Constructing a new object might be expensive
                return -std::pow(alpha[0].val, 2) * wavefBox(x, alpha);
            }};
            // Ensures that the initial parameter is sufficiently far from the best parameter, which is pi / l
            vmcp::IntType const initialParamFactor = 5;

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations / varParamsFactor;
                 ++i, m_.val += mStep * varParamsFactor) {
                for (auto [j, l_] = std::tuple{vmcp::IntType{0}, lInit}; j != lIterations / varParamsFactor;
                     ++j, l_ += lStep * varParamsFactor) {
                    vmcp::VarParams<1> const initialParam{
                        vmcp::VarParam{initialParamFactor * std::numbers::pi_v<vmcp::FPType> / l_}};

                    vmcp::Bounds<1> bound{vmcp::Bound{-l_ / 2, l_ / 2}};
                    vmcp::Energy expectedEn{1 / (2 * m_.val) *
                                            std::pow(vmcp::hbar * std::numbers::pi_v<vmcp::FPType> / l_, 2)};
                    vmcp::VMCResult vmcr = vmcp::VMCEnergy<1, 1, 1>(wavefBox, initialParam, secondDerBox, m_,
                                                                    potBox, bound, numberEnergies, rndGen);

                    CHECK(std::abs(vmcr.energy.val - expectedEn.val) <
                          std::max((allowedStdDevs * std::sqrt(vmcr.variance.val)), stdDevTolerance));
                }
            }
        } */
    }

    // TODO:
    SUBCASE("1D triangular well potential") {}

    // TODO:
    SUBCASE("1D radial Schroedinger eq") {}
}
