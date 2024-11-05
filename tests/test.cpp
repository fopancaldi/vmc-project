#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "vmcp.hpp"

#include <functional>
#include <numbers>
#include <tuple>

constexpr vmcp::UIntType seed = 64826;
// TODO: Rename this
constexpr vmcp::FPType varianceTolerance = 1e-9f;
constexpr bool testImpSamp = true;

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
                return (std::pow((x[0][0].val * (m * omega / (vmcp::hbar))), 2) -
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
        vmcp::IntType const mIterations = 40;
        vmcp::IntType const omegaIterations = 40;
        vmcp::Mass mass{1.f};
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
                mass.val = m_;

                vmcp::Energy expectedEn{vmcp::hbar * omega_ / 2};
                vmcp::VMCResult vmcr =
                    vmcp::VMCEnergy<1, 1>(wavefHO, vmcp::VarParams<0>{}, testImpSamp, gradHOArr, secondDerHO,
                                          mass, potHO, bounds, numberEnergies, rndGen);

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
        vmcp::IntType const mIterations = 40;
        vmcp::IntType const lIterations = 40;
        vmcp::Mass mass = vmcp::Mass{1.f};
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
                mass.val = m_;

                vmcp::Energy expectedEn{1 / (2 * m_) *
                                        std::pow(vmcp::hbar * std::numbers::pi_v<vmcp::FPType> / l_, 2)};
                vmcp::VMCResult vmcr = vmcp::VMCEnergy<1, 1>(
                    wavefBox, vmcp::VarParams<0>{}, testImpSamp, gradBoxArr, secondDerBox, mass,
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

    // TODO: to be finished & adding references
    // SUBCASE("1D triangular well potential") {
    //     struct WavefTrg {
    //         vmcp::FPType alpha;
    //         vmcp::FPType m;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
    //             return std::pow(std::numbers::e_v<vmcp::FPType>,
    //                             -std::pow(2 * m * alpha / (vmcp::hbar * vmcp::hbar), 1 / 3) *
    //                                 std::abs(x[0][0].val));
    //         }
    //     };
    //     struct PotTrg {
    //         vmcp::FPType alpha;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> x) const { return alpha * std::abs(x[0][0].val);
    //         }
    //     };
    //     struct GradTrg {
    //         vmcp::FPType alpha;
    //         vmcp::FPType m;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
    //             return -std::pow(2 * m * alpha / (vmcp::hbar * vmcp::hbar), 1 / 3) *
    //                    std::signbit(x[0][0].val) * WavefTrg{alpha, m}(x, vmcp::VarParams<0>{});
    //         }
    //     };
    //     struct SecondDerTrg {
    //         vmcp::FPType alpha;
    //         vmcp::FPType m;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
    //             return std::pow(2 * m * alpha / (vmcp::hbar * vmcp::hbar), 2 / 3) *
    //                    WavefTrg{alpha, m}(x, vmcp::VarParams<0>{});
    //         }
    //     };
    //
    //    vmcp::IntType const numberEnergies = 100;
    //    vmcp::Bounds<1> const bounds = {vmcp::Bound{-100, 100}};
    //    vmcp::RandomGenerator rndGen{seed};
    //    vmcp::FPType const alphaInit = 0.1f;
    //    vmcp::FPType const mInit = 0.5f;
    //    vmcp::FPType const alphaStep = 0.2f;
    //    vmcp::FPType const mStep = 0.2f;
    //    vmcp::IntType const alphaIterations = 10;
    //    vmcp::IntType const mIterations = 10;
    //    vmcp::Mass mass = vmcp::Mass{1.f};
    //    WavefTrg wavefTrg{alphaInit, mInit};
    //    PotTrg potTrg{alphaInit};
    //    GradTrg gradTrg{alphaInit, mInit};
    //    SecondDerTrg secondDerTrg{alphaInit, mInit};
    //    std::array<GradTrg, 1> gradTrgArr = {gradTrg};
    //    for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
    //        wavefTrg.m = m_;
    //        gradTrg.m = m_;
    //        secondDerTrg.m = m_;
    //        for (auto [j, alpha_] = std::tuple{vmcp::IntType{0}, alphaInit}; j != alphaIterations;
    //             ++j, alpha_ += alphaStep) {
    //            wavefTrg.alpha = alpha_;
    //            potTrg.alpha = alpha_;
    //            gradTrg.alpha = alpha_;
    //            secondDerTrg.alpha = alpha_;
    //            mass.val = m_;
    //
    //            vmcp::Energy expectedEn{std::pow(2 * m_, -1 / 3) *
    //                                    std::pow(alpha_ * vmcp::hbar * 3 / 4, 2 / 3)};
    //            vmcp::VMCResult vmcr =
    //                vmcp::VMCEnergy<1, 1>(wavefTrg, vmcp::VarParams<0>{}, testImpSamp, gradTrgArr,
    //                                      secondDerTrg, mass, potTrg, bounds, numberEnergies, rndGen);
    //
    //            if (std::abs(vmcr.variance.val) < varianceTolerance) {
    //                CHECK(std::abs(vmcr.energy.val - expectedEn.val) < varianceTolerance);
    //            } else {
    //                CHECK(std::abs(vmcr.energy.val - expectedEn.val) < vmcr.variance.val);
    //            }
    //        }
    //    }
    //}

    // TODO: to be finished & adding references
    // SUBCASE("1D radial Schroedinger eq") {
    //     struct WavefRad {
    //         vmcp::FPType k;
    //         vmcp::FPType m;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
    //             return r[0][0].val * std::pow(std::numbers::e_v<vmcp::FPType>,
    //                                           -m * k / (vmcp::hbar * vmcp::hbar) * r[0][0].val);
    //         }
    //     };
    //     struct PotRad {
    //         vmcp::FPType k;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> r) const { return (-k / r[0][0].val); }
    //     };
    //     struct GradRad {
    //         vmcp::FPType k;
    //         vmcp::FPType m;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
    //             return (1 - m * k / (vmcp::hbar * vmcp::hbar) * r[0][0].val) *
    //                    WavefRad{k, m}(r, vmcp::VarParams<0>{});
    //         }
    //     };
    //     struct SecondDerRad {
    //         vmcp::FPType k;
    //         vmcp::FPType m;
    //         vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
    //             return (std::pow(1 - r[0][0].val * m * k / (vmcp::hbar * vmcp::hbar), 2) -
    //                     m * k / (vmcp::hbar * vmcp::hbar)) *
    //                    WavefRad{k, m}(r, vmcp::VarParams<0>{});
    //         }
    //     };
    //
    //    vmcp::IntType const numberEnergies = 100;
    //    vmcp::Bounds<1> const bounds = {vmcp::Bound{0, 100}};
    //    vmcp::RandomGenerator rndGen{seed};
    //    vmcp::FPType const kInit = 1.2f;
    //    vmcp::FPType const mInit = 0.3f;
    //    vmcp::FPType const kStep = 0.1f;
    //    vmcp::FPType const mStep = 0.2f;
    //    vmcp::IntType const kIterations = 10;
    //    vmcp::IntType const mIterations = 10;
    //    vmcp::Mass mass = vmcp::Mass{1.f};
    //    WavefRad wavefRad{kInit, mInit};
    //    PotRad potRad{kInit};
    //    GradRad gradRad{kInit, mInit};
    //    SecondDerRad secondDerRad{kInit, mInit};
    //    std::array<GradRad, 1> gradRadArr = {gradRad};
    //
    //    for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
    //        wavefRad.m = m_;
    //        gradRad.m = m_;
    //        secondDerRad.m = m_;
    //        for (auto [j, k_] = std::tuple{vmcp::IntType{0}, kInit}; j != kIterations; ++j, k_ += kStep) {
    //            wavefRad.k = k_;
    //            potRad.k = k_;
    //            gradRad.k = k_;
    //            secondDerRad.k = k_;
    //            mass.val = m_;
    //
    //            vmcp::Energy expectedEn{-m_ * std::pow((1.6021766 * std::pow(10, -19)), 4) /
    //                                    (2 * vmcp::hbar)};
    //            vmcp::VMCResult vmcr =
    //                vmcp::VMCEnergy<1, 1>(wavefRad, vmcp::VarParams<0>{}, testImpSamp, gradRadArr,
    //                                      secondDerRad, mass, potRad, bounds, numberEnergies, rndGen);
    //
    //            if (std::abs(vmcr.variance.val) < varianceTolerance) {
    //                CHECK(std::abs(vmcr.energy.val - expectedEn.val) < varianceTolerance);
    //            } else {
    //                CHECK(std::abs(vmcr.energy.val - expectedEn.val) < vmcr.variance.val);
    //            }
    //        }
    //    }
    //}
}
