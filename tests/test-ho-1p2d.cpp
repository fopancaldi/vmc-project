//!
//! @file test-ho-1p2d.cpp
//! @brief Tests for the harmonic oscillator system, one particle in two dimensions
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "test.hpp"
#include "vmcp.hpp"

#include <chrono>
#include <fstream>
#include <numbers>
#include <string>
#include <tuple>

TEST_CASE("Testing the harmonic oscillator" * doctest::may_fail()) {
    std::ofstream file_stream;
    file_stream.open(logFilePath, std::ios_base::app);
    SUBCASE("2D harmonic oscillator, one particle") {
        vmcp::CoordBounds<2> const coordBounds = {vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}},
                                                  vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Masses<1> const mInit{1.f};
        vmcp::FPType const omegaInit = 1.f;
        vmcp::Mass const mStep{0.2f};
        vmcp::FPType const omegaStep = 0.2f;
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const omegaIterations = iterations;
        struct PotHO {
            vmcp::Mass m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<2, 1> x) const {
                return (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) * m.val * omega * omega / 2;
            }
        };
        vmcp::FPType const derivativeStep = coordBounds[0].Length().val / derivativeStepDenom;

        SUBCASE("No variational parameters") {
            struct WavefHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<2, 1> x, vmcp::VarParams<0>) const {
                    return std::exp(-(std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) * m.val * omega /
                                    (2 * vmcp::hbar));
                }
            };
            struct FirstDerHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::IntType dimension;
                FirstDerHO(vmcp::Mass m_, vmcp::FPType omega_, vmcp::IntType dimension_)
                    : m{m_}, omega{omega_}, dimension{dimension_} {
                    assert(dimension >= 0);
                    assert(dimension <= 1);
                }
                vmcp::FPType operator()(vmcp::Positions<2, 1> x, vmcp::VarParams<0>) const {
                    vmcp::UIntType uDim = static_cast<vmcp::UIntType>(dimension);
                    return -x[0][uDim].val * m.val * omega / vmcp::hbar *
                           WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };
            struct LaplHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<2, 1> x, vmcp::VarParams<0>) const {
                    return (m.val * omega / vmcp::hbar *
                                (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) -
                            2) *
                           m.val * omega / vmcp::hbar * WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };
            PotHO potHO{mInit[0], omegaInit};
            WavefHO wavefHO{mInit[0], omegaInit};
            vmcp::Gradients<2, 1, FirstDerHO> gradHO = {FirstDerHO{mInit[0], omegaInit, 0},
                                                        FirstDerHO{mInit[0], omegaInit, 1}};
            vmcp::Laplacians<1, LaplHO> laplHO{mInit[0], omegaInit};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                 ++i, m_[0].val += mStep.val) {
                potHO.m = m_[0];
                wavefHO.m = m_[0];
                gradHO[0][0].m = m_[0];
                gradHO[0][1].m = m_[0];
                laplHO[0].m = m_[0];
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                     ++j, omega_ += omegaStep) {
                    wavefHO.omega = omega_;
                    potHO.omega = omega_;
                    laplHO[0].omega = omega_;

                    vmcp::Energy const expectedEn{vmcp::hbar * omega_};
                    std::string const logMessage =
                        "mass: " + std::to_string(m_[0].val) + ", ang. vel.: " + std::to_string(omega_);

                    SUBCASE("Metropolis algorithm, analytical derivative") {
                        vmcp::VMCResult const vmcr = vmcp::VMCEnergy<2, 1, 0>(
                            wavefHO, vmcp::ParamBounds<0>{}, laplHO, std::array{m_}, potHO, coordBounds,
                            numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                    SUBCASE("Metropolis algorithm, numerical derivative") {
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<2, 1, 0>(
                            wavefHO, vmcp::ParamBounds<0>{}, false, derivativeStep, std::array{m_}, potHO,
                            coordBounds, numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                    SUBCASE("Importance algorithm sampling, analytical derivative") {
                        vmcp::VMCResult const vmcr = vmcp::VMCEnergy<2, 1, 0>(
                            wavefHO, vmcp::ParamBounds<0>{}, gradHO, laplHO, std::array{m_}, potHO,
                            coordBounds, numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                    /* SUBCASE("Importance sampling algorithm, numerical derivative") {
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<2, 1, 0>(
                            wavefHO, vmcp::ParamBounds<0>{}, true, derivativeStep, m_, potHO,
                            coordBounds, numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    } */
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "1p2d harmonic oscillator, no var. parameters (seconds): " << duration.count()
                        << '\n';
        }

        SUBCASE("One variational parameter") {
            PotHO potHO{mInit[0], omegaInit};
            auto const wavefHO{[](vmcp::Positions<2, 1> x, vmcp::VarParams<1> alpha) {
                return std::exp(-alpha[0].val * (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) / 2);
            }};
            std::array const laplHO{[](vmcp::Positions<2, 1> x, vmcp::VarParams<1> alpha) {
                return (alpha[0].val * (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) - 2) *
                       alpha[0].val *
                       std::exp(-alpha[0].val * (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) / 2);
            }};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                 i += vpIterationsFactor, m_[0] += mStep * vpIterationsFactor) {
                potHO.m = m_[0];
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                     j += vpIterationsFactor, omega_ += omegaStep * vpIterationsFactor) {
                    potHO.omega = omega_;

                    vmcp::VarParam const bestParam{m_[0].val * omega_ / vmcp::hbar};
                    vmcp::ParamBounds<1> const parBound{
                        NiceBound(bestParam, minParamFactor, maxParamFactor, maxParDiff)};
                    vmcp::Energy const expectedEn{vmcp::hbar * omega_};
                    std::string const logMessage{"mass: " + std::to_string(m_[0].val) +
                                                 ", ang. vel.: " + std::to_string(omega_)};

                    SUBCASE("Metropolis algorithm, analytical derivative") {
                        auto startOnePar = std::chrono::high_resolution_clock::now();
                        vmcp::VMCResult const vmcr = vmcp::VMCEnergy<2, 1, 1>(
                            wavefHO, parBound, laplHO, std::array{m_}, potHO, coordBounds, numEnergies,
                            vmcp::StatFuncType::regular, numSamples, rndGen);
                        auto stopOnePar = std::chrono::high_resolution_clock::now();
                        auto durationOnePar = duration_cast<std::chrono::seconds>(stopOnePar - startOnePar);
                        file_stream << "Mass " << m_[0].val << " and ang. vel. " << omega_
                                    << " (seconds): " << durationOnePar.count() << '\n';
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "1p2d harmonic oscillator, one var. parameter (seconds): " << duration.count()
                        << '\n';
        }
    }
}
