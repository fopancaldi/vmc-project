//!
//! @file test-ho-1p1d.cpp
//! @brief Tests for the harmonic oscillator system, with one particle in one dimension
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

TEST_CASE("Testing the harmonic oscillator") {
    std::ofstream file_stream;
    file_stream.open(logFilePath, std::ios_base::app);

    SUBCASE("One particle in one dimension") {
        vmcp::CoordBounds<1> const coordBound = {vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Mass const mInit{1.f};
        vmcp::FPType const omegaInit = 1.f;
        vmcp::Mass const mStep{0.1f};
        vmcp::FPType const omegaStep = 0.1f;
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const omegaIterations = iterations;
        struct PotHO {
            vmcp::Mass m;
            vmcp::FPType omega;
            vmcp::FPType operator()(vmcp::Positions<1, 1> x) const {
                return x[0][0].val * x[0][0].val * (m.val * omega * omega / 2);
            }
        };

        /*SUBCASE("No variational parameters, with Metropolis or importance sampling") {
            struct WavefHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    return std::exp(-x[0][0].val * x[0][0].val * (m.val * omega / (2 * vmcp::hbar)));
                }
            };
            struct FirstDerHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    return (-x[0][0].val * (m.val * omega / (vmcp::hbar))) *
                           WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };
            struct LaplHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    return (std::pow(x[0][0].val * m.val * omega / vmcp::hbar, 2) -
                            (m.val * omega / vmcp::hbar)) *
                           WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };

            WavefHO wavefHO{mInit.val, omegaInit};
            vmcp::Gradients<1, 1, FirstDerHO> gradHO{FirstDerHO{mInit.val, omegaInit}};
            vmcp::Laplacians<1, LaplHO> laplHO{LaplHO{mInit, omegaInit}};
            PotHO potHO{mInit.val, omegaInit};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
                potHO.m = m_;
                wavefHO.m = m_;
                gradHO[0][0].m = m_;
                laplHO[0].m = m_;
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                     ++j, omega_ += omegaStep) {
                    wavefHO.omega = omega_;
                    potHO.omega = omega_;
                    laplHO[0].omega = omega_;

                    vmcp::Energy const expectedEn{vmcp::hbar * omega_ / 2};
                    vmcp::VMCResult<0> const vmcrMetr =
                        vmcp::VMCEnergy<1, 1, 0>(wavefHO, vmcp::ParamBounds<0>{}, laplHO, std::array{m_},
                                                 potHO, coordBound, numEnergies, rndGen);
                    vmcp::VMCResult<0> const vmcrImpSamp =
                        vmcp::VMCEnergy<1, 1, 0>(wavefHO, vmcp::ParamBounds<0>{}, gradHO, laplHO,
                                                 std::array{m_}, potHO, coordBound, numEnergies, rndGen);

                    std::string logMessage{"mass: " + std::to_string(m_.val) +
                                           ", ang. vel.: " + std::to_string(omega_)};
                    CHECK_MESSAGE(abs(vmcrMetr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                    CHECK_MESSAGE(abs(vmcrMetr.energy - expectedEn) <
                                      max(vmcrMetr.stdDev * allowedStdDevs, stdDevTolerance),
                                  logMessage);
                    CHECK_MESSAGE(abs(vmcrImpSamp.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                    CHECK_MESSAGE(abs(vmcrImpSamp.energy - expectedEn) <
                                      max(vmcrImpSamp.stdDev * allowedStdDevs, stdDevTolerance),
                                  logMessage);
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "Harmonic oscillator, no var. parameters (seconds): " << duration.count() << '\n';
        }*/

        SUBCASE("One variational parameter") {
            PotHO potHO{mInit, omegaInit};
            auto const wavefHO{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
                return std::exp(-alpha[0].val * x[0][0].val * x[0][0].val / 2);
            }};
            std::array laplHO{[](vmcp::Positions<1, 1> x, vmcp::VarParams<1> alpha) {
                return (std::pow(x[0][0].val * alpha[0].val, 2) - alpha[0].val) *
                       std::exp(-alpha[0].val * x[0][0].val * x[0][0].val / 2);
            }};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                 i += vpIterationsFactor, m_ += mStep * vpIterationsFactor) {
                potHO.m = m_;
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                     j += vpIterationsFactor, omega_ += omegaStep * vpIterationsFactor) {
                    potHO.omega = omega_;

                    vmcp::VarParam bestParam{m_.val * omega_ / (2 * vmcp::hbar)};
                    vmcp::ParamBounds<1> const parBound{
                        NiceBound(bestParam, minParamFactor, maxParamFactor, maxParDiff)};
                    vmcp::Energy const expectedEn{vmcp::hbar * omega_ / 2};

                    auto startOnePar = std::chrono::high_resolution_clock::now();
                    vmcp::VMCResult<1> const vmcr =
                        vmcp::VMCEnergy<1, 1, 1>(wavefHO, parBound, laplHO, std::array{m_}, potHO, coordBound,
                                                 numEnergies / vpNumEnergiesFactor, rndGen);
                    auto stopOnePar = std::chrono::high_resolution_clock::now();
                    auto durationOnePar = duration_cast<std::chrono::seconds>(stopOnePar - startOnePar);
                    file_stream << "Harmonic oscillator, one var.parameter, with mass " << m_.val
                                << " and ang. vel. " << omega_ << " (seconds): " << durationOnePar.count()
                                << '\n';

                    std::string logMessage{"mass: " + std::to_string(m_.val) +
                                           ", ang. vel.: " + std::to_string(omega_)};
                    CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                    CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                      max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                  logMessage);

                    std::cout << "m: " << m_.val << "   omega: " << omega_ << '\n';
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "Harmonic oscillator, one var. parameter (seconds): " << duration.count() << '\n';
        }
    }
}
