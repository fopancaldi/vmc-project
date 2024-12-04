/*
//!
//! @file test-trg-2p1d.cpp
//! @brief Tests for the triangular well system, with one particle in one dimension
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "test.hpp"
#include "vmcp.hpp"

#include <boost/math/special_functions/airy.hpp>
#include <chrono>
#include <fstream>
#include <numbers>
#include <string>
#include <tuple>

TEST_CASE("Testing the triangular potential well") {
    std::ofstream file_stream;
    file_stream.open(logFilePath, std::ios_base::app);
            // LF TODO: to be finished & adding references
            SUBCASE("1D triangular well potential") {
                vmcp::IntType const numberEnergies = 1 << 4;
                vmcp::CoordBounds<1> const coordBound = {vmcp::Bound{vmcp::Coordinate{-1},
       vmcp::Coordinate{1}}}; vmcp::RandomGenerator rndGen{seed}; vmcp::Mass const mInit{0.1f};
                vmcp::FPType const FInit = 0.1f;
                vmcp::Mass const mStep{0.05f};
                vmcp::FPType const FStep = 0.05f;
                vmcp::IntType const mIterations = iterations;
                vmcp::IntType const FIterations = iterations;
                constexpr vmcp::FPType V0 = 1e-9;

                struct PotTriangular {
                    vmcp::Mass m;
                    vmcp::FPType F;

                    vmcp::FPType operator()(vmcp::Positions<1, 1> x) const { return V0 - F * x[0][0].val;
       }
                };

                SUBCASE("No variational parameters, with Metropolis or importance sampling") {
                    struct WavefTriangular {
                        vmcp::Mass m;
                        vmcp::FPType F;

                        vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                            auto const alpha = std::pow(2 * m.val * F / (vmcp::hbar * vmcp::hbar), 1.0f
       / 3.0f); auto const xi = alpha * x[0][0].val; return boost::math::airy_ai(xi);
                        }
                    };

                    struct FirstDerTriangular {
                        vmcp::Mass m;
                        vmcp::FPType F;

                        vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                            auto const alpha = std::pow(2 * m.val * F / (vmcp::hbar * vmcp::hbar), 1.0
       / 3.0); auto const xi = alpha * x[0][0].val; return alpha * boost::math::airy_ai_prime(xi);
                        }
                    };

                    struct LaplTriangular {
                        vmcp::Mass m;
                        vmcp::FPType F;

                        vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                            auto const alpha = std::pow(2 * m.val * F / vmcp::hbar / vmcp::hbar, 1.0
       / 3.0); auto const xi = alpha * x[0][0].val; return std::pow(alpha, 2) * (xi *
       boost::math::airy_ai(xi));
                        }
                    };

                    WavefTriangular wavefTriangular{mInit.val, FInit};
                    vmcp::Gradients<1, 1, FirstDerTriangular> gradTriangular{
                        FirstDerTriangular{mInit.val, FInit}};
                    vmcp::Laplacians<1, LaplTriangular> laplTriangular{LaplTriangular{mInit, FInit}};
                    PotTriangular potTriangular{mInit.val, FInit};

                    auto start = std::chrono::high_resolution_clock::now();

                    for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ +=
       mStep) { potTriangular.m = m_; wavefTriangular.m = m_; gradTriangular[0][0].m = m_;
                        laplTriangular[0].m = m_;
                        for (auto [j, F_] = std::tuple{vmcp::IntType{0}, FInit}; j != FIterations;
                             ++j, F_ += FStep) {
                            potTriangular.F = F_;
                            wavefTriangular.F = F_;
                            gradTriangular[0][0].F = F_;
                            laplTriangular[0].F = F_;

                            vmcp::Energy const expectedEn{
                                boost::math::airy_ai_zero<vmcp::FPType>(1) *
                                std::pow(std::pow(vmcp::hbar * F_, 2) / (2 * m_.val), 1.0f / 3.0f)};
                            vmcp::VMCResult const vmcrMetr = vmcp::VMCEnergy<1, 1, 0>(
                                wavefTriangular, vmcp::ParamBounds<0>{}, laplTriangular, std::array{m_},
                                potTriangular, coordBound, numberEnergies, rndGen);

                            vmcp::VMCResult const vmcrImpSamp = vmcp::VMCEnergy<1, 1, 0>(
                                wavefTriangular, vmcp::ParamBounds<0>{}, gradTriangular, laplTriangular,
                                std::array{m_}, potTriangular, coordBound, numberEnergies, rndGen);

                            std::string logMessage{"mass: " + std::to_string(m_.val) +
                                                   ", slope: " + std::to_string(F_)};
                            CHECK_MESSAGE(abs(vmcrMetr.energy - expectedEn) < vmcEnergyTolerance,
       logMessage); CHECK_MESSAGE(abs(vmcrMetr.energy - expectedEn) < max(vmcrMetr.stdDev *
       allowedStdDevs, stdDevTolerance), logMessage); CHECK_MESSAGE(abs(vmcrImpSamp.energy - expectedEn) <
       vmcEnergyTolerance, logMessage); CHECK_MESSAGE(abs(vmcrImpSamp.energy - expectedEn) <
                                              max(vmcrImpSamp.stdDev * allowedStdDevs, stdDevTolerance),
                                          logMessage);
                        }
                    }

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
                    std::cout << "Triangular potential simulation (seconds): " << duration.count() <<
       '\n';
                }
        }*/
