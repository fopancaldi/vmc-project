//!
//! @file test-rad-1p1d.cpp
//! @brief Tests for the hydrogen atom radial problem, with one particle in cartesian coordinates
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
TEST_CASE("Testing the radial problem") {
    std::ofstream file_stream;
    file_stream.open(logFilePath, std::ios_base::app);

    SUBCASE("1D radial Schrodinger eq") {
        vmcp::CoordBounds<1> const coordBound = {vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Mass const mInit{1.f};
        vmcp::FPType const kInit{1.f};
        vmcp::Mass const mStep{0.1f};
        vmcp::FPType const kStep{0.1f};
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const kIterations = iterations;

        struct PotRad {
            vmcp::FPType k;
            vmcp::FPType operator()(vmcp::Positions<1, 1> r) const {
                vmcp::FPType pot = -k / std::abs(r[0][0].val);
                assert(!std::isnan(pot));
                return pot;
            }
        };

        SUBCASE("No variational parameters, with Metropolis or importance sampling") {
            struct WavefRad {
                vmcp::Mass m;
                vmcp::FPType k;
                vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
                    vmcp::FPType a0 = vmcp::hbar * vmcp::hbar / (k * m.val);
                    return std::abs(r[0][0].val) * std::exp(-std::abs(r[0][0].val) / a0);
                }
            };
            struct FirstDerRad {
                vmcp::Mass m;
                vmcp::FPType k;
                vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
                    vmcp::FPType a0 = vmcp::hbar * vmcp::hbar / (k * m.val);
                    return (std::exp(-std::abs(r[0][0].val) / a0) -
                            WavefRad{m, k}(r, vmcp::VarParams<0>{}) / a0) *
                           std::copysign(vmcp::FPType{1}, r[0][0].val);
                }
            };
            struct LaplRad {
                vmcp::Mass m;
                vmcp::FPType k;
                vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
                    vmcp::FPType a0 = vmcp::hbar * vmcp::hbar / (k * m.val);
                    return WavefRad{m, k}(r, vmcp::VarParams<0>{}) / (a0 * a0) -
                           (2 / a0) * std::exp(-std::abs(r[0][0].val) / a0);
                }
            };

            WavefRad wavefRad{mInit, kInit};
            vmcp::Gradients<1, 1, FirstDerRad> gradRad{FirstDerRad{mInit, kInit}};
            vmcp::Laplacians<1, LaplRad> laplRad{LaplRad{mInit, kInit}};
            PotRad potRad{kInit};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [j, k_] = std::tuple{vmcp::IntType{0}, kInit}; j != kIterations; ++j, k_ += kStep) {
                potRad.k = k_;
                wavefRad.k = k_;
                gradRad[0][0].k = k_;
                laplRad[0].k = k_;

                for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
                    wavefRad.m = m_;
                    gradRad[0][0].m = m_;
                    laplRad[0].m = m_;

                    vmcp::Energy const expectedEn{-k_ * k_ * m_.val / (2 * vmcp::hbar * vmcp::hbar)};
                    vmcp::Positions<1, 1> startPoss = FindPeak_<1, 1>(wavefRad, vmcp::VarParams<0>{}, potRad,
                                                                      coordBound, points_peakSearch, rndGen);

                    vmcp::VMCResult<0> const vmcrMetr = vmcp::VMCEnergy<1, 1, 0>(
                        wavefRad, startPoss, vmcp::ParamBounds<0>{}, laplRad, std::array{m_}, potRad,
                        coordBound, numEnergies, vmcp::StatFuncType::regular, bootstrapSamples, rndGen);
                    vmcp::VMCResult<0> const vmcrImpSamp = vmcp::VMCEnergy<1, 1, 0>(
                        wavefRad, startPoss, vmcp::ParamBounds<0>{}, gradRad, laplRad, std::array{m_}, potRad,
                        coordBound, numEnergies, vmcp::StatFuncType::regular, bootstrapSamples, rndGen);

                    std::string logMessage{"mass: " + std::to_string(m_.val) +
                                           "  Coulomb const: " + std::to_string(k_)};
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
            file_stream << "Radial problem, no var. parameters (seconds): " << duration.count() << '\n';
        }

        /* SUBCASE("One variational parameter") {
             auto const wavefRad{[](vmcp::Positions<1, 1> r, vmcp::VarParams<1> alpha) {
                 return std::abs(r[0][0].val) * std::exp(-std::abs(r[0][0].val) * alpha[0].val);
             }};

             std::array laplRad{[](vmcp::Positions<1, 1> r, vmcp::VarParams<1> alpha) {
                 return alpha[0].val *
                        (std::abs(r[0][0].val) * std::exp(-std::abs(r[0][0].val) * alpha[0].val) *
                             alpha[0].val -
                         (2 * std::exp(-std::abs(r[0][0].val) * alpha[0].val) /
         std::abs(r[0][0].val)));
             }};

                                  auto start = std::chrono::high_resolution_clock::now();


             for (auto [j, k_] = std::tuple{vmcp::IntType{0}, kInit}; j != kIterations;
                  j += vpIterationsFactor, k_ += kStep * vpIterationsFactor) {
                 PotRad potRad{kInit};

                 for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                      i += vpIterationsFactor, m_ += mStep * vpIterationsFactor) {
                     vmcp::VarParam bestParam{k_ * m_.val / std::pow(vmcp::hbar, 2)};
                     vmcp::ParamBounds<1> const parBound{
                         NiceBound(bestParam, vmcp::minParamFactor, vmcp::maxParamFactor, vmcp::maxParDiff)};
                     vmcp::Energy const expectedEn{-k_ * k_ * m_.val / (2 * vmcp::hbar * vmcp::hbar)};
vmcp::Positions<1, 1> const startPoss =
                        FindPeak_<1, 1>(wavefHO, vmcp::VarParams<1>{bestParam}, potHO, coordBounds,
                                        points_peakSearch, rndGen);

                     vmcp::VMCResult const vmcr = vmcp::VMCEnergy<1, 1, 1>(
                         wavefRad, startPoss, parBound, laplRad, std::array{m_}, potRad, coordBound,
numEnergies, vmcp::StatFuncType::regular, bootstrapSamples, rndGen); auto stopOnePar =
std::chrono::high_resolution_clock::now(); auto durationOnePar =
duration_cast<std::chrono::seconds>(stopOnePar - startOnePar); file_stream << "Radial problem, one
var.parameter, with mass " << m_.val
                                 << " (seconds): " << durationOnePar.count() << '\n';

                     std::string logMessage{"mass: " + std::to_string(m_.val) +
                                            "  Coulomb const: " + std::to_string(k_)};
                     CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                     CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                       max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                   logMessage);
                 }
             }
             auto stop = std::chrono::high_resolution_clock::now();
             auto duration = duration_cast<std::chrono::seconds>(stop - start);
             file_stream << "Radial problem, one var. parameter (seconds): " << duration.count() <<
         '\n';
         }*/
    }
}