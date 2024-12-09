//!
//! @file test-box.cpp
//! @brief Tests for the potential box system
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

TEST_CASE("Testing the potential box") {
    std::ofstream file_stream;
    file_stream.open(logFilePath, std::ios_base::app);

    SUBCASE("One particle in one dimension") {
        // l = length of the box
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Masses<1> const mInit = {vmcp::Mass{1.f}};
        vmcp::FPType const lInit = 1;
        vmcp::FPType const mStep = 0.2f;
        vmcp::FPType const lStep = 0.2f;
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const lIterations = iterations;
        auto potBox{[](vmcp::Positions<1, 1>) -> vmcp::FPType { return 0; }};

        SUBCASE("No variational parameters") {
            struct WavefBox {
                vmcp::FPType l;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    if (std::abs(x[0][0].val) <= l / 2) {
                        return std::cos(std::numbers::pi_v<vmcp::FPType> * x[0][0].val / l);
                    } else {
                        return 0;
                    }
                }
            };
            struct FirstDerBox {
                vmcp::FPType l;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    if (std::abs(x[0][0].val) <= l / 2) {
                        return -(std::numbers::pi_v<vmcp::FPType> / l) *
                               std::sin(std::numbers::pi_v<vmcp::FPType> * x[0][0].val / l);
                    } else {
                        return 0;
                    }
                }
            };
            struct LaplBox {
                vmcp::FPType l;
                vmcp::FPType operator()(vmcp::Positions<1, 1> x, vmcp::VarParams<0>) const {
                    if (std::abs(x[0][0].val) <= l / 2) {
                        return -std::pow(std::numbers::pi_v<vmcp::FPType> / l, 2) *
                               std::cos(std::numbers::pi_v<vmcp::FPType> * x[0][0].val / l);
                    } else {
                        return 0;
                    }
                }
            };
            WavefBox wavefBox{lInit};
            vmcp::Gradients<1, 1, FirstDerBox> gradBox{FirstDerBox{lInit}};
            vmcp::Laplacians<1, LaplBox> laplBox{lInit};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                 ++i, m_[0].val += mStep) {
                for (auto [j, l_] = std::tuple{vmcp::IntType{0}, lInit}; j != lIterations; ++j, l_ += lStep) {
                    wavefBox.l = l_;
                    gradBox[0][0].l = l_;
                    laplBox[0].l = l_;

                    vmcp::CoordBounds<1> const coordBound{
                        vmcp::Bound{vmcp::Coordinate{-l_ / 2}, vmcp::Coordinate{l_ / 2}}};
                    vmcp::Energy const expectedEn{
                        1 / (2 * m_[0].val) *
                        std::pow(vmcp::hbar * std::numbers::pi_v<vmcp::FPType> / l_, 2)};
                    std::string logMessage{"mass: " + std::to_string(m_[0].val) +
                                           ", length: " + std::to_string(l_)};
                    vmcp::FPType const derivativeStep = coordBound[0].Length().val / derivativeStepDenom;

                    SUBCASE("Metropolis algorithm, analytical derivative") {
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 1, 0>(
                            wavefBox, vmcp::ParamBounds<0>{}, laplBox, std::array{m_}, potBox, coordBound,
                            numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                    SUBCASE("Metropolis algorithm, numerical derivative") {
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 1, 0>(
                            wavefBox, vmcp::ParamBounds<0>{}, false, derivativeStep, std::array{m_}, potBox,
                            coordBound, numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                    SUBCASE("Importance sampling algorithm, analytical derivative") {
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 1, 0>(
                            wavefBox, vmcp::ParamBounds<0>{}, gradBox, laplBox, std::array{m_}, potBox,
                            coordBound, numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                    /* SUBCASE("Importance sampling algorithm, numerical derivative") {
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 1, 0>(
                            wavefBox, vmcp::ParamBounds<0>{}, true, derivativeStep, std::array{m_}, potBox,
                            coordBound, numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    } */
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "1p1d particle in a box, no var. parameters (seconds): " << duration.count()
                        << '\n';
        }
    }
}
