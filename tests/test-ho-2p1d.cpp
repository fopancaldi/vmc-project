//!
//! @file test-ho-2p1d.cpp
//! @brief Tests for the harmonic oscillator system, with two particles in one dimension
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

    SUBCASE("Two particles in one dimension") {
        vmcp::CoordBounds<1> const coordBounds = {vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}}};
        vmcp::RandomGenerator rndGen{seed};
        std::array<vmcp::Mass, 2> const mInit{1.f, 5.f};
        std::array<vmcp::FPType, 2> const omegaInit{1.f, 5.f};
        std::array<vmcp::Mass, 2> const mInitVP{mInit[0], mInit[0]};
        std::array<vmcp::FPType, 2> const omegaInitVP{omegaInit[0], omegaInit[0]};
        vmcp::Mass const m1Step{0.1f};
        vmcp::Mass const m2Step{0.5f};
        vmcp::Mass const mStepVP = m1Step;
        vmcp::FPType const omega1Step = 0.1f;
        vmcp::FPType const omega2Step = 0.5f;
        vmcp::FPType const omegaStepVP = omega1Step;
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const omegaIterations = iterations;
        struct PotHO {
            std::array<vmcp::Mass, 2> m;
            std::array<vmcp::FPType, 2> omega;
            vmcp::FPType operator()(vmcp::Positions<1, 2> x) const {
                return x[0][0].val * x[0][0].val * (m[0].val * omega[0] * omega[0] / 2) +
                       x[1][0].val * x[1][0].val * (m[1].val * omega[1] * omega[1] / 2);
            }
        };
        vmcp::FPType const derivativeStep = coordBounds[0].Length().val / derivativeStepDenom;

        SUBCASE("No variational parameters") {
            struct WavefHO {
                std::array<vmcp::Mass, 2> m;
                std::array<vmcp::FPType, 2> omega;
                vmcp::FPType operator()(vmcp::Positions<1, 2> x, vmcp::VarParams<0>) const {
                    return std::exp(-(x[0][0].val * x[0][0].val * m[0].val * omega[0] +
                                      x[1][0].val * x[1][0].val * m[1].val * omega[1]) /
                                    (2 * vmcp::hbar));
                }
            };
            struct FirstDerHO {
                std::array<vmcp::Mass, 2> m;
                std::array<vmcp::FPType, 2> omega;
                vmcp::IntType particle;
                FirstDerHO(std::array<vmcp::Mass, 2> m_, std::array<vmcp::FPType, 2> omega_,
                           vmcp::IntType particle_)
                    : m{m_}, omega{omega_}, particle{particle_} {
                    assert(particle >= 0);
                    assert(particle <= 1);
                }
                vmcp::FPType operator()(vmcp::Positions<1, 2> x, vmcp::VarParams<0>) const {
                    vmcp::UIntType uPar = static_cast<vmcp::UIntType>(particle);
                    return -x[uPar][0].val * m[uPar].val * omega[uPar] / vmcp::hbar *
                           WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };
            struct LaplHO {
                std::array<vmcp::Mass, 2> m;
                std::array<vmcp::FPType, 2> omega;
                vmcp::IntType particle;
                LaplHO(std::array<vmcp::Mass, 2> m_, std::array<vmcp::FPType, 2> omega_,
                       vmcp::IntType particle_)
                    : m{m_}, omega{omega_}, particle{particle_} {
                    assert(particle >= 0);
                    assert(particle <= 1);
                }
                vmcp::FPType operator()(vmcp::Positions<1, 2> x, vmcp::VarParams<0>) const {
                    vmcp::UIntType uPar = static_cast<vmcp::UIntType>(particle);
                    return (std::pow(x[uPar][0].val * m[uPar].val * omega[uPar] / vmcp::hbar, 2) -
                            m[uPar].val * omega[uPar] / vmcp::hbar) *
                           WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                }
            };
            PotHO potHO{mInit, omegaInit};
            WavefHO wavefHO{mInit, omegaInit};
            vmcp::Gradients<1, 2, FirstDerHO> gradsHO{FirstDerHO{mInit, omegaInit, 0},
                                                      FirstDerHO{mInit, omegaInit, 1}};
            vmcp::Laplacians<2, LaplHO> laplsHO{LaplHO{mInit, omegaInit, 0}, LaplHO{mInit, omegaInit, 1}};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                 ++i, m_[0] += m1Step, m_[1] += m2Step) {
                potHO.m = m_;
                wavefHO.m = m_;
                gradsHO[0][0].m = m_;
                gradsHO[1][0].m = m_;
                laplsHO[0].m = m_;
                laplsHO[1].m = m_;
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                     ++j, omega_[0] += omega1Step, omega_[1] += omega2Step) {
                    potHO.omega = omega_;
                    wavefHO.omega = omega_;
                    gradsHO[0][0].omega = omega_;
                    gradsHO[1][0].omega = omega_;
                    laplsHO[0].omega = omega_;
                    laplsHO[1].omega = omega_;

                    vmcp::Energy const expectedEn{vmcp::hbar * (omega_[0] + omega_[1]) / 2};
                    std::string const genericLogMes =
                        "masses: " + std::to_string(m_[0].val) + ", " + std::to_string(m_[1].val) +
                        ", ang. vels.: " + std::to_string(omega_[0]) + ", " + std::to_string(omega_[1]);
                    vmcp::Positions<1, 2> startPoss = FindPeak_<1, 2>(wavefHO, vmcp::VarParams<0>{}, potHO,
                                                                      coordBounds, points_peakSearch, rndGen);
                    {
                        // Metropolis update, analytical derivative
                        std::string const logMes = metrLogMes + ", " + anDerLogMes + ", " + genericLogMes;
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 2, 0>(
                            wavefHO, startPoss, vmcp::ParamBounds<0>{}, laplsHO, m_, potHO, coordBounds, numEnergies,
                            vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMes);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMes);
                    }
                    {
                        // Metropolis update, numerical derivative
                        std::string const logMes = metrLogMes + ", " + numDerLogMes + ", " + genericLogMes;
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 2, 0>(
                            wavefHO, startPoss, vmcp::ParamBounds<0>{}, false, derivativeStep, m_, potHO, coordBounds,
                            numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMes);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMes);
                    }
                    /* {
                        // Importance sampling update, analytical derivative
                        std::string const logMes = metrLogMes + ", " + anDerLogMes + ", " + genericLogMes;
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 2, 0>(
                            wavefHO, startPoss, vmcp::ParamBounds<0>{}, gradsHO, laplsHO, m_, potHO, coordBounds,
                            numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMes);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMes);
                    } */
                    /* {
                        // Importance sampling update, numerical derivative
                        std::string const logMes = metrLogMes + ", " + anDerLogMes + ", " + genericLogMes;
                        vmcp::VMCResult<0> const vmcr = vmcp::VMCEnergy<1, 2, 0>(
                            wavefHO, startPoss, vmcp::ParamBounds<0>{}, true, derivativeStep, m_, potHO, coordBounds,
                            numEnergies, vmcp::StatFuncType::regular, numSamples, rndGen);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMes);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMes);
                    } */
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "2p1d harmonic oscillator, no var. parameters (seconds): " << duration.count()
                        << '\n';
        }

        SUBCASE("One variational parameter (using the same mass and ang. vel. for both particles)") {
            struct LaplHO {
                vmcp::IntType particle;
                LaplHO(vmcp::IntType particle_) : particle{particle_} {
                    assert(particle >= 0);
                    assert(particle <= 1);
                }
                vmcp::FPType operator()(vmcp::Positions<1, 2> x, vmcp::VarParams<1> alpha) const {
                    vmcp::UIntType uPar = static_cast<vmcp::UIntType>(particle);
                    return (std::pow(x[uPar][0].val * alpha[0].val, 2) - alpha[0].val) *
                           std::exp(-alpha[0].val * (x[0][0].val * x[0][0].val + x[1][0].val * x[1][0].val) /
                                    2);
                }
            };
            PotHO potHO{mInitVP, omegaInitVP};
            auto const wavefHO{[](vmcp::Positions<1, 2> x, vmcp::VarParams<1> alpha) {
                return std::exp(-alpha[0].val * (x[0][0].val * x[0][0].val + x[1][0].val * x[1][0].val) / 2);
            }};
            vmcp::Laplacians<2, LaplHO> const laplsHO{LaplHO{0}, LaplHO{1}};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInitVP}; i != mIterations;
                 i += vpIterationsFactor * 2, m_[0] += mStepVP * vpIterationsFactor * 2,
                          m_[1] += mStepVP * vpIterationsFactor * 2) {
                potHO.m = m_;
                for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInitVP}; j != omegaIterations;
                     j += vpIterationsFactor * 2, omega_[0] += omegaStepVP * vpIterationsFactor * 2,
                              omega_[1] += omegaStepVP * vpIterationsFactor * 2) {
                    potHO.omega = omega_;

                    vmcp::VarParam bestParam{m_[0].val * omega_[0] / vmcp::hbar};
                    vmcp::ParamBounds<1> const parBound{
                        NiceBound(bestParam, minParamFactor, maxParamFactor, maxParDiff)};
                    vmcp::Energy const expectedEn{vmcp::hbar * omega_[0]};
                    std::string const genericLogMes =
                        "mass: " + std::to_string(m_[0].val) + ", ang. vel.: " + std::to_string(omega_[0]);
                    vmcp::Positions<1, 2> const startPoss =
                        FindPeak_<1, 2>(wavefHO, vmcp::VarParams<1>{bestParam}, potHO, coordBounds,
                                        points_peakSearch, rndGen);
                  
                    {
                        // Metropolis update, analytical derivative
                        std::string const logMes = metrLogMes + ", " + anDerLogMes + ", " + genericLogMes;
                        auto startOnePar = std::chrono::high_resolution_clock::now();
                        vmcp::VMCResult<1> const vmcr =
                            vmcp::VMCEnergy<1, 2, 1>(wavefHO, startPoss, parBound, laplsHO, std::array{m_},
                                                     potHO, coordBounds, numEnergies / vpNumEnergiesFactor,
                                                     vmcp::StatFuncType::regular, bootstrapSamples, rndGen);
                        auto stopOnePar = std::chrono::high_resolution_clock::now();
                        auto durationOnePar = duration_cast<std::chrono::seconds>(stopOnePar - startOnePar);
                        file_stream << logMes << " (seconds): " << durationOnePar.count() << '\n';
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMes);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMes);
                    }
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "2p1d harmonic oscillator, one var. parameter (seconds): " << duration.count()
                        << '\n';
        }
    }
}
