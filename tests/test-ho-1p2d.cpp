#include "tests.hpp"

SUBCASE("2D harmonic oscillator, one particle") {
            vmcp::IntType const numberEnergies = 1 << 4;
            vmcp::CoordBounds<2> const coordBounds{vmcp::Bound{vmcp::Coordinate{-100}, vmcp::Coordinate{100}},
                                                   vmcp::Bound{vmcp::Coordinate{-100},
       vmcp::Coordinate{100}}}; vmcp::RandomGenerator rndGen{seed}; vmcp::Mass const mInit{1.f}; vmcp::FPType
       const omegaInit = 1.f; vmcp::Mass const mStep{0.2f}; vmcp::FPType const omegaStep = 0.2f; vmcp::IntType
       const mIterations = iterations; vmcp::IntType const omegaIterations = iterations; struct PotHO {
                vmcp::Mass m;
                vmcp::FPType omega;
                vmcp::FPType operator()(vmcp::Positions<2, 1> x) const {
                    return (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) * (m.val * omega * omega /
       2);
                }
            };

            SUBCASE("No variational parameters, with Metropolis or importance sampling") {
                struct WavefHO {
                    vmcp::Mass m;
                    vmcp::FPType omega;
                    vmcp::FPType operator()(vmcp::Positions<2, 1> x, vmcp::VarParams<0>) const {
                        return std::exp(-(std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) *
                                        (m.val * omega / (2 * vmcp::hbar)));
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
                        return (-x[0][uDim].val * (m.val * omega / (vmcp::hbar)) *
                                (WavefHO{m, omega}(x, vmcp::VarParams<0>{})));
                    }
                };
                struct LaplHO {
                    vmcp::Mass m;
                    vmcp::FPType omega;
                    vmcp::FPType operator()(vmcp::Positions<2, 1> x, vmcp::VarParams<0>) const {
                        return ((m.val * omega / vmcp::hbar *
                                     (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) -
                                 2)) *
                               (m.val * omega / vmcp::hbar) * WavefHO{m, omega}(x, vmcp::VarParams<0>{});
                    }
                };

                WavefHO wavefHO{mInit.val, omegaInit};
                vmcp::Gradients<2, 1, FirstDerHO> gradHO{FirstDerHO{mInit, omegaInit, 0},
                                                         FirstDerHO{mInit, omegaInit, 1}};
                vmcp::Laplacians<1, LaplHO> laplHO{LaplHO{mInit, omegaInit}};
                PotHO potHO{mInit.val, omegaInit};

                auto start = std::chrono::high_resolution_clock::now();

                for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                     ++i, m_.val += mStep.val) {
                    potHO.m = m_;
                    wavefHO.m = m_;
                    gradHO[0][0].m = m_;
                    gradHO[0][1].m = m_;
                    laplHO[0].m = m_;
                    for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                         ++j, omega_ += omegaStep) {
                        wavefHO.omega = omega_;
                        potHO.omega = omega_;
                        laplHO[0].omega = omega_;

                        vmcp::Energy const expectedEn{vmcp::hbar * omega_};
                        vmcp::VMCResult const vmcrMetr =
                            vmcp::VMCEnergy<2, 1, 0>(wavefHO, vmcp::ParamBounds<0>{}, laplHO, std::array{m_},
                                                     potHO, coordBounds, numberEnergies, rndGen);
                        vmcp::VMCResult const vmcrImpSamp =
                            vmcp::VMCEnergy<2, 1, 0>(wavefHO, vmcp::ParamBounds<0>{}, gradHO, laplHO,
                                                     std::array{m_}, potHO, coordBounds, numberEnergies,
       rndGen);

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
                file_stream << "2D Harmonic oscillator, no var. parameters (seconds): " << duration.count()
                            << '\n';
            }

            SUBCASE("One variational parameter") {
                PotHO potHO{mInit, omegaInit};

                auto const wavefHO{[](vmcp::Positions<2, 1> x, vmcp::VarParams<1> alpha) {
                    return std::exp(-alpha[0].val * (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)));
                }};
                std::array laplHO{[](vmcp::Positions<2, 1> x, vmcp::VarParams<1> alpha) {
                    return (alpha[0].val * (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)) - 1) * 4 *
                           alpha[0].val *
                           std::exp(-alpha[0].val * (std::pow(x[0][0].val, 2) + std::pow(x[0][1].val, 2)));
                }};

                auto start = std::chrono::high_resolution_clock::now();

                for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                     i += varParamsFactor, m_ += mStep * varParamsFactor) {
                    potHO.m = m_;
                    for (auto [j, omega_] = std::tuple{vmcp::IntType{0}, omegaInit}; j != omegaIterations;
                         j += varParamsFactor, omega_ += omegaStep * varParamsFactor) {
                        potHO.omega = omega_;

                        vmcp::VarParam bestParam{m_.val * omega_ / vmcp::hbar};
                        vmcp::ParamBounds<1> const parBound{
                            NiceBound(bestParam, minParamFactor, maxParamFactor, maxParDiff)};
                        vmcp::Energy const expectedEn{vmcp::hbar * omega_};

                        auto startOnePar = std::chrono::high_resolution_clock::now();
                        vmcp::VMCResult const vmcr =
                            vmcp::VMCEnergy<2, 1, 1>(wavefHO, parBound, laplHO, std::array{m_}, potHO,
                                                     coordBounds, numberEnergies, rndGen);
                        auto stopOnePar = std::chrono::high_resolution_clock::now();
                        auto durationOnePar = duration_cast<std::chrono::seconds>(stopOnePar - startOnePar);
                        file_stream << "2D Harmonic oscillator, one var.parameter, with mass " << m_.val
                                    << " and ang. vel. " << omega_ << " (seconds): " << durationOnePar.count()
                                    << '\n';

                        std::string logMessage{"mass: " + std::to_string(m_.val) +
                                               ", ang. vel.: " + std::to_string(omega_)};
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                        CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                          max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                      logMessage);
                    }
                }

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = duration_cast<std::chrono::seconds>(stop - start);
                file_stream << "2D Harmonic oscillator, one var. parameter (seconds): " << duration.count()
                            << '\n';
            }
        }
