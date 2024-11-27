#inlcude "tests.hpp"

SUBCASE("1D radial Schroedinger eq") {
        vmcp::IntType const numberEnergies = 1 << 4;
        vmcp::CoordBounds<1> const coordBound = {vmcp::Bound{vmcp::Coordinate{0.1}, vmcp::Coordinate{1}}};
        vmcp::RandomGenerator rndGen{seed};
        vmcp::Mass const mInit{1.f};
        vmcp::FPType const zInit{1.f};
        vmcp::Mass const mStep{0.1f};
        vmcp::FPType const zStep{1.f};
        vmcp::IntType const mIterations = iterations;
        vmcp::IntType const zIterations = iterations / 8;
        constexpr vmcp::FPType e = 1.f;
        struct PotRad {
            vmcp::FPType z;
            vmcp::FPType operator()(vmcp::Positions<1, 1> r) const { return -z / r[0][0].val; }
        };

        SUBCASE("No variational parameters, with Metropolis or importance sampling") {
            struct WavefRad {
                vmcp::Mass m;
                vmcp::FPType z;
                vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
                    vmcp::FPType a0 = std::pow(vmcp::hbar, 2) / (m.val * e * e);
                    return std::exp(-z * r[0][0].val / a0);
                }
            };
            struct FirstDerRad {
                vmcp::Mass m;
                vmcp::FPType z;
                vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
                    vmcp::FPType a0 = std::pow(vmcp::hbar, 2) / (m.val * e * e);
                    return -(z / a0) * WavefRad{m, e}(r, vmcp::VarParams<0>{});
                }
            };
            struct LaplRad {
                vmcp::Mass m;
                vmcp::FPType z;
                vmcp::FPType operator()(vmcp::Positions<1, 1> r, vmcp::VarParams<0>) const {
                    vmcp::FPType a0 = std::pow(vmcp::hbar, 2) / (m.val * e * e);
                    return std::pow((z / a0), 2) * WavefRad{m, e}(r, vmcp::VarParams<0>{});
                }
            };

            WavefRad wavefRad{mInit, zInit};
            vmcp::Gradients<1, 1, FirstDerRad> gradRad{FirstDerRad{mInit, zInit}};
            vmcp::Laplacians<1, LaplRad> laplRad{LaplRad{mInit, zInit}};
            PotRad potRad{zInit};

            auto start = std::chrono::high_resolution_clock::now();

            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations; ++i, m_ += mStep) {
                wavefRad.m = m_;
                gradRad[0][0].m = m_;
                laplRad[0].m = m_;
                for (auto [j, z_] = std::tuple{vmcp::IntType{0}, zInit}; j != zIterations; ++j, z_ += zStep) {
                    wavefRad.z = z_;
                    potRad.z = z_;
                    laplRad[0].z = z_;

                    vmcp::Energy const expectedEn{-z_ * z_ * std::pow(e, 4) * m_.val /
                                                  (2 * vmcp::hbar * vmcp::hbar)};
                    vmcp::VMCResult const vmcrMetr =
                        vmcp::VMCEnergy<1, 1, 0>(wavefRad, vmcp::ParamBounds<0>{}, laplRad, std::array{m_},
                                                 potRad, coordBound, numberEnergies, rndGen);
                    vmcp::VMCResult const vmcrImpSamp =
                        vmcp::VMCEnergy<1, 1, 0>(wavefRad, vmcp::ParamBounds<0>{}, gradRad, laplRad,
                                                 std::array{m_}, potRad, coordBound, numberEnergies, rndGen);

                    std::string logMessage{"mass: " + std::to_string(m_.val) +
                                           ", atomic number: " + std::to_string(z_)};
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

        SUBCASE("One variational parameter") {
            PotRad potRad{zInit};
            auto const wavefRad{[](vmcp::Positions<1, 1> r, vmcp::VarParams<1> alpha) {
                return std::exp(-alpha[0].val * r[0][0].val);
            }};
            std::array laplRad{[](vmcp::Positions<1, 1> r, vmcp::VarParams<1> alpha) {
                return -std::pow(alpha[0].val, 2) * std::exp(-alpha[0].val * r[0][0].val);
            }};

            auto start = std::chrono::high_resolution_clock::now();
            for (auto [i, m_] = std::tuple{vmcp::IntType{0}, mInit}; i != mIterations;
                 i += varParamsFactor, m_ += mStep * varParamsFactor) {
                for (auto [j, z_] = std::tuple{vmcp::IntType{0}, zInit}; j != zIterations;
                     j += varParamsFactor, z_ += zStep * varParamsFactor) {
                    potRad.z = z_;

                    vmcp::VarParam bestParam{m_.val * e * e / std::pow(vmcp::hbar, 2)};
                    vmcp::ParamBounds<1> const parBound{
                        NiceBound(bestParam, minParamFactor, maxParamFactor, maxParDiff)};
                    vmcp::Energy const expectedEn{-z_ * z_ * std::pow(e, 4) * m_.val /
                                                  (2 * vmcp::hbar * vmcp::hbar)};

                    auto startOnePar = std::chrono::high_resolution_clock::now();
                    vmcp::VMCResult const vmcr =
                        vmcp::VMCEnergy<1, 1, 1>(wavefRad, parBound, laplRad, std::array{m_}, potRad,
                                                 coordBound, numberEnergies, rndGen);
                    auto stopOnePar = std::chrono::high_resolution_clock::now();
                    auto durationOnePar = duration_cast<std::chrono::seconds>(stopOnePar - startOnePar);
                    file_stream << "Radial problem, one var.parameter, with mass" << m_.val
                                << " atomic number " << z_ << " (seconds): " << durationOnePar.count()
                                << '\n';

                    std::string logMessage{"mass: " + std::to_string(m_.val) +
                                           ", atomic number: " + std::to_string(z_)};
                    CHECK_MESSAGE(abs(vmcr.energy - expectedEn) < vmcEnergyTolerance, logMessage);
                    CHECK_MESSAGE(abs(vmcr.energy - expectedEn) <
                                      max(vmcr.stdDev * allowedStdDevs, stdDevTolerance),
                                  logMessage);
                }
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::seconds>(stop - start);
            file_stream << "Radial problem, one var. parameter (seconds): " << duration.count() << '\n';
        }
    }
