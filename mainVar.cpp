//!
//! @file main.cpp
//! @brief main file
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the helper functions used inside ComputeEnergies in main.cpp
//!

#include "main.hpp"

// Constant Parameters
constexpr vmcp::FPType Beta = vmcp::FPType{2.82843f};
constexpr vmcp::FPType Gamma = vmcp::FPType{2.82843f};
constexpr vmcp::FPType OmegaHO{1};
constexpr vmcp::Mass M{1};
const vmcp::FPType ADistance = 0.0043 * std::sqrt(vmcp::hbar / (M.val * OmegaHO));
const vmcp::FPType latticeSpacing = ADistance * std::sqrt(vmcp::FPType{2});

constexpr vmcp::FPType derivativeStep{0.01f};
constexpr vmcp::IntType numEnergies = 1 << 9;
constexpr vmcp::IntType bootstrapSamples = 1000;
constexpr vmcp::FPType confLvl{95};

vmcp::RandomGenerator gen{(std::random_device())()};

// Non Interacting Harmonic Oscillator
template <vmcp::Dimension D, vmcp::ParticNum N>
void HONoInt(vmcp::StatFuncType statFunction) {
    using namespace vmcp;

    CoordBounds<D> const coordBounds = MakeCoordBounds<D>(Coordinate{-10}, Coordinate{10});

    struct PotHO {
        Masses<N> m;
        FPType omegaHO;
        FPType gamma;
        FPType operator()(Positions<D, N> x) const {
            Coordinate *begin = &x[0][0];
            Coordinate *end = &x[0][0] + N * D;
            // TODO:val -> coord
            FPType result =
                std::transform_reduce(begin, end, FPType{0.f}, std::plus<>(),
                                      [D_ = D, gamma_ = gamma, &begin](Coordinate &c) {
                                          // Compute the index of the current element
                                          auto const index = &c - begin;
                                          // Check if it is in the last dimension
                                          bool isLastDimension =
                                              ((static_cast<UIntType>(index) + 1u) % D_ == 0) && (D_ != 1);
                                          return c.val * c.val * (isLastDimension ? gamma_ * gamma_ : 1);
                                      }) *
                m[0].val * omegaHO * omegaHO / 2;
            assert(!std::isnan(result));

            return result;
        }
    };
    struct WavefHO {
        FPType alpha;
        FPType beta;
        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            //   Harmonic oscillator term
            Coordinate *begin = &x[0][0];
            Coordinate *end = &x[0][0] + N * D;

            FPType expArg = std::transform_reduce(
                begin, end, FPType{0}, std::plus<>(), [beta_ = beta, &begin](Coordinate &c) {
                    auto const index = &c - begin;
                    bool isLastDimension = ((static_cast<UIntType>(index) + 1u) % D == 0) && (D != 1);
                    return c.val * c.val * (isLastDimension ? beta_ : 1);
                });

            return std::exp(-alpha * expArg);
        }
    };
    struct FirstDerHO {
        FPType alpha;
        FPType beta;
        UIntType dimension;
        UIntType particle;
        FirstDerHO(FPType alpha_, FPType beta_, UIntType dimension_, UIntType particle_)
            : alpha{alpha_}, beta{beta_}, dimension{dimension_}, particle{particle_} {
            assert(dimension < D);
            assert(particle < N);
        }
        FirstDerHO() : alpha{}, beta{}, dimension{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            // Harmonic oscillator first derivative term
            FPType result = -2 * alpha * x[particle][dimension].val *
                            (((dimension == (D - 1)) && (D != 1)) ? beta : 1) *
                            WavefHO{alpha, beta}(x, VarParams<0>{});
            assert(!std::isnan(result));

            return result;
        }
    };
    struct LaplHO {
        FPType alpha;
        FPType beta;
        UIntType particle;

        LaplHO(FPType alpha_, FPType beta_, UIntType particle_)
            : alpha{alpha_}, beta{beta_}, particle{particle_} {
            assert(particle < N);
        }

        LaplHO() : alpha{}, beta{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            UIntType uPar = particle;
            Coordinate *begin = &x[uPar][0];
            Coordinate *end = &x[uPar][0] + D;

            FPType result =
                (std::pow(2 * alpha, 2) *
                     std::transform_reduce(begin, end, FPType{0.f}, std::plus<>(),
                                           [D_ = D, beta_ = beta, &begin](Coordinate &c) {
                                               // Compute the index of the current element
                                               auto const index = &c - begin;
                                               // Check if it is in the last dimension
                                               bool isLastDimension =
                                                   ((static_cast<UIntType>(index) + 1u) % D_ == 0) &&
                                                   (D_ != 1);
                                               return c.val * c.val * (isLastDimension ? beta_ * beta_ : 1);
                                           }) -
                 2 * alpha * ((D == 1) ? 1 : (D - 1 + beta))) *
                WavefHO{alpha, beta}(x, VarParams<0>{});
            assert(!std::isnan(result));

            return result;
        }
    };

    Masses<N> mass;
    mass.fill(M);

    std::vector<FPType> alphaVals;
    std::generate_n(std::back_inserter(alphaVals), 20, [i = 0]() mutable -> FPType {
        return FPType{0.25f} + static_cast<FPType>(++i) * FPType{0.025f};
    });

    // Exact energies vector (all the elements equal the exact energy value for the ground state)
    Energy const exactEn{((D == 1) ? 1 : (D - 1 + Gamma)) * N * hbar * OmegaHO / 2};
    std::vector<FPType> exactEns;
    std::generate_n(std::back_inserter(exactEns), 100, [exactEn]() -> FPType { return exactEn.val; });

    // Analytic vectors
    std::vector<FPType> energyValsMetrAn;
    std::vector<FPType> energyValsImpSampAn;
    std::vector<ConfInterval> confIntsMetrAn;
    std::vector<ConfInterval> confIntsImpSampAn;

    // Numeric vectors
    std::vector<FPType> energyValsMetrNum;
    std::vector<FPType> energyValsImpSampNum;
    std::vector<ConfInterval> confIntsMetrNum;
    std::vector<ConfInterval> confIntsImpSampNum;

    // Perform Monte Carlo Methods
    for (FPType alphaVal : alphaVals) {
        PotHO potHO{mass, OmegaHO, Gamma};
        WavefHO wavefHO{alphaVal, Beta};
        Gradients<D, N, FirstDerHO> gradsHO;
        for (ParticNum n = 0u; n < N; n++) {
            for (Dimension d = 0u; d < D; d++) {
                gradsHO[n][d] = FirstDerHO{alphaVal, Beta, d, n};
            }
        }

        std::array<LaplHO, N> laplHO;
        std::generate(laplHO.begin(), laplHO.end(), [counter = ParticNum{0}, alphaVal]() mutable {
            return LaplHO{alphaVal, Beta, counter++};
        });

        Positions<D, N> startPoss = BuildFCCStartPoint_<D, N>(coordBounds, latticeSpacing);

        // Analaytic
        VMCResult<0> const vmcrMetrAn =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, laplHO, mass, potHO, coordBounds,
                               numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntMetrAn = GetConfInt(vmcrMetrAn.energy, vmcrMetrAn.stdDev, confLvl);
        std::cout << "Metropolis:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrAn.energy << " +/- " << vmcrMetrAn.stdDev << "\tconf. int. with conf. lvl. of "
                  << confLvl << "%: " << confIntMetrAn.min.val << " - " << confIntMetrAn.max << '\n';

        VMCResult<0> const vmcrImpSampAn =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, gradsHO, laplHO, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntImpSampAn = GetConfInt(vmcrImpSampAn.energy, vmcrImpSampAn.stdDev, confLvl);
        std::cout << "ImpSamp:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrImpSampAn.energy << " +/- " << vmcrImpSampAn.stdDev
                  << "\tconf. int. with conf. lvl. of " << confLvl << "%: " << confIntImpSampAn.min << " - "
                  << confIntImpSampAn.max << '\n';

        energyValsMetrAn.push_back(vmcrMetrAn.energy.val);
        energyValsImpSampAn.push_back(vmcrImpSampAn.energy.val);
        confIntsMetrAn.push_back(confIntMetrAn);
        confIntsImpSampAn.push_back(confIntImpSampAn);

        // Numeric
        VMCResult<0> const vmcrMetrNum =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, false, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntMetrNum = GetConfInt(vmcrMetrNum.energy, vmcrMetrNum.stdDev, confLvl);

        VMCResult<0> const vmcrImpSampNum =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, true, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntImpSampNum = GetConfInt(vmcrImpSampNum.energy, vmcrImpSampNum.stdDev, confLvl);

        energyValsMetrNum.push_back(vmcrMetrNum.energy.val);
        energyValsImpSampNum.push_back(vmcrImpSampNum.energy.val);
        confIntsMetrNum.push_back(confIntMetrNum);
        confIntsImpSampNum.push_back(confIntImpSampNum);
    }

    // Plotting the energies:
    // Construct file paths
    std::string folder = "./artifacts/NoInt/D=" + std::to_string(D) + "_N=" + std::to_string(N);
    std::string metrAnFile = folder + "/plot_Metr_An.pdf";
    std::string impSampAnFile = folder + "/plot_ImpSamp_An.pdf";
    std::string metrNumFile = folder + "/plot_Metr_Num.pdf";
    std::string impSampNumFile = folder + "/plot_ImpSamp_Num.pdf";

    std::filesystem::create_directories(folder);

    // Analaytic
    sciplot::Canvas canvasMetrAn =
        MakeGraphNoInt<D, N>(alphaVals, energyValsMetrAn, errorValsMetrAn, confIntsMetrAn, exactEns);
    sciplot::Canvas canvasImpSampAn =
        MakeGraphNoInt<D, N>(alphaVals, energyValsImpSampAn, errorValsImpSampAn, confIntsImpSampAn, exactEns);

    canvasMetrAn.save(metrAnFile);
    canvasImpSampAn.save(impSampAnFile);

    // Numeric
    sciplot::Canvas canvasMetrNum =
        MakeGraphNoInt<D, N>(alphaVals, energyValsMetrNum, errorValsMetrNum, confIntsMetrNum, exactEns);
    sciplot::Canvas canvasImpSampNum = MakeGraphNoInt<D, N>(
        alphaVals, energyValsImpSampNum, errorValsImpSampNum, confIntsImpSampNum, exactEns);

    canvasMetrAn.save(metrNumFile);
    canvasImpSampAn.save(impSampNumFile);
}

// Interacting Harmonic Oscillator
template <vmcp::Dimension D, vmcp::ParticNum N>
void HOInt(vmcp::StatFuncType statFunction, std::vector<vmcp::FPType> &energyVals,
           std::vector<vmcp::ConfInterval> confInts, std::vector<vmcp::IntType> &NVals) {
    using namespace vmcp;
    assert(N != 1);
    CoordBounds<D> const coordBounds = MakeCoordBounds<D>(Coordinate{-10}, Coordinate{10});

    struct PotHO {
        Masses<N> m;
        FPType omegaHO;
        FPType gamma;
        FPType operator()(Positions<D, N> x) const {
            Coordinate *begin = &x[0][0];
            Coordinate *end = &x[0][0] + N * D;

            FPType result =
                std::transform_reduce(begin, end, FPType{0.f}, std::plus<>(),
                                      [D_ = D, gamma_ = gamma, &begin](Coordinate &c) {
                                          // Compute the index of the current element
                                          auto const index = &c - begin;
                                          // Check if it is in the last dimension
                                          bool isLastDimension =
                                              ((static_cast<UIntType>(index) + 1u) % D_ == 0) && (D_ != 1);
                                          return c.val * c.val * (isLastDimension ? gamma_ * gamma_ : 1);
                                      }) *
                m[0].val * omegaHO * omegaHO / 2;
            assert(!std::isnan(result));

            return result;
        }
    };
    struct WavefHO {
        FPType beta;
        FPType a;
        FPType operator()(Positions<D, N> x, VarParams<1> alpha) const {
            //   Harmonic oscillator term
            Coordinate *begin = &x[0][0];
            Coordinate *end = &x[0][0] + N * D;

            FPType expArg = std::transform_reduce(
                begin, end, FPType{0}, std::plus<>(), [beta_ = beta, &begin](Coordinate &c) {
                    auto const index = &c - begin;
                    bool isLastDimension = ((static_cast<UIntType>(index) + 1u) % D == 0) && (D != 1);
                    return c.val * c.val * (isLastDimension ? beta_ : 1);
                });
            // Interaction term
            FPType interactionTerm = FPType{0.f};

            for (ParticNum i = 0u; i < N - 1; i++) {
                for (ParticNum j = i + 1u; j < N; j++) {
                    FPType r_ij = Distance(x[i], x[j]);
                    if (r_ij > a) {
                        interactionTerm += std::log(FPType{1} - a / r_ij);
                    } else {
                        return FPType{0};
                    }
                }
            }
            assert(!std::isnan(interactionTerm));

            return std::exp(-alpha[0].val * expArg + interactionTerm);
        }
    };
    struct LaplHO {
        FPType beta;
        FPType a;
        UIntType particle;

        LaplHO(FPType beta_, FPType a_, UIntType particle_) : beta{beta_}, a{a_}, particle{particle_} {
            assert(particle < N);
        }

        LaplHO() : beta{}, a{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<1> alpha) const {
            UIntType uPar = particle;
            Coordinate *begin = &x[uPar][0];
            Coordinate *end = &x[uPar][0] + D;

            FPType sumXSqrd = std::transform_reduce(
                begin, end, FPType{0.f}, std::plus<>(), [D_ = D, beta_ = beta, &begin](Coordinate &c) {
                    // Compute the index of the current element
                    auto const index = &c - begin;
                    // Check if it is in the last dimension
                    bool isLastDimension = ((static_cast<UIntType>(index) + 1u) % D_ == 0) && (D_ != 1);
                    return c.val * c.val * (isLastDimension ? beta_ * beta_ : 1);
                });
            FPType phiK = std::exp(-alpha[0].val * sumXSqrd);

            FPType nonIntLapl = (std::pow(2 * alpha[0].val, 2) * sumXSqrd -
                                 2 * alpha[0].val * ((D == 1) ? 1 : (D - 1 + beta))) *
                                phiK;
            assert(!std::isnan(nonIntLapl));

            std::vector<FPType> gradHO(D);
            std::vector<FPType> gradInt(D);
            FPType innerProd{0};

            std::generate_n(std::back_inserter(gradHO), D, [&, d = Dimension{0u}]() mutable {
                FPType result =
                    -2 * alpha[0].val * x[particle][d].val * ((d == (D - 1)) && (D != 1) ? beta : 1) * phiK;
                d++;
                return result;
            });
            for (ParticNum n = 0u; n < N; n++) {
                if (n == particle) {
                    continue;
                }
                std::generate_n(std::back_inserter(gradInt), D, [&, d = 0u]() mutable {
                    FPType r_pn = Distance(x[particle], x[n]);
                    FPType u_pnPrime = a / (r_pn * (r_pn - a));
                    FPType result = (x[particle][d].val - x[n][d].val) * u_pnPrime / r_pn;
                    d++;
                    return result;
                });
                innerProd += std::inner_product(gradHO.begin(), gradHO.end(), gradInt.begin(), FPType{0.f});
            }

            FPType pureIntTerms{0};
            std::vector<FPType> intVector(D);

            for (ParticNum n = 0u; n < N; n++) {
                if (n == particle) {
                    continue;
                }
                FPType r_pn = Distance(x[particle], x[n]);
                if (r_pn <= a) {
                    return FPType{0.f};
                }

                FPType u_pnPrime = a / (r_pn * (r_pn - a));
                FPType u_pnPrime2 = (a * a - 2 * a * r_pn) / std::pow(r_pn * (r_pn - a), 2);

                for (Dimension d = 0u; d < D; d++) {
                    intVector[d] += (x[particle][d].val - x[n][d].val) * u_pnPrime / r_pn;
                }

                pureIntTerms += u_pnPrime2 + 2 * u_pnPrime / r_pn;
            }
            pureIntTerms +=
                std::inner_product(intVector.begin(), intVector.end(), intVector.begin(), FPType{0.f});
            assert(!std::isnan(pureIntTerms));

            FPType result = WavefHO{beta, a}(x, VarParams<1>{alpha}) *
                            ((nonIntLapl + 2 * innerProd) / phiK + pureIntTerms);
            assert(!std::isnan(result));
            // std::cout << "NIL: " << nonIntLapl << "\n";
            //  std::cout << "Lapl: " << result << "\n";

            return result;
        }
    };

    Masses<N> mass;
    mass.fill(M);

    PotHO potHO{mass, OmegaHO, Gamma};
    WavefHO wavefHO{Beta, ADistance};
    std::array<LaplHO, N> laplHO;
    std::generate(laplHO.begin(), laplHO.end(),
                  [counter = ParticNum{0}]() mutable { return LaplHO{Beta, ADistance, counter++}; });

    Positions<D, N> startPoss = BuildFCCStartPoint_<D, N>(coordBounds, latticeSpacing);

    // One variational parameter
    ParamBounds<1> const alphaBounds{Bound{VarParam{0.1f}, VarParam{2}}};

    VMCResult<1> const vmcrBest =
        VMCEnergy<D, N, 1>(wavefHO, startPoss, alphaBounds, laplHO, mass, potHO, coordBounds, numEnergies,
                           statFunction, bootstrapSamples, gen);
    ConfInterval confInt = GetConfInt(vmcrBest.energy, vmcrBest.stdDev, confLvl);
    std::cout << "Energy with the best alpha for N=" + std::to_string(N) + ":\n"
              << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy << " +/- "
              << vmcrBest.stdDev << "\tconf. int. with conf. lvl. of " << confLvl << "%: " << confInt.min
              << " - " << confInt.max << '\n';

    energyVals.push_back(vmcrBest.energy.val);
    confInts.push_back(confInt);
    NVals.push_back(N);
}

int main() {
    vmcp::StatFuncType statFunction = vmcp::StatFuncType::bootstrap;

    // Non interacting plots of energy as a function of (non variational) parameter alpha
    HONoInt<3, 5>(statFunction);
    // HONoInt<3, 10>(statFunction);
    // HONoInt<3, 50>(statFunction);

    // Interacting plots of energy as a function of the number of particles using alpha as a variational
    // parameter
    std::vector<vmcp::FPType> energyVals;
    std::vector<vmcp::FPType> errorVals;
    std::vector<vmcp::ConfInterval> confInts;
    std::vector<vmcp::IntType> NVals;

    // HOInt<3, 10>(statFunction, energyVals, confInts, NVals);
    /*HOInt<3, 5>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 10>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 15>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 20>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 25>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 30>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 35>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 40>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 45>(statFunction, energyVals, confInts, NVals);
     HOInt<3, 50>(statFunction, energyVals, confInts, NVals);*/

    vmcp::DrawGraphInt(NVals, energyVals, confInts);
}
