//!
//! @file main.cpp
//! @brief main file
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the helper functions used inside ComputeEnergies in main.cpp
//!

#include "main.hpp"
#include <chrono>
#include <thread>

// LF TODO: passa aVMCEnergues poss invece di aDistance
// LF TODO: alpha più fitti

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

vmcp::RandomGenerator gen{(std::random_device())()};

template <vmcp::Dimension D, vmcp::ParticNum N>
void HONoInt(vmcp::StatFuncType F) {
    using namespace vmcp;

    // FIRST PART
    // Computing energies and standard deviations

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
            assert(!std::isnan(potential));

            return result;
        }
    };
    struct WavefHO {
        FPType alpha;
        FPType beta;
        FPType a;
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
        FPType a;
        UIntType dimension;
        UIntType particle;
        FirstDerHO(FPType alpha_, FPType beta_, FPType a_, bool interactions_, UIntType dimension_,
                   UIntType particle_)
            : alpha{alpha_}, beta{beta_}, a{a_}, dimension{dimension_}, particle{particle_} {
            assert(dimension < D);
            assert(particle < N);
        }
        FirstDerHO() : alpha{}, beta{}, a{}, dimension{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            // Harmonic oscillator first derivative term
            FPType result = -2 * alpha * x[particle][dimension].val *
                            ((dimension == (D - 1)) && (D != 1) ? beta : 1) *
                            WavefHO{alpha, beta, a}(x, VarParams<0>{});
            assert(!std::isnan(result));

            return result;
        }
    };
    struct LaplHO {
        FPType alpha;
        FPType beta;
        FPType a;
        UIntType particle;

        LaplHO(FPType alpha_, FPType beta_, FPType a_, UIntType particle_)
            : alpha{alpha_}, beta{beta_}, a{a_}, particle{particle_} {
            assert(particle < N);
        }

        LaplHO() : alpha{}, beta{}, a{}, particle{} {}

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
                WavefHO{alpha, beta, a, false}(x, VarParams<0>{});
            assert(!std::isnan(result));

            return result;
        }
    };

    Masses<N> mass;
    mass.fill(M);

    std::vector<FPType> alphaVals;
    std::generate_n(std::back_inserter(alphaVals), 18, [i = 0]() mutable -> FPType {
        return FPType{0.1f} + static_cast<FPType>(++i) * FPType{0.05f};
    });

    // Exact energies vector (all the elements equal the exact energy value for the ground state)
    vmcp::Energy const exactEn{((D == 1) ? 1 : (D - 1 + Gamma)) * N * hbar * OmegaHO / 2};
    std::vector<FPType> exactEns;
    std::generate_n(std::back_inserter(exactEns), 100, [exactEn]() -> FPType { return exactEn.val; });

    // Analytic vectors
    std::vector<FPType> energyValsMetrAn;
    std::vector<FPType> energyValsImpSampAn;
    std::vector<FPType> errorValsMetrAn;
    std::vector<FPType> errorValsImpSampAn;

    // Numeric vectors
    std::vector<FPType> energyValsMetrNum;
    std::vector<FPType> energyValsImpSampNum;
    std::vector<FPType> errorValsMetrNum;
    std::vector<FPType> errorValsImpSampNum;

    // Perform Monte Carlo Method
    for (FPType alphaVal : alphaVals) {
        PotHO potHO{mass, OmegaHO, Gamma};
        WavefHO wavefHO{alphaVal, Beta, ADistance};
        Gradients<D, N, FirstDerHO> gradsHO;
        for (ParticNum n = 0u; n < N; ++n) {
            for (Dimension d = 0u; d < D; ++d) {
                gradsHO[n][d] = FirstDerHO{alphaVal, Beta, ADistance, d, n};
            }
        }

        std::array<LaplHO, N> laplHO;
        std::generate(laplHO.begin(), laplHO.end(), [counter = UIntType{0}, alphaVal]() mutable {
            return LaplHO{alphaVal, Beta, ADistance, counter++};
        });

        vmcp::Positions<D, N> poss = BuildFCCStartPoint_<D, N>(coordBounds, latticeSpacing);

        // Analaytic
        VMCResult<0> const vmcrMetrAn =
            VMCEnergy<D, N, 0>(wavefHO, poss, ParamBounds<0>{}, laplHO, mass, potHO, coordBounds, numEnergies,
                               F, bootstrapSamples, gen, latticeSpacing);
        std::cout << "Metropolis:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrAn.energy.val << " +/- " << vmcrMetrAn.stdDev.val << '\n';
        VMCResult<0> const vmcrImpSampAn =
            VMCEnergy<D, N, 0>(wavefHO, poss, ParamBounds<0>{}, gradsHO, laplHO, mass, potHO, coordBounds,
                               numEnergies, F, bootstrapSamples, gen, latticeSpacing);
        std::cout << "ImpSamp:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrImpSampAn.energy.val << " +/- " << vmcrImpSampAn.stdDev.val << '\n';

        energyValsMetrAn.push_back(vmcrMetrAn.energy.val);
        energyValsImpSampAn.push_back(vmcrImpSampAn.energy.val);
        errorValsMetrAn.push_back(vmcrMetrAn.stdDev.val);
        errorValsImpSampAn.push_back(vmcrImpSampAn.stdDev.val);

        // Numeric
        VMCResult<0> const vmcrMetrNum =
            VMCEnergy<D, N, 0>(wavefHO, poss, ParamBounds<0>{}, false, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, F, bootstrapSamples, gen);
        VMCResult<0> const vmcrImpSampNum =
            VMCEnergy<D, N, 0>(wavefHO, poss, ParamBounds<0>{}, true, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, F, bootstrapSamples, gen);

        energyValsMetrNum.push_back(vmcrMetrNum.energy.val);
        energyValsImpSampNum.push_back(vmcrImpSampNum.energy.val);
        errorValsMetrNum.push_back(vmcrMetrNum.stdDev.val);
        errorValsImpSampNum.push_back(vmcrImpSampNum.stdDev.val);
    }

    // SECOND PART
    // Plotting the energies

    // Construct file paths
    std::string folder = "./artifacts/NoInt/D=" + std::to_string(D) + "_N=" + std::to_string(N);
    std::string metrAnFile = folder + "/plot_Metr_An.pdf";
    std::string impSampAnFile = folder + "/plot_ImpSamp_An.pdf";
    std::string metrNumFile = folder + "/plot_Metr_Num.pdf";
    std::string impSampNumFile = folder + "/plot_ImpSamp_Num.pdf";

    std::filesystem::create_directories(folder);

    // Analaytic
    sciplot::Canvas canvasMetrAn =
        DrawGraphNoInt<D, N>(alphaVals, energyValsMetrAn, errorValsMetrAn, exactEns);
    sciplot::Canvas canvasImpSampAn =
        DrawGraphNoInt<D, N>(alphaVals, energyValsImpSampAn, errorValsImpSampAn, exactEns);

    canvasMetrAn.save(metrAnFile);
    canvasImpSampAn.save(impSampAnFile);

    // Numeric
    sciplot::Canvas canvasMetrNum =
        DrawGraphNoInt<D, N>(alphaVals, energyValsMetrNum, errorValsMetrNum, exactEns);
    sciplot::Canvas canvasImpSampNum =
        DrawGraphNoInt<D, N>(alphaVals, energyValsImpSampNum, errorValsImpSampNum, exactEns);

    canvasMetrAn.save(metrNumFile);
    canvasImpSampAn.save(impSampNumFile);
}

template <vmcp::Dimension D, vmcp::ParticNum N>
void HOInt(vmcp::StatFuncType F, std::vector<vmcp::FPType> &energyVals, std::vector<vmcp::FPType> &errorVals,
           std::vector<vmcp::IntType> &NVals) {
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
            assert(!std::isnan(potential));

            return result;
        }
    };
    struct WavefHO {
        FPType beta;
        FPType a;
        bool interactions;
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
            if (interactions) {
                for (ParticNum i = 0u; i < N - 1; ++i) {
                    for (ParticNum j = i + 1u; j < N; ++j) {
                        FPType r_ij = Distance(x[i], x[j]);
                        if (r_ij > a)
                            interactionTerm += std::log(FPType{1.f} - a / r_ij);
                        else {
                            return FPType{0};
                        }
                    }
                }
                assert(!std::isnan(interactionTerm));
            }

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

            FPType nonIntLapl =
                (std::pow(2 * alpha[0].val, 2) *
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
                 2 * alpha[0].val * ((D == 1) ? 1 : (D - 1 + beta))) *
                WavefHO{beta, a, false}(x, VarParams<1>{alpha});
            assert(!std::isnan(nonIntLapl));

            FPType pureIntTerms{0.f};
            std::array<FPType, D> arrInt = {FPType{0.f}};

            for (ParticNum n = 0u; n < N; ++n) {
                if (n == particle) {
                    continue;
                }
                FPType r_pn = Distance(x[particle], x[n]);
                if (r_pn <= a) {
                    return FPType{0.f};
                }

                FPType u_pnPrime = a / (r_pn * (r_pn - a));
                FPType u_pnPrime2 = (a * a - 2 * a * r_pn) / std::pow(r_pn * (r_pn - a), 2);

                for (Dimension d = 0u; d < D; ++d) {
                    arrInt[d] += (x[particle][d].val - x[n][d].val) * u_pnPrime / r_pn;
                }

                pureIntTerms += u_pnPrime2 + 2 * u_pnPrime / r_pn;
            }
            pureIntTerms += std::inner_product(arrInt.begin(), arrInt.end(), arrInt.begin(), FPType{0.f});
            assert(!std::isnan(pureIntTerms));

            std::array<FPType, D> gradHO;
            std::array<FPType, D> gradInt;

            std::generate_n(gradHO.begin(), D, [&, d = UIntType{0}]() mutable {
                return -2 * alpha[0].val * x[particle][d].val * ((d == (D - 1)) && (D != 1) ? beta : 1) *
                       WavefHO{beta, a, false}(x, VarParams<1>{alpha});
                ++d;
            });
            for (ParticNum n = 0u; n < N; ++n) {
                if (n == particle) {
                    continue;
                }
                std::generate_n(gradInt.begin(), D, [&, d = 0u]() mutable {
                    FPType r_pn = Distance(x[particle], x[n]);
                    FPType u_pnPrime = a / (r_pn * (r_pn - a));
                    return (x[particle][d].val - x[n][d].val) * u_pnPrime / r_pn;
                    ++d;
                });
            }

            FPType result = WavefHO{beta, a, true}(x, VarParams<1>{alpha}) *
                            ((nonIntLapl + 2 * std::inner_product(gradHO.begin(), gradHO.end(),
                                                                  gradInt.begin(), FPType{0.f})) /
                                 WavefHO{beta, a, false}(x, VarParams<1>{alpha}) +
                             pureIntTerms);
            assert(!std::isnan(result));

            return result;
        }
    };

    Masses<N> mass;
    mass.fill(M);

    PotHO potHO{mass, OmegaHO, Gamma};
    WavefHO wavefHO{Beta, ADistance, true};
    std::array<LaplHO, N> laplHO;
    std::generate(laplHO.begin(), laplHO.end(),
                  [counter = UIntType{0}]() mutable { return LaplHO{Beta, ADistance, counter++}; });

    vmcp::Positions<D, N> poss = BuildFCCStartPoint_<D, N>(coordBounds, latticeSpacing);

    // One variational parameter
    ParamBounds<1> const alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

    VMCResult<1> const vmcrBest = VMCEnergy<D, N, 1>(wavefHO, poss, alphaBounds, laplHO, mass, potHO,
                                                     coordBounds, numEnergies, F, bootstrapSamples, gen);
    std::cout << "Energy with the best alpha for N=" + std::to_string(N) + ":\n"
              << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val << " +/- "
              << vmcrBest.stdDev.val << '\n';

    energyVals.push_back(vmcrBest.energy.val);
    errorVals.push_back(vmcrBest.stdDev.val);
    NVals.push_back(N);
}

int main() {
    vmcp::StatFuncType statFunc = vmcp::StatFuncType::bootstrap;

    /*   // Non interacting plots of energy as a function of (non variational) parameter alpha
       HONoInt<3, 5>(statFunc);
       HONoInt<3, 10>(statFunc);
       HONoInt<3, 50>(statFunc);

   */
    // Interacting plots of energy as a function of the number of particles using alpha as a variational
    // parameter
    std::vector<vmcp::FPType> energyVals;
    std::vector<vmcp::FPType> errorVals;
    std::vector<vmcp::IntType> NVals;

    HOInt<3, 2>(statFunc, energyVals, errorVals, NVals);
    HOInt<3, 3>(statFunc, energyVals, errorVals, NVals);
    /* HOInt<3, 5>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 10>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 15>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 20>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 25>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 30>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 35>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 35>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 40>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 45>(statFunc, energyVals, errorVals, NVals);
     HOInt<3, 50>(statFunc, energyVals, errorVals, NVals);*/

    vmcp::DrawGraphInt(NVals, energyVals, errorVals);
}

/*
std::cout << "i: " << i << ""; // TODO: remove
        std::cout << "  VP: " << currentParams[0].val << "\n";*/