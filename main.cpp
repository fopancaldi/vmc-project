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

// Constant Parameters
constexpr vmcp::FPType betaInit = vmcp::FPType{2.82843f};
constexpr vmcp::FPType gammaInit = vmcp::FPType{2.82843f};
constexpr vmcp::FPType aInit = vmcp::FPType{1.f};
constexpr vmcp::FPType omegaInit{1.f};
constexpr vmcp::Masses<1> mInit{1.f};

constexpr vmcp::FPType derivativeStep{0.01f};
constexpr vmcp::IntType numEnergies = 1 << 7;
constexpr vmcp::IntType numSamples = 1000;

vmcp::RandomGenerator gen{(std::random_device())()};

template <vmcp::Dimension D, vmcp::ParticNum N>
void HO(vmcp::StatFuncType F, bool interactions) {
    using namespace vmcp;

    // FIRST PART
    // Computing energies and standard deviations

    CoordBounds<D> const coordBounds = MakeCoordBounds<D>(Coordinate{-10}, Coordinate{10});

    struct PotHO {
        // TODO: masses N
        Masses<1> m;
        FPType omega;
        FPType gamma;
        FPType operator()(Positions<D, N> x) const {
            FPType *begin = &x[0][0].val;
            FPType *end = &x[0][0].val + N * D;

            FPType m_ = m[0].val;
            FPType omega_ = omega;
            FPType gamma_ = gamma;
            IntType const D_ = D;

            FPType potential =
                std::transform_reduce(begin, end, FPType{0.f}, std::plus<>(),
                                      [&D_, &omega_, &m_, &gamma_, &begin](FPType &val) {
                                          // Compute the index of the current element
                                          auto index = &val - begin;
                                          // Check if it is in the last dimension
                                          bool isLastDimension = ((index + 1) % D_ == 0) && (D_ != 1);
                                          return val * val *
                                                 (isLastDimension ? gamma_ * gamma_ : omega_ * omega_ * m_);
                                      }) /
                2;
            assert(!std::isnan(potential));

            return potential;
        }
    };

    // LF TODO: rename expArg (is the argument of exponential)
    struct WavefHO {
        FPType alpha;
        FPType beta;
        FPType a;
        bool interactions;

        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            //   Harmonic oscillator term
            FPType *begin = &x[0][0].val;
            FPType *end = &x[0][0].val + N * D;
            FPType expArg = std::transform_reduce(
                begin, end, FPType{0.f}, std::plus<>(), [&beta = beta, &begin](FPType &val) {
                    auto index = &val - begin;
                    bool isLastDimension = ((static_cast<UIntType>(index) + 1u) % D == 0) && (D != 1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    return val * val * (isLastDimension ? beta : 1);
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
                            return FPType{0.f};
                        }
                    }
                }
                assert(!std::isnan(interactionTerm));
            }

            return std::exp(-alpha * expArg + interactionTerm);
        }
    };

    struct FirstDerHO {
        FPType alpha;
        FPType beta;
        FPType a;
        bool interactions;
        UIntType dimension;
        UIntType particle;

        FirstDerHO(FPType alpha_, FPType beta_, FPType a_, bool interactions_, UIntType dimension_,
                   UIntType particle_)
            : alpha{alpha_}, beta{beta_}, a{a_}, interactions{interactions_}, dimension{dimension_},
              particle{particle_} {
            assert(dimension < D);
            assert(particle < N);
        }

        FirstDerHO() : alpha{}, beta{}, a{}, interactions{}, dimension{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            // Harmonic oscillator first derivative term
            FPType firstDerHO = -2 * alpha * x[particle][dimension].val *
                                ((dimension == (D - 1)) && (D != 1) ? beta : 1) *
                                WavefHO{alpha, beta, a, interactions}(x, VarParams<0>{});
            assert(!std::isnan(firstDerHO));

            // Interaction first derivative term term
            FPType firstDerInt{0.f};
            if (interactions) {
                for (ParticNum n = 0u; n < N; ++n) {
                    if (n == particle) {
                        continue;
                    }
                    FPType r_pn = Distance(x[particle], x[n]);
                    if (r_pn <= a) {
                        return FPType{0.f};
                    }
                    FPType u_pnPrime = a / (r_pn * (r_pn - a));
                    firstDerInt += (x[particle][dimension].val - x[n][dimension].val) * (u_pnPrime) / r_pn;
                }

                firstDerInt *= WavefHO{alpha, beta, a, true}(x, VarParams<0>{});
                assert(!std::isnan(firstDerInt));
            }

            return firstDerHO + firstDerInt;
        }
    };

    struct LaplHO {
        FPType alpha;
        FPType beta;
        FPType a;
        bool interactions;
        UIntType particle;

        LaplHO(FPType alpha_, FPType beta_, FPType a_, bool interactions_, UIntType particle_)
            : alpha{alpha_}, beta{beta_}, a{a_}, interactions{interactions_}, particle{particle_} {
            assert(particle < N);
        }

        LaplHO() : alpha{}, beta{}, a{}, interactions{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<0>) const {
            UIntType uPar = particle;
            FPType *begin = &x[uPar][0].val;
            FPType *end = &x[uPar][0].val + (D - 1);
            UIntType D_ = D;

            FPType nonIntLapl =
                (std::pow(2 * alpha, 2) *
                     std::transform_reduce(begin, end, FPType{0.f}, std::plus<>(),
                                           [&D_, &beta = beta, &begin](FPType &val) {
                                               // Compute the index of the current element
                                               auto index = &val - begin;

                                               // Check if it is in the last dimension
                                               bool isLastDimension =
                                                   ((static_cast<UIntType>(index) + 1u) % D_ == 0) &&
                                                   (D_ != 1);
                                               return val * val * (isLastDimension ? beta * beta : 1);
                                           }) -
                 2 * alpha * (D == 1 ? 1 : D - 1 + beta)) *
                WavefHO{alpha, beta, a, false}(x, VarParams<0>{});
            assert(!std::isnan(nonIntLapl));

            if (interactions) {
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
                    return -2 * alpha * x[particle][d].val * ((d == (D - 1)) && (D != 1) ? beta : 1) *
                           WavefHO{alpha, beta, a, false}(x, VarParams<0>{});
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

                FPType intLapl = WavefHO{alpha, beta, a, true}(x, VarParams<0>{}) *
                                 ((nonIntLapl + 2 * std::inner_product(gradHO.begin(), gradHO.end(),
                                                                       gradInt.begin(), FPType{0.f})) /
                                      WavefHO{alpha, beta, a, false}(x, VarParams<0>{}) +
                                  pureIntTerms);
                assert(!std::isnan(intLapl));

                return intLapl;
            } else {
                return nonIntLapl;
            }
        }
    };

    Masses<N> mass;
    std::fill(std::begin(mass), std::end(mass), Mass{mInit[0]});

    std::vector<FPType> alphaVals;
    std::generate_n(std::back_inserter(alphaVals), 18, [i = 0]() mutable -> FPType {
        return FPType{0.1f} + static_cast<FPType>(++i) * FPType{0.05f};
    });

    // Exact energies vector (all the elements equal the exact energy value for the ground state)
    vmcp::Energy const exactEn{D * N * hbar * omegaInit / 2};
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
        PotHO potHO{mInit, omegaInit, gammaInit};
        WavefHO wavefHO{alphaVal, betaInit, aInit, interactions};
        vmcp::Gradients<D, N, FirstDerHO> gradsHO;
        for (ParticNum n = 0u; n < N; ++n) {
            for (Dimension d = 0u; d < D; ++d) {
                gradsHO[n][d] = FirstDerHO{alphaVal, betaInit, aInit, interactions, d, n};
            }
        }
        std::array<LaplHO, N> laplHO;
        std::generate(laplHO.begin(), laplHO.end(),
                      [counter = UIntType{0}, &alphaVal, interactions]() mutable {
                          return LaplHO{alphaVal, betaInit, aInit, interactions, counter++};
                      });

        // Analaytic
        VMCResult<0> const vmcrMetrAn = VMCEnergy<D, N, 0>(wavefHO, ParamBounds<0>{}, laplHO, mass, potHO,
                                                           coordBounds, numEnergies, F, numSamples, gen);
        std::cout << "Metropolis:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrAn.energy.val << " +/- " << vmcrMetrAn.stdDev.val << '\n';
        VMCResult<0> const vmcrImpSampAn =
            VMCEnergy<D, N, 0>(wavefHO, ParamBounds<0>{}, gradsHO, laplHO, mass, potHO, coordBounds,
                               numEnergies, F, numSamples, gen);
        std::cout << "ImpSamp:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrImpSampAn.energy.val << " +/- " << vmcrImpSampAn.stdDev.val << '\n';

        energyValsMetrAn.push_back(vmcrMetrAn.energy.val);
        energyValsImpSampAn.push_back(vmcrImpSampAn.energy.val);
        errorValsMetrAn.push_back(vmcrMetrAn.stdDev.val);
        errorValsImpSampAn.push_back(vmcrImpSampAn.stdDev.val);

        // Numeric
        VMCResult<0> const vmcrMetrNum =
            VMCEnergy<D, N, 0>(wavefHO, ParamBounds<0>{}, false, derivativeStep, mass, potHO, coordBounds,
                               numEnergies, F, numSamples, gen);
        VMCResult<0> const vmcrImpSampNum =
            VMCEnergy<D, N, 0>(wavefHO, ParamBounds<0>{}, true, derivativeStep, mass, potHO, coordBounds,
                               numEnergies, F, numSamples, gen);

        energyValsMetrNum.push_back(vmcrMetrNum.energy.val);
        energyValsImpSampNum.push_back(vmcrImpSampNum.energy.val);
        errorValsMetrNum.push_back(vmcrMetrNum.stdDev.val);
        errorValsImpSampNum.push_back(vmcrImpSampNum.stdDev.val);
    }

    // SECOND PART
    // Plotting the energies

    // Construct file paths
    std::string folder = "./artifacts/D=" + std::to_string(D) + "_N=" + std::to_string(N);
    std::string metrAnFile = folder + "/plot_Metr_An.pdf";
    std::string impSampAnFile = folder + "/plot_ImpSamp_An.pdf";
    std::string metrNumFile = folder + "/plot_Metr_Num.pdf";
    std::string impSampNumFile = folder + "/plot_ImpSamp_Num.pdf";

    std::filesystem::create_directories(folder);

    // Analaytic
    sciplot::Canvas canvasMetrAn = DrawGraph<D, N>(alphaVals, energyValsMetrAn, errorValsMetrAn, exactEns);
    sciplot::Canvas canvasImpSampAn =
        DrawGraph<D, N>(alphaVals, energyValsImpSampAn, errorValsImpSampAn, exactEns);

    canvasMetrAn.save(metrAnFile);
    canvasImpSampAn.save(impSampAnFile);

    // Numeric
    sciplot::Canvas canvasMetrNum = DrawGraph<D, N>(alphaVals, energyValsMetrNum, errorValsMetrNum, exactEns);
    sciplot::Canvas canvasImpSampNum =
        DrawGraph<D, N>(alphaVals, energyValsImpSampNum, errorValsImpSampNum, exactEns);

    canvasMetrAn.save(metrNumFile);
    canvasImpSampAn.save(impSampNumFile);
}

int main() {
    bool interactions = false;
    vmcp::StatFuncType statFunc = vmcp::StatFuncType::regular;
    HO<1, 1>(statFunc, interactions);
    // HO<3, 50>(statFunc, interactions);
    // HO<3, 100>(statFunc, interactions);
}
