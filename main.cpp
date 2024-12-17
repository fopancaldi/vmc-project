//!
//! @file main.cpp
//! @brief main file
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!

#include "main.hpp"

// Constant Parameters
constexpr vmcp::FPType Beta = vmcp::FPType{2.82843f};
constexpr vmcp::FPType Gamma = vmcp::FPType{2.82843f};
constexpr vmcp::FPType OmegaHO{1};
constexpr vmcp::Mass ParticlesMass{1};
const vmcp::FPType ADistance = 0.0043 * std::sqrt(vmcp::hbar / (ParticlesMass.val * OmegaHO));
const vmcp::FPType latticeSpacing = 100 * ADistance * std::sqrt(vmcp::FPType{2});

constexpr vmcp::FPType derivativeStep{0.01f};
constexpr vmcp::IntType numEnergies = 1 << 9;
constexpr vmcp::IntType bootstrapSamples = 5000;
constexpr vmcp::FPType confLvl{95};

constexpr vmcp::UIntType genSeed = 19436u;
vmcp::RandomGenerator gen(genSeed);

// Non Interacting Harmonic Oscillator
template <vmcp::Dimension D, vmcp::ParticNum N>
void HONoInt(vmcp::StatFuncType statFunction) {
    using namespace vmcp;

    //////////////////////////////////////////////////
    // OBJECTS USED INSIDE THE WHOLE FUNCTIONS

    CoordBounds<D> const coordBounds = MakeCoordBounds<D>(Coordinate{-10}, Coordinate{10});
    Positions<D, N> startPoss = BuildFCCStartPoint_<D, N>(coordBounds, latticeSpacing);

    // STRUCTS WITHOUT VARIATIONAL PARAMETERS
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
    mass.fill(ParticlesMass);

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

    std::vector<FPType> alphaVals;

    //////////////////////////////////////////////////
    // FIRST GROUP OF PLOTS

    // Generate values for alpha for the first group of plots only (vector will be cleared later)
    std::generate_n(std::back_inserter(alphaVals), 29, [i = 0]() mutable -> FPType {
        return FPType{0.2f} + static_cast<FPType>(++i) * FPType{0.02f};
    });

    // Perform Monte Carlo Methods with alphaVals
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

        // Analaytic
        VMCResult<0> const vmcrMetrAn =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, laplHO, mass, potHO, coordBounds,
                               numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntMetrAn = GetConfInt(vmcrMetrAn.energy, vmcrMetrAn.stdDev, confLvl);
        std::cout << "Metropolis Analytic:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrAn.energy << " +/- " << vmcrMetrAn.stdDev << "\tconf. int. with conf. lvl. of "
                  << confLvl << "%: " << confIntMetrAn.min.val << " - " << confIntMetrAn.max << '\n';

        VMCResult<0> const vmcrImpSampAn =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, gradsHO, laplHO, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntImpSampAn = GetConfInt(vmcrImpSampAn.energy, vmcrImpSampAn.stdDev, confLvl);
        std::cout << "ImpSamp Analytic:\n"
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
        std::cout << "Metropolis Numeric:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrNum.energy << " +/- " << vmcrMetrNum.stdDev << "\tconf. int. with conf. lvl. of "
                  << confLvl << "%: " << confIntMetrNum.min << " - " << confIntMetrNum.max << "\n\n";

        VMCResult<0> const vmcrImpSampNum =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, true, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntImpSampNum = GetConfInt(vmcrImpSampNum.energy, vmcrImpSampNum.stdDev, confLvl);
        std::cout << "ImpSamp Numeric:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrImpSampNum.energy << " +/- " << vmcrImpSampNum.stdDev
                  << "\tconf. int. with conf. lvl. of " << confLvl << "%: " << confIntImpSampNum.min << " - "
                  << confIntImpSampNum.max << "\n\n";

        energyValsMetrNum.push_back(vmcrMetrNum.energy.val);
        energyValsImpSampNum.push_back(vmcrImpSampNum.energy.val);
        confIntsMetrNum.push_back(confIntMetrNum);
        confIntsImpSampNum.push_back(confIntImpSampNum);
    }

    // Plotting the energies with alphaVals:
    // Construct file paths
    std::string folder = "./artifacts/NoInt/Regular/D=" + std::to_string(D) + "_N=" + std::to_string(N);
    std::string metrAnFile = folder + "/plot_Metr_An.pdf";
    std::string impSampAnFile = folder + "/plot_ImpSamp_An.pdf";
    std::string metrNumFile = folder + "/plot_Metr_Num.pdf";
    std::string impSampNumFile = folder + "/plot_ImpSamp_Num.pdf";

    std::filesystem::create_directories(folder);

    // Analaytic
    sciplot::Canvas canvasMetrAn =
        MakeGraphNoInt<D, N>(alphaVals, energyValsMetrAn, confIntsMetrAn, exactEns, "Metropolis Analytic");
    sciplot::Canvas canvasImpSampAn = MakeGraphNoInt<D, N>(alphaVals, energyValsImpSampAn, confIntsImpSampAn,
                                                           exactEns, "Imp. Samp. Analytic");

    canvasMetrAn.save(metrAnFile);
    canvasImpSampAn.save(impSampAnFile);

    // Numeric
    sciplot::Canvas canvasMetrNum =
        MakeGraphNoInt<D, N>(alphaVals, energyValsMetrNum, confIntsMetrNum, exactEns, "Metropolis Numeric");
    sciplot::Canvas canvasImpSampNum = MakeGraphNoInt<D, N>(
        alphaVals, energyValsImpSampNum, confIntsImpSampNum, exactEns, "Imp. Samp. Numeric");

    canvasMetrNum.save(metrNumFile);
    canvasImpSampNum.save(impSampNumFile);

    //////////////////////////////////////////////////
    // SECOND GROUP OF PLOTS

    // Clear data vectors for the next groups of plots
    energyValsMetrAn.clear();
    energyValsImpSampAn.clear();
    confIntsMetrNum.clear();
    confIntsImpSampNum.clear();
    alphaVals.clear();

    // Generate values for alpha for the second group of plots
    std::generate_n(std::back_inserter(alphaVals), 11, [i = 0]() mutable -> FPType {
        return FPType{0.44f} + static_cast<FPType>(++i) * FPType{0.01f};
    });

    // Perform Monte Carlo Methods with NEW alphaVals
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

        // Analaytic
        VMCResult<0> const vmcrMetrAn =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, laplHO, mass, potHO, coordBounds,
                               numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntMetrAn = GetConfInt(vmcrMetrAn.energy, vmcrMetrAn.stdDev, confLvl);
        std::cout << "Metropolis Analytic:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrAn.energy << " +/- " << vmcrMetrAn.stdDev << "\tconf. int. with conf. lvl. of "
                  << confLvl << "%: " << confIntMetrAn.min.val << " - " << confIntMetrAn.max << "\n";

        VMCResult<0> const vmcrImpSampAn =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, gradsHO, laplHO, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntImpSampAn = GetConfInt(vmcrImpSampAn.energy, vmcrImpSampAn.stdDev, confLvl);
        std::cout << "ImpSamp Analytic:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrImpSampAn.energy << " +/- " << vmcrImpSampAn.stdDev
                  << "\tconf. int. with conf. lvl. of " << confLvl << "%: " << confIntImpSampAn.min << " - "
                  << confIntImpSampAn.max << "\n";

        energyValsMetrAn.push_back(vmcrMetrAn.energy.val);
        energyValsImpSampAn.push_back(vmcrImpSampAn.energy.val);
        confIntsMetrAn.push_back(confIntMetrAn);
        confIntsImpSampAn.push_back(confIntImpSampAn);

        // Numeric
        VMCResult<0> const vmcrMetrNum =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, false, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntMetrNum = GetConfInt(vmcrMetrNum.energy, vmcrMetrNum.stdDev, confLvl);
        std::cout << "Metropolis Numeric:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrMetrNum.energy << " +/- " << vmcrMetrNum.stdDev << "\tconf. int. with conf. lvl. of "
                  << confLvl << "%: " << confIntMetrNum.min.val << " - " << confIntMetrNum.max << "\n";

        VMCResult<0> const vmcrImpSampNum =
            VMCEnergy<D, N, 0>(wavefHO, startPoss, ParamBounds<0>{}, true, derivativeStep, mass, potHO,
                               coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
        ConfInterval confIntImpSampNum = GetConfInt(vmcrImpSampNum.energy, vmcrImpSampNum.stdDev, confLvl);
        std::cout << "ImpSamp Numeric:\n"
                  << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
                  << vmcrImpSampNum.energy << " +/- " << vmcrImpSampNum.stdDev
                  << "\tconf. int. with conf. lvl. of " << confLvl << "%: " << confIntImpSampNum.min.val
                  << " - " << confIntImpSampNum.max << "\n\n";

        energyValsMetrNum.push_back(vmcrMetrNum.energy.val);
        energyValsImpSampNum.push_back(vmcrImpSampNum.energy.val);
        confIntsMetrNum.push_back(confIntMetrNum);
        confIntsImpSampNum.push_back(confIntImpSampNum);
    }

    // ONE VARIATIONAL PARAMETERS STRUCTS (Potential does not have a var param so reuse previous struct)
    // Structs with alpha as variational parameter to then find best alpha value via grandient descent
    struct WavefHOVar {
        FPType beta;
        FPType operator()(Positions<D, N> x, VarParams<1> alpha) const {
            Coordinate *begin = &x[0][0];
            Coordinate *end = &x[0][0] + N * D;

            FPType expArg = std::transform_reduce(
                begin, end, FPType{0}, std::plus<>(), [beta_ = beta, &begin](Coordinate &c) {
                    auto const index = &c - begin;
                    bool isLastDimension = ((static_cast<UIntType>(index) + 1u) % D == 0) && (D != 1);
                    return c.val * c.val * (isLastDimension ? beta_ : 1);
                });

            return std::exp(-alpha[0].val * expArg);
        }
    };
    struct FirstDerHOVar {
        FPType beta;
        UIntType dimension;
        UIntType particle;
        FirstDerHOVar(FPType beta_, UIntType dimension_, UIntType particle_)
            : beta{beta_}, dimension{dimension_}, particle{particle_} {
            assert(dimension < D);
            assert(particle < N);
        }
        FirstDerHOVar() : beta{}, dimension{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<1> alpha) const {
            FPType result = -2 * alpha[0].val * x[particle][dimension].val *
                            (((dimension == (D - 1)) && (D != 1)) ? beta : 1) * WavefHOVar{beta}(x, {alpha});
            assert(!std::isnan(result));

            return result;
        }
    };
    struct LaplHOVar {
        FPType beta;
        UIntType particle;

        LaplHOVar(FPType beta_, UIntType particle_) : beta{beta_}, particle{particle_} {
            assert(particle < N);
        }
        LaplHOVar() : beta{}, particle{} {}

        FPType operator()(Positions<D, N> x, VarParams<1> alpha) const {
            UIntType uPar = particle;
            Coordinate *begin = &x[uPar][0];
            Coordinate *end = &x[uPar][0] + D;

            FPType result =
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
                WavefHOVar{beta}(x, {alpha});
            assert(!std::isnan(result));

            return result;
        }
    };

    PotHO potHO{mass, OmegaHO, Gamma};
    WavefHOVar wavefHOVar{Beta};
    Gradients<D, N, FirstDerHOVar> gradsHOVar;
    for (ParticNum n = 0u; n < N; n++) {
        for (Dimension d = 0u; d < D; d++) {
            gradsHOVar[n][d] = FirstDerHOVar{Beta, d, n};
        }
    }

    std::array<LaplHOVar, N> laplHOVar;
    std::generate(laplHOVar.begin(), laplHOVar.end(),
                  [counter = ParticNum{0}]() mutable { return LaplHOVar{Beta, counter++}; });

    // All the following vectors will be filled with a single element as we have only one variational
    // parameters (sciplot requires vectors as inputs)
    std::vector<FPType> varParamsMetrAn;
    std::vector<FPType> varParamsImpSampAn;
    std::vector<FPType> varParamsMetrNum;
    std::vector<FPType> varParamsImpSampNum;

    std::vector<FPType> varEnergiesMetrAn;
    std::vector<FPType> varEnergiesImpSampAn;
    std::vector<FPType> varEnergiesMetrNum;
    std::vector<FPType> varEnergiesImpSampNum;

    vmcp::VarParam const bestParam{ParticlesMass.val * OmegaHO / (2 * hbar)};
    vmcp::ParamBounds<1> const alphaBounds{NiceBound(bestParam, minParamFactor, maxParamFactor, maxParDiff)};

    // Analytic (Dense because the alphavals are few and very close to each other)
    VMCResult<1> const vmcrMetrAnDense =
        VMCEnergy<D, N, 1>(wavefHOVar, startPoss, alphaBounds, laplHOVar, mass, potHO, coordBounds,
                           numEnergies, statFunction, bootstrapSamples, gen);
    std::cout << "Metropolis analytic best alpha: " << std::setprecision(3)
              << vmcrMetrAnDense.bestParams[0].val << std::setprecision(5)
              << "\tenergy: " << vmcrMetrAnDense.energy << " +/- " << vmcrMetrAnDense.stdDev << "\n\n";
    VMCResult<1> const vmcrImpSampAnDense =
        VMCEnergy<D, N, 1>(wavefHOVar, startPoss, alphaBounds, gradsHOVar, laplHOVar, mass, potHO,
                           coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
    std::cout << "ImpSamp analytic best alpha: " << std::setprecision(3)
              << vmcrImpSampAnDense.bestParams[0].val << std::setprecision(5)
              << "\tenergy: " << vmcrImpSampAnDense.energy << " +/- " << vmcrImpSampAnDense.stdDev << "\n\n";

    // Numeric (Dense because the alphavals are few and very close to each other)
    VMCResult<1> const vmcrMetrNumDense =
        VMCEnergy<D, N, 1>(wavefHOVar, startPoss, alphaBounds, false, derivativeStep, mass, potHO,
                           coordBounds, numEnergies, statFunction, bootstrapSamples, gen);
    std::cout << "Metropolis Numeric best alpha: " << std::setprecision(3)
              << vmcrMetrNumDense.bestParams[0].val << std::setprecision(5)
              << "\tenergy: " << vmcrMetrNumDense.energy << " +/- " << vmcrMetrNumDense.stdDev << "\n\n";
    VMCResult<1> const vmcrImpSampNumDense =
        VMCEnergy<D, N, 1>(wavefHOVar, startPoss, alphaBounds, true, derivativeStep, mass, potHO, coordBounds,
                           numEnergies, statFunction, bootstrapSamples, gen);
    std::cout << "ImpSamp Numeric best alpha: " << std::setprecision(3)
              << vmcrImpSampNumDense.bestParams[0].val << std::setprecision(5)
              << "\tenergy: " << vmcrImpSampNumDense.energy << " +/- " << vmcrImpSampNumDense.stdDev
              << "\n\n";

    varParamsMetrAn.push_back(vmcrMetrAnDense.bestParams[0].val);
    varParamsImpSampAn.push_back(vmcrImpSampAnDense.bestParams[0].val);
    varParamsMetrNum.push_back(vmcrMetrNumDense.bestParams[0].val);
    varParamsImpSampNum.push_back(vmcrImpSampNumDense.bestParams[0].val);

    varEnergiesMetrAn.push_back(vmcrMetrAnDense.energy.val);
    varEnergiesImpSampAn.push_back(vmcrImpSampAnDense.energy.val);
    varEnergiesMetrNum.push_back(vmcrMetrNumDense.energy.val);
    varEnergiesImpSampNum.push_back(vmcrImpSampNumDense.energy.val);

    // Plotting the energies with NEW alphaVals:
    // Construct file paths
    std::string folderDense = "./artifacts/NoInt/Dense/D=" + std::to_string(D) + "_N=" + std::to_string(N);

    std::string metrAnFileDense = folderDense + "/plot_Metr_An.pdf";
    std::string impSampAnFileDense = folderDense + "/plot_ImpSamp_An.pdf";
    std::string metrNumFileDense = folderDense + "/plot_Metr_Num.pdf";
    std::string impSampNumFileDense = folderDense + "/plot_ImpSamp_Num.pdf";

    std::filesystem::create_directories(folderDense);

    // Analaytic
    sciplot::Canvas canvasMetrAnDense =
        MakeGraphNoIntVar<D, N>(alphaVals, energyValsMetrAn, confIntsMetrAn, exactEns, varParamsMetrAn,
                                varEnergiesMetrAn, "Metropolis Analytic");
    sciplot::Canvas canvasImpSampAnDense =
        MakeGraphNoIntVar<D, N>(alphaVals, energyValsImpSampAn, confIntsImpSampAn, exactEns,
                                varParamsImpSampAn, varEnergiesImpSampAn, "Imp. Samp. Analytic");

    canvasMetrAnDense.save(metrAnFileDense);
    canvasImpSampAnDense.save(impSampAnFileDense);

    // Numeric
    sciplot::Canvas canvasMetrNumDense =
        MakeGraphNoIntVar<D, N>(alphaVals, energyValsMetrNum, confIntsMetrNum, exactEns, varParamsMetrNum,
                                varEnergiesMetrNum, "Metropolis Numeric");
    sciplot::Canvas canvasImpSampNumDense =
        MakeGraphNoIntVar<D, N>(alphaVals, energyValsImpSampNum, confIntsImpSampNum, exactEns,
                                varParamsImpSampNum, varEnergiesImpSampNum, "Imp. Samp. Numeric");

    canvasMetrNumDense.save(metrNumFileDense);
    canvasImpSampNumDense.save(impSampNumFileDense);
}

// Interacting Harmonic Oscillator
template <vmcp::Dimension D, vmcp::ParticNum N>
void HOInt(vmcp::StatFuncType statFunction, std::vector<vmcp::FPType> &energyVals,
           std::vector<vmcp::ConfInterval> &confInts, std::vector<vmcp::ParticNum> &NVals) {
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

            FPType result =
                WavefHO{beta, a}(x, {alpha}) * ((nonIntLapl + 2 * innerProd) / phiK + pureIntTerms);
            assert(!std::isnan(result));

            return result;
        }
    };

    Masses<N> mass;
    mass.fill(ParticlesMass);

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
    // Division by N in order to plot Energy / # of particles
    ConfInterval confInt =
        GetConfInt(Energy{vmcrBest.energy.val / N}, Energy{vmcrBest.stdDev.val / N}, confLvl);
    std::cout << "Energy with the best alpha for N=" + std::to_string(N) + ":\n"
              << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val / N
              << " +/- " << vmcrBest.stdDev.val / N << "\tconf. int. with conf. lvl. of " << confLvl
              << "%: " << confInt.min << " - " << confInt.max << '\n';

    energyVals.push_back(vmcrBest.energy.val / N);
    confInts.push_back(confInt);
    NVals.push_back(N);
}

// In order to avoid producing every graph (and thus having the largest execution time),
// we suggest to comment some functions inside main() before compiling.
// E.g. //HONoInt<3, 5>(statFunction); to avoid producing non-interaction graphs.
// Additionaly, commenting single or multiple HOInt function removes only those selected points from
// the produced graph (obviously the higher the number of particles, the longer the execution time will be)
int main() {
    vmcp::StatFuncType statFunction = vmcp::StatFuncType::bootstrap;

    // Non interacting plots of energy vs (non variational) parameter alpha
    HONoInt<3, 5>(statFunction);

    // Interacting plots of energy/N vs N (i.e. the number of particles) using alpha as a
    // variational parameter
    std::vector<vmcp::FPType> energyVals;
    std::vector<vmcp::ConfInterval> confInts;
    std::vector<vmcp::ParticNum> NVals;

    HOInt<3, 2>(statFunction, energyVals, confInts, NVals);
    HOInt<3, 3>(statFunction, energyVals, confInts, NVals);
    HOInt<3, 4>(statFunction, energyVals, confInts, NVals);
    HOInt<3, 5>(statFunction, energyVals, confInts, NVals);
    HOInt<3, 6>(statFunction, energyVals, confInts, NVals);
    HOInt<3, 7>(statFunction, energyVals, confInts, NVals);

    vmcp::DrawGraphInt(energyVals, confInts, NVals);
}
