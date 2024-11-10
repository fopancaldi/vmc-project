//
//
// Contains the definition of the templates declared in vmcalgs.hpp
// This file is supposed to be #included at the end of vmcalgs.hpp and nowhere else
// It is just a way to improve the readability of vmcalgs.hpp
//

// The functions that have a trailing underscore would, in a regular project, be (declared and) defined in a
// .cpp file, therefore making them unreachable for the user
// Since they are templated, they must be defined in a header

#ifndef VMCPROJECT_VMCALGS_INL
#define VMCPROJECT_VMCALGS_INL

#include "vmcalgs.hpp"

#include <algorithm>
#include <cassert>

namespace vmcp {

// Constants
// Their name is before the underscore
// After the underscore is the function(s) that use them
constexpr FPType hbar = 1; // Natural units
constexpr IntType numSamplesForPeakSearch = 100;
// LF TODO: Rename this one
// Also, is there a better criterion to choose this than to simply fix it (which is a bad idea since it might
// turn out to be too large for some programs)?
constexpr FPType deltaT = 0.005f;
// FP TODO: Rename this one
constexpr IntType boundSteps = 100;
constexpr IntType thermalizationMoves = 100;
// FP TODO: Rename this one, also find if it is really 10 * thermalization moves
constexpr IntType movesForgetICs = 10 * thermalizationMoves;
constexpr FPType minPsi_peakSearch = 1e-6f;
constexpr IntType maxLoops_gradDesc = 100;
constexpr IntType stepDenom_gradDesc = 100;
constexpr FPType stoppingThreshold_gradDesc = FPType{1e-6f};

template <Dimension D, ParticNum N>
std::vector<Energy> Energies_(std::vector<EnAndPos<D, N>> const &eps) {
    std::vector<Energy> result;
    result.reserve(eps.size());
    std::transform(eps.begin(), eps.end(), std::back_inserter(result),
                   [](EnAndPos<D, N> const &ep) { return ep.energy; });
    return result;
}

// FP/LF TODO: Not really a fan of the name, though surely it is better than AddTo_
// Computes the position of the particles when the n-th one is moved by delta along the d-th direction (with
// both n and d starting from 0)
template <Dimension D, ParticNum N>
Positions<D, N> PerturbPos_(Positions<D, N> const &poss, Dimension d, ParticNum n, Coordinate delta) {
    assert(d < D);
    assert(n < N);
    Positions<D, N> result = poss;
    // TODO: Maybe define operator+= for coordinates?
    result[n][d].val += delta.val;
    return result;
}

// Computes the (approximate) peak of the potential
// In practice, choose some points in the integration domain and see where the potential is largest
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Positions<D, N> FindPeak_(Wavefunction const &psi, VarParams<V> params, Potential const &pot,
                          Bounds<D> bounds, IntType points, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());

    Position<D> center;
    std::transform(bounds.begin(), bounds.end(), center.begin(),
                   [](Bound b) { return Coordinate{(b.upper + b.lower) / 2}; });
    Positions<D, N> result;
    std::fill(result.begin(), result.end(), center);
    std::uniform_real_distribution<FPType> unif(0, 1);
    for (IntType i = 0; i != points; ++i) {
        Positions<D, N> newPoss;
        // FP TODO: Is it really necessary to specify Position<D>?
        // In general, study class template argument deduction
        for (Position<D> &p : newPoss) {
            std::transform(bounds.begin(), bounds.end(), p.begin(), [&unif, &gen](Bound b) {
                return Coordinate{b.lower + unif(gen) * (b.upper - b.lower)};
            });
        }
        // The requirement ... > minPsi avoids having psi(...) = nan in the future, which breaks the update
        // algorithms
        if ((pot(newPoss) > pot(result)) && (psi(newPoss, params) > minPsi_peakSearch)) {
            result = newPoss;
        }
    }
    return result;
}

// Updates the wavefunction with the Metropolis algorithms and outputs the number of succesful updates
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
UIntType MetropolisUpdate_(Wavefunction const &psi, VarParams<V> params, Positions<D, N> &poss, FPType step,
                           RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    UIntType succesfulUpdates = 0u;
    for (Position<D> &p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        std::uniform_real_distribution<FPType> unifDist(0, 1);
        std::transform(p.begin(), p.end(), p.begin(), [&gen, &unifDist, step](Coordinate c) {
            return Coordinate{c.val + (unifDist(gen) - FPType{0.5f}) * step};
        });
        if (unifDist(gen) > std::pow(psi(poss, params) / oldPsi, 2)) {
            p = oldPos;

        } else {
            ++succesfulUpdates;
        }
    }
    return succesfulUpdates;
}

// Computes the drift force by using its analytic expression
template <Dimension D, ParticNum N, VarParNum V, class FirstDerivative, class Wavefunction>
std::array<FPType, D> DriftForceAnalytic_(Wavefunction const &psi, Positions<D, N> poss, VarParams<V> params,
                                          std::array<FirstDerivative, D> const &grad) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    std::array<FPType, D> gradVal;
    std::transform(grad.begin(), grad.end(), gradVal.begin(),
                   [&poss, params](FirstDerivative const &fd) { return fd(poss, params); });
    std::array<FPType, D> result;
    std::transform(gradVal.begin(), gradVal.end(), result.begin(),
                   [&psi, &poss, params](FPType f) { return 2 * f / psi(poss, params); });
    return result;
}

// Computes the drift force by numerically estimating the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
std::array<FPType, D> DriftForceNumeric_(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep,
                                         Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    std::array<FPType, D> driftForce;
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            driftForce[d] = 2 *
                            (psi(PerturbPos_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                             psi(PerturbPos_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                            (derivativeStep * psi(poss, params));
        }
    }
}

// LF TODO: Tell that this functions applies formula at page 22 Jensen
//  Updates the wavefunction with the importance sampling algorithm and outputs the number of succesful
//  updates
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative>
IntType ImportanceSamplingUpdate_(Wavefunction const &psi, VarParams<V> params,
                                  std::array<FirstDerivative, D> const &grad, Mass mass,
                                  Positions<D, N> &poss, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());

    FPType const diffusionConst = hbar * hbar / (2 * mass.val);

    IntType successfulUpdates = 0;
    for (Position<D> &p : poss) {
        Position const oldPos = p;
        FPType const oldPsi = psi(poss, params);
        std::array<FPType, D> const oldDriftForce = DriftForceAnalytic_<D, N, V>(psi, poss, params, grad);
        std::normal_distribution<FPType> normal(0.f, diffusionConst * deltaT);
        for (Dimension d = 0u; d != D; ++d) {
            p[d].val = oldPos[d].val + diffusionConst * deltaT * (oldDriftForce[d] + normal(gen));
        }
        FPType const newPsi = psi(poss, params);
        std::array<FPType, D> newDriftForce = DriftForceAnalytic_<D, N, V>(psi, poss, params, grad);
        FPType forwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            forwardExponent -=
                std::pow(p[d].val - oldPos[d].val - diffusionConst * deltaT * oldDriftForce[d], 2) /
                (4 * diffusionConst * deltaT);
        }
        FPType const forwardProb = std::exp(forwardExponent);
        FPType backwardExponent = 0;
        for (Dimension d = 0u; d != D; ++d) {
            backwardExponent -=
                std::pow(oldPos[d].val - p[d].val - diffusionConst * deltaT * newDriftForce[d], 2) /
                (4 * diffusionConst * deltaT);
        }
        FPType const backwardProb = std::exp(backwardExponent);
        FPType const acceptanceRatio = (newPsi * newPsi * backwardProb) / (oldPsi * oldPsi * forwardProb);
        std::uniform_real_distribution<FPType> unifDist(0.f, 1.f);
        if (unifDist(gen) < acceptanceRatio) {
            ++successfulUpdates;
        } else {
            p = oldPos;
        }
    }
    return successfulUpdates;
}

// Computes the local energy with the analytic formula for the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class SecondDerivative, class Potential>
Energy LocalEnergyAnalytic_(Wavefunction const &psi, VarParams<V> params, SecondDerivative const &secondDer,
                            Mass mass, Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, SecondDerivative>());
    static_assert(IsPotential<D, N, Potential>());

    return Energy{-(hbar * hbar / (2 * mass.val)) * (secondDer(poss, params) / psi(poss, params)) +
                  pot(poss)};
}

// Computes the local energy by numerically estimating the derivative of the wavefunction
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
Energy LocalEnergyNumeric_(Wavefunction const &psi, VarParams<V> params, FPType derivativeStep, Mass mass,
                           Potential const &pot, Positions<D, N> poss) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsPotential<D, N, Potential>());

    Energy result{pot(poss)};
    // TODO: Maybe redo this without the indices? Open to suggestions
    for (ParticNum n = 0u; n != N; ++n) {
        for (Dimension d = 0u; d != D; ++d) {
            result.val += -std::pow(hbar, 2) / (2 * mass.val) *
                          (psi(PerturbPos_<D, N>(poss, d, n, Coordinate{derivativeStep}), params) -
                           2 * psi(poss, params) +
                           psi(PerturbPos_<D, N>(poss, d, n, Coordinate{-derivativeStep}), params)) /
                          std::pow(derivativeStep, 2);
        }
    }
    return result;
}

// Computes the energies that will be averaged to obtain the estimate of the GS energy of the VMC algorithm
// by either using the analytical formula for the derivative or estimating it numerically
// Use the wrappers to select which method should be used
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
std::vector<EnAndPos<D, N>>
WrappedVMCEnAndPoss_(Wavefunction const &psi, VarParams<V> params, bool useAnalitycal, bool useImpSamp,
                     std::array<FirstDerivative, D> const &grad, SecondDerivative const &secondDer,
                     FPType derivativeStep, Mass mass, Potential const &pot, Bounds<D> bounds,
                     int numberEnergies, RandomGenerator &gen) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(IsWavefunctionDerivative<D, N, V, FirstDerivative>());
    static_assert(IsWavefunctionDerivative<D, N, V, SecondDerivative>());
    static_assert(IsPotential<D, N, Potential>());
    assert(numberEnergies >= 0);

    std::vector<EnAndPos<D, N>> result;

    // Step 1: Find a good starting point, in the sense that it's easy to move away from
    Positions<D, N> const peak = FindPeak_<D, N>(psi, params, pot, bounds, numSamplesForPeakSearch, gen);

    // Step 2: Choose the step
    Bound const smallestBound = *(std::min_element(
        bounds.begin(), bounds.end(), [](Bound b1, Bound b2) { return b1.Length() < b2.Length(); }));
    FPType const step = smallestBound.Length() / boundSteps;

    // Step 3: Save the energies to be averaged
    // FP TODO: Rename useAnalitycal everywhere
    Positions<D, N> poss = peak;
    auto const LocalEnergy{[&]() {
        if (useAnalitycal) {
            return LocalEnergyAnalytic_<D, N>(psi, params, secondDer, mass, pot, poss);
        } else {
            return LocalEnergyNumeric_<D, N>(psi, params, derivativeStep, mass, pot, poss);
        }
    }};
    auto const Update{[&]() {
        if (useImpSamp) {
            ImportanceSamplingUpdate_<D, N>(psi, params, grad, mass, poss, gen);
        } else {
            MetropolisUpdate_<D, N>(psi, params, poss, step, gen);
        }
    }};

    // Move away from the peak, in order to forget the dependence on the initial conditions
    for (int i = 0; i != movesForgetICs; ++i) {
        Update();
    }
    for (int i = 0; i != numberEnergies; ++i) {
        for (int j = 0; j != thermalizationMoves; ++j) {
            Update();
        }
        result.emplace_back(LocalEnergy(), poss);
    }

    // FP TODO: Step 4: Adjust the step size

    return result;
}

// FP TODO: Move into a .cpp
VMCResult AvgAndVar_(std::vector<Energy> const &v) {
    assert(v.size() > 1);
    auto const size = v.size();
    Energy const avg{std::accumulate(v.begin(), v.end(), Energy{0},
                                     [](Energy e1, Energy e2) { return Energy{e1.val + e2.val}; })
                         .val /
                     size};
    EnVariance const var{std::accumulate(v.begin(), v.end(), EnVariance{0},
                                         [avg](EnVariance ev, Energy e) {
                                             return EnVariance{ev.val + std::pow(e.val - avg.val, 2)};
                                         })
                             .val /
                         (size - 1)};
    return VMCResult{avg, var};
}

// FP TODO: Explain better
// Computes the energies after moving in the cardinal directions in parameter space
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction>
std::array<Energy, D> ReweightedEnergies_(Wavefunction const &psi, VarParams<V> oldParams,
                                          std::vector<EnAndPos<N, D>> oldEnAndPoss, FPType step) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());

    std::array<Energy, V> result;
    for (UIntType v = 0u; v != V; ++v) {
        VarParams<V> newParams = oldParams;
        newParams[v].val += step;
        std::vector<Energy> reweightedLocalEnergies;
        std::transform(oldEnAndPoss.begin(), oldEnAndPoss.end(), std::back_inserter(reweightedLocalEnergies),
                       [&psi, newParams, oldParams](EnAndPos<D, N> const &ep) {
                           return Energy{
                               std::pow(psi(ep.positions, newParams) / psi(ep.positions, oldParams), 2) *
                               ep.energy.val};
                       });

        FPType const numerator =
            std::accumulate(reweightedLocalEnergies.begin(), reweightedLocalEnergies.end(), FPType{0},
                            [](FPType f, Energy e) { return f + e.val; });
        FPType const denominator = std::accumulate(
            oldEnAndPoss.begin(), oldEnAndPoss.end(), FPType{0},
            [&psi, newParams, oldParams](FPType f, EnAndPos<D, N> const &ep) {
                return f + std::pow(psi(ep.positions, newParams) / psi(ep.positions, oldParams), 2);
            });

        result[v] = Energy{numerator / denominator};
    }
    return result;
}

// Calculates the parameters of the wavefunction that minimize the energy
template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class EnAndPossCalculator>
VarParams<V> BestParameters_(VarParams<V> initialParams, Wavefunction const &psi,
                             EnAndPossCalculator const &epCalc) {
    static_assert(IsWavefunction<D, N, V, Wavefunction>());
    static_assert(std::is_invocable_r_v<std::vector<EnAndPos<D, N>>, EnAndPossCalculator, VarParams<V>>);
    static_assert(V != UIntType{0});

    // Use the gradiest descent with momentum algorithm, with termination condition: stop if the new
    // parameters are too close to the old ones
    VarParams<V> result = initialParams;
    FPType const initialParamsNorm = std::sqrt(
        std::inner_product(initialParams.begin(), initialParams.end(), initialParams.begin(), FPType{0},
                           std::plus<>(), [](VarParam v1, VarParam v2) { return v1.val * v2.val; }));
    FPType gradientStep = (initialParamsNorm / stepDenom_gradDesc) * 25;

    for (IntType i = 0; i != maxLoops_gradDesc; ++i) {
        std::vector<EnAndPos<D, N>> const currentEnAndPoss = epCalc(result);
        Energy const currentEnergy = AvgAndVar_(Energies_(currentEnAndPoss)).energy;

        std::array<Energy, V> energiesIncreasedParam =
            ReweightedEnergies_<D, N, V>(psi, result, currentEnAndPoss, gradientStep);
        std::array<Energy, V> energiesDecreasedParam =
            ReweightedEnergies_<D, N, V>(psi, result, currentEnAndPoss, -gradientStep);
        std::array<FPType, V> gradient;
        for (VarParNum v = 0u; v != V; ++v) {
            gradient[v] =
                (energiesIncreasedParam[v].val - energiesDecreasedParam[v].val) / (2 * gradientStep);
        }

        // Try smaller and smaller steps until the new paraterers reslt in a smaller energy
        FPType gradMultiplier = 2;
        VarParams<V> newParams;
        do {
            gradMultiplier /= 2;
            for (VarParNum v = 0u; v != V; ++v) {
                newParams[v].val = result[v].val - gradMultiplier * gradient[v];
            }

        } while (AvgAndVar_(Energies_(epCalc(newParams))).energy.val > currentEnergy.val);

        // Set the next gradient step to the size of this step
        FPType const oldParamsNorm = std::sqrt(std::accumulate(
            result.begin(), result.end(), FPType{0}, [](FPType f, VarParam v) { return f + v.val * v.val; }));
        gradientStep = std::sqrt(
            std::inner_product(newParams.begin(), newParams.end(), result.begin(), FPType{0}, std::plus<>(),
                               [](VarParam v1, VarParam v2) { return std::pow(v1.val - v2.val, 2); }));
        result = newParams;

        if ((gradientStep / oldParamsNorm) < stoppingThreshold_gradDesc) {
            break;
        }
    }

    return result;
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class SecondDerivative, class Potential>
std::vector<EnAndPos<D, N>> VMCEnAndPoss(Wavefunction const &psi, VarParams<V> params,
                                         SecondDerivative const &secondDer, Mass mass, Potential const &pot,
                                         Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    struct FakeDeriv {
        FPType operator()(Positions<D, N> const &, VarParams<V>) const {
            assert(false);
            return FPType{0};
        }
    };
    std::array<FakeDeriv, D> fakeGrad;
    std::fill(fakeGrad.begin(), fakeGrad.end(), FakeDeriv{});
    FPType const fakeStep = std::numeric_limits<FPType>::quiet_NaN();
    return WrappedVMCEnAndPoss_<D, N, V>(psi, params, true, false, fakeGrad, secondDer, fakeStep, mass, pot,
                                         bounds, numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class SecondDerivative, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> initialParams, SecondDerivative const &secondDer,
                    Mass mass, Potential const &pot, Bounds<D> bounds, int numberEnergies,
                    RandomGenerator &gen) {
    if constexpr (V != UIntType{0}) {
        auto const enPossCalculator{[&](VarParams<V> params) {
            return VMCEnAndPoss<D, N, V>(psi, params, secondDer, mass, pot, bounds, numberEnergies, gen);
        }};
        VarParams<V> const bestParams = BestParameters_<D, N, V>(initialParams, psi, enPossCalculator);

        return AvgAndVar_(Energies_(
            VMCEnAndPoss<D, N, V>(psi, bestParams, secondDer, mass, pot, bounds, numberEnergies, gen)));
    } else {
        VarParams<0> const fakeParams{};
        return AvgAndVar_(Energies_(
            VMCEnAndPoss<D, N, V>(psi, fakeParams, secondDer, mass, pot, bounds, numberEnergies, gen)));
    }
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
std::vector<EnAndPos<D, N>> VMCEnAndPoss(Wavefunction const &psi, VarParams<V> params,
                                         std::array<FirstDerivative, D> const &grad,
                                         SecondDerivative const &secondDer, Mass mass, Potential const &pot,
                                         Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    FPType const fakeStep = std::numeric_limits<FPType>::quiet_NaN();
    return WrappedVMCEnAndPoss_<D, N, V>(psi, params, true, false, grad, secondDer, fakeStep, mass, pot,
                                         bounds, numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class FirstDerivative,
          class SecondDerivative, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> initialParams,
                    std::array<FirstDerivative, D> const &grad, SecondDerivative const &secondDer, Mass mass,
                    Potential const &pot, Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    if constexpr (V != UIntType{0}) {
        auto const enPossCalculator{[&](VarParams<V> params) {
            return VMCEnAndPoss<D, N, V>(psi, params, grad, secondDer, mass, pot, bounds, numberEnergies,
                                         gen);
        }};
        VarParams<V> const bestParams = BestParameters_<D, N, V>(initialParams, psi, enPossCalculator);

        return AvgAndVar_(Energies_(
            VMCEnAndPoss<D, N, V>(psi, bestParams, grad, secondDer, mass, pot, bounds, numberEnergies, gen)));
    } else {
        VarParams<0> const fakeParams{};
        return AvgAndVar_(Energies_(
            VMCEnAndPoss<D, N, V>(psi, fakeParams, grad, secondDer, mass, pot, bounds, numberEnergies, gen)));
    }
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
std::vector<EnAndPos<D, N>> VMCEnAndPoss(Wavefunction const &psi, VarParams<V> params, bool useImpSamp,
                                         FPType derivativeStep, Mass mass, Potential const &pot,
                                         Bounds<D> bounds, int numberEnergies, RandomGenerator &gen) {
    struct FakeDeriv {
        FPType operator()(Positions<D, N> const &, VarParams<V>) const {
            assert(false);
            return FPType{0};
        }
    };
    std::array<FakeDeriv, D> fakeGrad;
    std::fill(fakeGrad.begin(), fakeGrad.end(), FakeDeriv{});
    return WrappedVMCEnAndPoss_<D, N, V>(psi, params, false, useImpSamp, fakeGrad, FakeDeriv{},
                                         derivativeStep, mass, pot, bounds, numberEnergies, gen);
}

template <Dimension D, ParticNum N, VarParNum V, class Wavefunction, class Potential>
VMCResult VMCEnergy(Wavefunction const &psi, VarParams<V> initialParams, bool useImpSamp,
                    FPType derivativeStep, Mass mass, Potential const &pot, Bounds<D> bounds,
                    int numberEnergies, RandomGenerator &gen) {
    if constexpr (V != UIntType{0}) {
        auto const enPossCalculator{[&](VarParams<V> params) {
            return VMCEnAndPoss<D, N, V>(psi, params, useImpSamp, derivativeStep, mass, pot, bounds,
                                         numberEnergies, gen);
        }};
        VarParams<V> const bestParams = BestParameters_<D, N, V>(initialParams, psi, enPossCalculator);

        return AvgAndVar_(Energies_(VMCEnAndPoss<D, N, V>(psi, bestParams, useImpSamp, derivativeStep, mass,
                                                          pot, bounds, numberEnergies, gen)));
    } else {
        VarParams<0> const fakeParams{};
        return AvgAndVar_(Energies_(VMCEnAndPoss<D, N, V>(psi, fakeParams, useImpSamp, derivativeStep, mass,
                                                          pot, bounds, numberEnergies, gen)));
    }
}

} // namespace vmcp

#endif
