#include "src/vmcp.hpp"

#include "sciplot/sciplot.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace vmcp;

// The various features of main can be toggled here
constexpr std::array features = {true, false};

int main() {
    RandomGenerator gen{(std::random_device())()};

    // Feature 1:
    // Various tests with the harmonic oscillator
    if constexpr (features[0]) { /*
         auto const wavefHO{[](Positions<1, 1> x, VarParams<1> alpha) {
             return std::pow(std::numbers::e_v<FPType>, -alpha[0].val * x[0][0].val * x[0][0].val / 2);
         }};
         auto const potHO{[](Positions<1, 1> x) { return x[0][0].val * x[0][0].val; }};
         std::array laplHO{[&wavefHO](Positions<1, 1> x, VarParams<1> alpha) {
             return (std::pow(alpha[0].val * x[0][0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
         }};
         IntType const numberEnergies = 100;
         CoordBounds<1> const coorBounds = {Bound{Coordinate{-100}, Coordinate{100}}};
         Masses<1> const mass{Mass{0.5f}};

         // One variational parameter
         ParamBounds<1> const alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};
         VMCResult const vmcrBest =
             VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, laplHO, mass, potHO, coorBounds, numberEnergies, gen);
         std::cout << "Energy with the best alpha:\n"
                   << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                   << " +/- " << vmcrBest.stdDev.val << '\n';

         // No variational parameters, plot as a fucntion of alpha
         std::vector<FPType> alphaVals;
         std::generate_n(std::back_inserter(alphaVals), 40, [i = 0]() mutable -> FPType {
             return FPType{0.1f} + static_cast<FPType>(++i) * 0.05f;
         });
         std::vector<FPType> energyVals;
         std::vector<FPType> errorVals;
         for (FPType alphaVal : alphaVals) {
             std::vector<vmcp::LocEnAndPoss<1, 1>> vmcLEPs = VMCLocEnAndPoss<1, 1, 1>(
                 wavefHO, VarParams<1>{alphaVal}, laplHO, mass, potHO, coorBounds, numberEnergies, gen);
             Energy const vmcEnergy = Mean(vmcLEPs);
             Energy const vmcStdDev = StdDev(vmcLEPs);
             std::cout << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " <<
         std::setprecision(5)
                       << vmcEnergy.val << " +/- " << vmcStdDev.val << '\n';
             energyVals.push_back(vmcEnergy.val);
             errorVals.push_back(5 * vmcStdDev.val);
         }

         // Important stuff
         // lineColor requires hexadecimal plus transparency (first 2 digits)
         sciplot::Plot2D plot;
         plot.drawCurveWithErrorBarsY(alphaVals, energyVals, errorVals)
             .lineColor("#00ff0000")
             .label("harmonic oscillator");

         // Cosmetics
         plot.legend().atTop();
         plot.xlabel("alpha");
         plot.ylabel("vmc energy");
         plot.border();

         // Hos to save
         sciplot::Figure fig = {{plot}};
         sciplot::Canvas canvas = {{fig}};
         canvas.save("artifacts/plot.pdf");*/

        Masses<2> const mInit{1.f, 5.f};
        std::array<vmcp::FPType, 2> const omegaInit{1.f, 5.f};

        // Feature 1:
        // Various tests with the harmonic oscillator
        if constexpr (features[0]) {
            struct PotHO {
                Masses<2> m;
                std::array<vmcp::FPType, 2> omega;
                vmcp::FPType operator()(vmcp::Positions<1, 2> x) const {
                    return x[0][0].val * x[0][0].val * (m[0].val * omega[0] * omega[0] / 2) +
                           x[1][0].val * x[1][0].val * (m[1].val * omega[1] * omega[1] / 2);
                }
            };

            PotHO potHO{mInit, omegaInit};

            auto const wavefHO{[](vmcp::Positions<1, 2> x, vmcp::VarParams<1> alpha) {
                return std::exp(-alpha[0].val * std::pow(x[0][0].val, 2)) *
                       std::exp(-alpha[1].val * std::pow(x[1][0].val, 2));
            }};
            struct LaplHO {
                vmcp::IntType particle;
                LaplHO(vmcp::IntType particle_) : particle{particle_} {
                    assert(particle >= 0);
                    assert(particle <= 1);
                }
                vmcp::FPType operator()(vmcp::Positions<1, 2> x, vmcp::VarParams<1> alpha) const {
                    vmcp::UIntType uPar = static_cast<vmcp::UIntType>(particle);
                    return (std::pow(x[uPar][0].val * alpha[uPar].val, 2) - 2 * alpha[uPar].val) *
                           std::exp(-alpha[0].val * std::pow(x[0][0].val, 2)) *
                           std::exp(-alpha[0].val * std::pow(x[1][0].val, 2));
                }
            };

            IntType const numberEnergies = 100;
            CoordBounds<1> const coorBounds = {Bound{Coordinate{-100}, Coordinate{100}}};
            Masses<2> const mass{Mass{0.5f}, Mass{1.f}};

            vmcp::Laplacians<2, LaplHO> laplHO{LaplHO{0}, LaplHO{1}};

            // Two variational parameters
            ParamBounds<1> const alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

            VMCResult const vmcrBest = VMCEnergy<1, 2, 1>(wavefHO, alphaBounds, laplHO, mass, potHO,
                                                          coorBounds, numberEnergies, gen);
            std::cout << "Energy with the best alpha:\n"
                      << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                      << " +/- " << vmcrBest.stdDev.val << '\n';

            // No variational parameters, plot as a fucntion of alpha
            std::vector<FPType> alphaVals;
            std::generate_n(std::back_inserter(alphaVals), 40, [i = 0]() mutable -> FPType {
                return FPType{0.1f} + static_cast<FPType>(++i) * 0.05f;
            });
            std::vector<FPType> energyVals;
            std::vector<FPType> errorVals;
            for (FPType alphaVal : alphaVals) {
                std::vector<vmcp::LocEnAndPoss<1, 2>> vmcLEPs = VMCLocEnAndPoss<1, 2, 1>(
                    wavefHO, VarParams<1>{alphaVal}, laplHO, mass, potHO, coorBounds, numberEnergies, gen);
                Energy const vmcEnergy = Mean(vmcLEPs);
                Energy const vmcStdDev = StdDev(vmcLEPs);
                std::cout << "alpha: " << std::setprecision(3) << alphaVal
                          << "\tenergy: " << std::setprecision(5) << vmcEnergy.val << " +/- " << vmcStdDev.val
                          << '\n';
                energyVals.push_back(vmcEnergy.val);
                errorVals.push_back(5 * vmcStdDev.val);
            }

            // Important stuff
            // lineColor requires hexadecimal plus transparency (first 2 digits)
            sciplot::Plot2D plot;

            plot.drawCurve(alphaVals, energyVals)
                .lineColor("#00ff0000") // Curve color (red)
                .label("harmonic oscillator");

            plot.drawErrorBarsY(alphaVals, energyVals, errorVals)
                .lineColor("#0000ff") // Error bar color (blue)
                .lineWidth(1)
                .label("");

            // Cosmetics
            plot.legend().atTop();
            plot.xlabel("alpha");
            plot.ylabel("vmc energy");
            plot.border();

            // Hos to save
            sciplot::Figure fig = {{plot}};
            sciplot::Canvas canvas = {{fig}};
            canvas.save("artifacts/plot.pdf");
        }
    }

    // Feature 2:
    // Various tests with the harmonic oscillator
    if constexpr (features[1]) {
        // It is not normalized, but it doesn't matter
        auto const wavefHO{[](Positions<1, 1> x, VarParams<1> alpha) {
            return std::pow(std::numbers::e_v<FPType>, -alpha[0].val * x[0][0].val * x[0][0].val / 2);
        }};
        auto const potHO{[](Positions<1, 1> x) { return x[0][0].val * x[0][0].val; }};
        auto const secondDerHO{[&wavefHO](Positions<1, 1> x, VarParams<1> alpha) {
            return (std::pow(alpha[0].val * x[0][0].val, 2) - alpha[0].val) * wavefHO(x, alpha);
        }};

        IntType const numberEnergies = 1 << 10;
        CoordBounds<1> const coorBounds = {Bound{Coordinate{-100}, Coordinate{100}}};
        Mass const mass{0.5f};

        // Number of samples for bootsrapping
        IntType const numSamples = 1000;

        // Level of confidence for confidence interval
        FPType const confLevel = 95.f;

        for (VarParam alpha{0.1f}; alpha.val <= 2; alpha.val += FPType{0.05f}) {
            std::vector<LocEnAndPoss<1, 1>> const locEnPos =
                VMCLocEnAndPoss<1, 1, 1>(wavefHO, VarParams<1>{alpha}, std::array{secondDerHO},
                                         std::array{mass}, potHO, coorBounds, numberEnergies, gen);

            PartialVMCResult const vmcr1 = Statistics(locEnPos, StatFuncType::blocking, numSamples, gen);
            PartialVMCResult const vmcr2 = Statistics(locEnPos, StatFuncType::bootstrap, numSamples, gen);

            ConfInterval confInt1 = GetConfInt(vmcr1.energy, vmcr1.stdDev, confLevel);
            ConfInterval confInt2 = GetConfInt(vmcr2.energy, vmcr2.stdDev, confLevel);

            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr1.energy.val << " +/- "
                      << vmcr1.stdDev.val << "\tconf. interval of " << confLevel
                      << "%: " << std::setprecision(3) << confInt1.min.val << " -- " << confInt1.max.val
                      << '\n';
            std::cout << "alpha: " << std::setprecision(3) << alpha.val
                      << "\tenergy: " << std::setprecision(5) << vmcr2.energy.val << " +/- "
                      << vmcr2.stdDev.val << "\tconf. interval of " << confLevel
                      << "%: " << std::setprecision(3) << confInt2.min.val << " -- " << confInt2.max.val
                      << '\n';
        }

        ParamBounds<1> alphaBounds{Bound{VarParam{0.5f}, VarParam{1.5f}}};

        VMCResult const vmcrBest =
            VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, std::array{secondDerHO}, std::array{mass}, potHO,
                               coorBounds, numberEnergies, gen);
        std::cout << "Energy with the best alpha:\n"
                  << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val
                  << " +/- " << vmcrBest.stdDev.val << '\n';
    }
}
