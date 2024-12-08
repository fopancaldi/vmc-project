#include "src/vmcp.hpp"

#include "sciplot/sciplot.hpp"

#include <iomanip>
#include <iostream>
#include <numbers>

using namespace vmcp;

int main() {
    RandomGenerator gen{(std::random_device())()};
    vmcp::IntType const numSamples = 10000;

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
        VMCEnergy<1, 1, 1>(wavefHO, alphaBounds, laplHO, mass, potHO, coorBounds, numberEnergies,
                           vmcp::StatFuncType::regular, numSamples, gen);
    std::cout << "Energy with the best alpha:\n"
              << std::setprecision(3) << "Energy: " << std::setprecision(5) << vmcrBest.energy.val << " +/- "
              << vmcrBest.stdDev.val << '\n';

    // No variational parameters, plot as a fucntion of alpha
    std::vector<FPType> alphaVals;
    std::generate_n(std::back_inserter(alphaVals), 40,
                    [i = 0]() mutable -> FPType { return FPType{0.1f} + static_cast<FPType>(++i) * 0.05f; });
    std::vector<FPType> energyVals;
    std::vector<FPType> errorVals;
    for (FPType alphaVal : alphaVals) {
        std::vector<vmcp::LocEnAndPoss<1, 1>> vmcLEPs = VMCLocEnAndPoss<1, 1, 1>(
            wavefHO, VarParams<1>{alphaVal}, laplHO, mass, potHO, coorBounds, numberEnergies, gen);
        Energy const vmcEnergy = Mean(vmcLEPs);
        Energy const vmcStdDev = StdDev(vmcLEPs);
        std::cout << "alpha: " << std::setprecision(3) << alphaVal << "\tenergy: " << std::setprecision(5)
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
    canvas.save("artifacts/plot.pdf");
}
