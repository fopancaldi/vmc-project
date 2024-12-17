
//!
//! @file main.hpp
//! @brief Helper functions for main
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the helper functions used in main.cpp
//! @see main.cpp
//!

#ifndef VMCPROJECT_MAIN_HPP
#define VMCPROJECT_MAIN_HPP

#include "sciplot/sciplot.hpp"
#include "src/vmcp.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <queue>
#include <set>

namespace vmcp {

//! @defgroup Helper functions for computing the energies
//! @brief Helper methods used for performing Monte Carlo analysis
//! @{

//! @brief Avoids manually repeating the Bound initialization by generating directly an array of bounds of
//! dimension D
//! @param lower The lower coordinate bound
//! @param upper The upper coordinate bound
//! @return An array of D bounds where all the bounds are the same
template <Dimension D>
CoordBounds<D> MakeCoordBounds(Coordinate lower, Coordinate upper) {
    assert(lower.val < upper.val);
    CoordBounds<D> coordBounds;
    std::fill(coordBounds.begin(), coordBounds.end(), Bound{lower, upper});
    assert(coordBounds.size() == D);
    return coordBounds;
}

//! @brief Computes Eucledian distance between two particles
//! @param x The first particle
//! @param y The second particle
//! @return The distance
template <Dimension D>
FPType Distance(Position<D> x, Position<D> y) {
    FPType sqrdDist =
        std::inner_product(x.begin(), x.end(), y.begin(), FPType{0}, std::plus<>(),
                           [](Coordinate &a, Coordinate &b) { return (a.val - b.val) * (a.val - b.val); });
    return std::sqrt(sqrdDist);
}

//! @brief Creates an initial position by placing particles in a face-centered cubic lattice
//! @param bounds The region in which the construction of the starting point will be performed
//! @param latticeSpacing The length of an edge in the cubic lattice
//! @return The positions of the starting point
template <Dimension D, ParticNum N>
Positions<D, N> BuildFCCStartPoint_(CoordBounds<D> bounds, FPType latticeSpacing) {
    static_assert(D == Dimension{3u});
    Positions<D, N> result;

    // Calculate the center of the bounds
    Position<D> center;
    for (Dimension d = 0; d < D; ++d) {
        center[d].val = (bounds[d].lower.val + bounds[d].upper.val) / 2;
    }

    FPType const diagSpacing = latticeSpacing / std::sqrt(vmcp::FPType{2});

    // Offset positions for FCC lattice arrangement
    FPType const offset[4][3] = {{0, 0, 0},
                                 {diagSpacing, diagSpacing, 0},
                                 {diagSpacing, 0, diagSpacing},
                                 {0, diagSpacing, diagSpacing}};

    // Priority queue for positions to fill
    std::queue<Position<D>> positionsToFill;
    positionsToFill.push(center);

    UIntType particlesPlaced = 0;
    std::set<std::array<FPType, D>> visited;

    // Place particles iteratively
    while (!positionsToFill.empty() && particlesPlaced < N) {
        Position<D> basePosition = positionsToFill.front();
        positionsToFill.pop();

        for (UIntType offsetIdx = 0; offsetIdx < 4; ++offsetIdx) {
            Position<D> shiftedPos = basePosition;
            for (Dimension d = 0; d < D; ++d) {
                shiftedPos[d].val += offset[offsetIdx][d];
            }

            // Check if the position is unique
            std::array<FPType, D> posKey;
            for (Dimension d = 0; d < D; ++d) {
                posKey[d] = shiftedPos[d].val;
            }
            if (visited.count(posKey)) {
                continue;
            }
            visited.insert(posKey);

            // Store the position
            result[particlesPlaced] = shiftedPos;
            ++particlesPlaced;

            if (particlesPlaced == N) {
                return result;
            }

            // Add neighboring positions to the queue
            for (Dimension d = 0; d < D; ++d) {
                for (IntType sign = -1; sign <= 1; sign += 2) {
                    Position<D> neighbor = basePosition;
                    neighbor[d].val += sign * latticeSpacing;
                    positionsToFill.push(neighbor);
                }
            }
        }
    }
    return result;
}

//! @}

//! @defgroup Helper functions for plotting the energies
//! @brief Helper methods used for plotting in three different cases: without interactions and var params,
//! without interactions but with var params, with both interactions and var params
//! @{

//! @brief Creates a 2 dimensional graph to plot the energies as a function of a non variational
//! parameter alpha and the exact ground state energy
//! @param alphaVals The values of alpha
//! @param energyVals The energies
//! @param confints The confidence intervals
//! @param exactEns The exact ground state energies (it is the same value for each alpha)
//! @param method The name of the method used (e.g. "Metropolis Analytical" or "Imp. Samp. Numeric")
//! @return The Canvas containing the plot
template <Dimension D, ParticNum N>
sciplot::Canvas MakeGraphNoInt(std::vector<FPType> const &alphaVals, std::vector<FPType> const &energyVals,
                               std::vector<ConfInterval> const &confInts, std::vector<FPType> const &exactEns,
                               std::string method) {
    // Manipulate confidence intervals
    std::vector<FPType> confIntErr(confInts.size());
    std::transform(confInts.begin(), confInts.end(), confIntErr.begin(),
                   [](const ConfInterval &ci) { return (ci.max.val - ci.min.val) / 2; });
    // Plotting
    sciplot::Plot2D plot;

    plot.legend().atTop().fontName("Palatino").fontSize(12).displayHorizontal().displayExpandWidthBy(2);
    plot.xlabel("alpha");
    plot.ylabel("vmc energy");

    plot.drawCurve(alphaVals, energyVals)
        .lineColor("#191970") // Midnight Blue
        .label(method + " D=" + std::to_string(D) + ", N=" + std::to_string(N));

    plot.drawErrorBarsY(alphaVals, energyVals, confIntErr)
        .lineColor("#FF6347") // Tomato red
        .lineWidth(1)
        .lineType(6)
        .label("Conf. Int. with 95% Conf Lvl");

    plot.drawCurve(sciplot::linspace(0.2, 0.8, 100), exactEns)
        .lineColor("#3CB371") // Medium Sea Green
        .label("Exact solution");

    plot.grid().lineWidth(1).lineColor("#9370DB").show(); // Lavender purple

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};
    return canvas;
}

//! @brief Creates a 2 dimensional graph to plot the energies as a function of alpha around its best
//! value, the exact ground state energy and the energy associated to the best alpha found with
//! gradient descent
//! @param alphaVals The values of alpha
//! @param energyVals The energies
//! @param confInts The confidence intervals
//! @param exactEns The exact ground state energies (it is the same value for each alpha)
//! @param varParams The variational parameters (just one in our case)
//! @param varEnergies The eenrgies associated to each variational parameter (just one in our case)
//! @param method The name of the method used (e.g. "Metropolis Analytical" or "Imp. Samp. Numeric")
//! @return The Canvas containing the plot
template <Dimension D, ParticNum N>
sciplot::Canvas MakeGraphNoIntVar(std::vector<FPType> const &alphaVals, std::vector<FPType> const &energyVals,
                                  std::vector<ConfInterval> const &confInts,
                                  std::vector<FPType> const &exactEns, std::vector<FPType> const &varParams,
                                  std::vector<FPType> const &varEnergies, std::string method) {
    // Manipulate confidence intervals
    std::vector<FPType> confIntErr(confInts.size());
    std::transform(confInts.begin(), confInts.end(), confIntErr.begin(),
                   [](const ConfInterval &ci) { return (ci.max.val - ci.min.val) / 2; });
    // Plotting
    sciplot::Plot2D plot;

    plot.legend().atTop().fontName("Palatino").fontSize(12).displayHorizontal().displayExpandWidthBy(2);
    plot.xlabel("alpha");
    plot.ylabel("vmc energy");
    plot.xrange(0.44, 0.56);

    plot.drawCurve(alphaVals, energyVals)
        .lineColor("#191970") // Midnight Blue
        .label(method + " D=" + std::to_string(D) + ", N=" + std::to_string(N));

    plot.drawErrorBarsY(alphaVals, energyVals, confIntErr)
        .lineColor("#FF6347") // Tomato red
        .lineWidth(1)
        .lineType(6)
        .label("Conf. Int. with 95% Conf. Lvl.");

    plot.drawCurve(sciplot::linspace(0.44, 0.56, 100), exactEns)
        .lineColor("#3CB371") // Medium Sea Green
        .label("Exact solution");

    plot.drawPoints(varParams, varEnergies)
        .lineColor("#FF0000") // Red
        .pointType(2)
        .pointSize(1)
        .label("Variational energy for best alpha");

    plot.grid().lineWidth(1).lineColor("#9370DB").show(); // Lavender purple

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};
    return canvas;
}

//! @brief Creates a 2 dimensional graph to plot the energies/N as a function of N (i.e. # of particles)
//! @param energyVals The energies
//! @param confints The confidence intervals
//! @param NVals The values of the number of particles
void DrawGraphInt(std::vector<FPType> const &energyVals, std::vector<ConfInterval> const &confInts,
                  std::vector<ParticNum> const &NVals) {
    // Manipulate confidence intervals
    std::vector<FPType> confIntErr(confInts.size());
    std::transform(confInts.begin(), confInts.end(), confIntErr.begin(),
                   [](const ConfInterval &ci) { return (ci.max.val - ci.min.val) / 2; });

    // Construct file paths
    std::string folder = "./artifacts/Int";
    std::string file = folder + "/plot.pdf";

    std::filesystem::create_directories(folder);

    sciplot::Plot2D plot;

    plot.legend().atTop().fontName("Palatino").fontSize(12).displayHorizontal().displayExpandWidthBy(2);
    plot.xlabel("N (# of particles)");
    plot.ylabel("vmc energy / N");
    plot.xrange(1, 8);

    plot.drawCurve(NVals, energyVals)
        .lineColor("#191970") // Midnight Blue
        .label("Metropolis Analytical");

    plot.drawErrorBarsY(NVals, energyVals, confIntErr)
        .lineColor("#FF6347") // Tomato red
        .lineWidth(1)
        .lineType(6)
        .label("Conf Int with 95% Conf Lvl");

    plot.grid().lineWidth(1).lineColor("#9370DB").show(); // Lavender purple

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};

    canvas.save(file);
}

//! @}

} // namespace vmcp

#endif