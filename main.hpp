//!
//! @file main.hpp
//! @brief Helper functions for main
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the helper functions used inside HO in main.cpp
//! @see main.cpp
//!

#ifndef VMCPROJECT_MAIN_HPP
#define VMCPROJECT_MAIN_HPP

#include "src/vmcp.hpp"

#include "sciplot/sciplot.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>

namespace vmcp {

//! @defgroup Helper functions for computing the energies
//! @brief Helper methods called in the first section of HO method
//! @{

//! @brief Avoids manually repeating the Bound initialization by generating directly an array of bounds of
//! dimension D
//! @param lower The lower coordinate bound
//! @param upper The upper coordinate bound
//! @return The array of D bounds where all the bounds are the same
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
    assert(sqrdDist >= FPType{0});
    return std::sqrt(sqrdDist);
}

//! @brief Creates an initial position placing particles in a face-centered cubic lattice.
//! @param bounds The region in which the construction of the starting point will be done
//! @param latticeSpacing The radius of the hard sphere particles
//! @return The positions of the starting point
//!
//! Helper for 'VMCLocEnAndPoss'.
template <Dimension D, ParticNum N>
Positions<D, N> BuildFCCStartPoint_(CoordBounds<D> bounds, FPType latticeSpacing) {
    static_assert(D == Dimension{3u});
    Positions<D, N> result;

    // Calculate the center of the bounds
    Position<D> center;
    for (Dimension d = 0; d < D; ++d) {
        center[d].val = (bounds[d].lower.val + bounds[d].upper.val) / 2;
    }

    FPType const diagSpacing = latticeSpacing / 2;

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
                for (int sign = -1; sign <= 1; sign += 2) {
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
//! @brief Helper methods called in the second section of HO method
//! @{

//! @brief Creates a 2 dimensional graph to plot the energies as a function of alpha and the exact ground
//! state energy
//! @param alphaVals The values of alpha
//! @param energyVals The energies
//! @param errorVals The standard deviations of energies
//! @param exactEns The exact ground state energies (of course it is the same value for each alpha)
//! @return The Canvas containing the plot
template <Dimension D, ParticNum N>
sciplot::Canvas DrawGraphNoInt(std::vector<FPType> const &alphaVals, std::vector<FPType> const &energyVals,
                               std::vector<FPType> const &errorVals, std::vector<FPType> const &exactEns) {
    sciplot::Plot2D plot;

    plot.fontName("Palatino").fontSize(14);

    plot.legend().atTop().fontSize(14).displayHorizontal().displayExpandWidthBy(2);
    plot.xlabel("alpha");
    plot.ylabel("vmc energy");

    plot.drawCurve(alphaVals, energyVals)
        .lineColor("#191970") // Midnight Blue
        .label("Harmonic Oscillator D=" + std::to_string(D) + ", N=" + std::to_string(N));

    plot.drawErrorBarsY(alphaVals, energyVals, errorVals)
        .lineColor("#FF6347") // Tomato red
        .lineWidth(1)
        .label("");

    plot.drawCurve(sciplot::linspace(0.1, 1, 100), exactEns)
        .lineColor("#3CB371") // Medium Sea Green
        .label("Exact solution");

    plot.grid().lineWidth(1).lineColor("#9370DB").show(); // Lavender purple

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};
    return canvas;
}

//! @brief Creates a 2 dimensional graph to plot the energies as a function of the numebr of particles
//! @param NVals The values of the number of particles
//! @param energyVals The energies
//! @param errorVals The standard deviations of energies
void DrawGraphInt(std::vector<IntType> const &NVals, std::vector<FPType> const &energyVals,
                  std::vector<FPType> const &errorVals) {
    // Construct file paths
    std::string folder = "./artifacts/Int";
    std::string file = folder + "/plot.pdf";

    std::filesystem::create_directories(folder);

    sciplot::Plot2D plot;

    plot.fontName("Palatino").fontSize(14);

    plot.legend().atTop().fontSize(14).displayHorizontal().displayExpandWidthBy(2);
    plot.xlabel("N");
    plot.ylabel("vmc energy");

    plot.drawCurve(NVals, energyVals)
        .lineColor("#191970") // Midnight Blue
        .label("Harmonic Oscillator with Interactions");

    plot.drawErrorBarsY(NVals, energyVals, errorVals)
        .lineColor("#FF6347") // Tomato red
        .lineWidth(1)
        .label("");

    plot.grid().lineWidth(1).lineColor("#9370DB").show(); // Lavender purple

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};

    canvas.save(file);
}

//! @}

} // namespace vmcp

#endif