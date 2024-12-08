//!
//! @file main.hpp
//! @brief Helper functions for main
//! @authors Lorenzo Fabbri, Francesco Orso Pancaldi
//!
//! Contains the definitions of the helper functions used inside ComputeEnergies in main.cpp
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

RandomGenerator gen{(std::random_device())()};

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

template <Dimension D>
FPType Distance(Position<D> x, Position<D> y) {
    FPType sqrdDist =
        std::transform_reduce(x.begin(), x.end(), y.begin(), FPType(0.f), std::plus<FPType>(),
                              [](Coordinate &a, Coordinate &b) { return (a.val - b.val) * (a.val - b.val); });
    assert(sqrdDist >= FPType{0.f});
    return std::sqrt(sqrdDist);
}

template <Dimension D, ParticNum N>
sciplot::Canvas DrawGraph(std::vector<FPType> const &alphaVals, std::vector<FPType> const &energyVals,
                          std::vector<FPType> const &errorVals, std::vector<FPType> const &exactEns) {
    sciplot::Plot2D plot;

    plot.fontName("Palatino");
    plot.fontSize(14);

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

} // namespace vmcp

#endif
