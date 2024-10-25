#include "vmcp.hpp"

#include <iostream>

using namespace vmcp;

int main() {
    // TODO: This line will be removed
    std::cout << "Test: " << Test(2) << '\n';

    Wavefunction<1, 1> psi{[](Positions<1, 1> x, VarParams) {
        return std::pow(pi, -0.25) * std::pow(e, -x[0][0] * x[0][0] / 2);
    }};
    VarParams varParams = {1};
    Bounds<1> bounds = {Bound{-10, 10}};
    Potential<1, 1> pot{[](Positions<1, 1> x) { return x[0][0] * x[0][0]; }};
    KinEnergy<1, 1> kin{[&psi](Positions<1, 1> x, VarParams) {
        return std::pow(pi, -0.25) * std::pow(e, -x[0][0] * x[0][0] / 2) * (1 - x[0][0] * x[0][0]);
    }};
    RandomGenerator gen{(std::random_device())()};

    std::vector<FPType> energies = VMCIntegral<1, 1>(psi, varParams, bounds, pot, kin, gen);
    std::cout << "Done.\n";
    for (FPType energy : energies) {
        std::cout << energy << '\n';
    }
}
