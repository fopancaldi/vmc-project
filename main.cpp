#include "vmcp.hpp"

#include <iostream>
#include <numbers>

using namespace vmcp;

int main() {
    auto psi{[](Positions<1, 1> x, VarParams) {
        return std::pow(std::numbers::pi_v<FPType>, -0.25) *
               std::pow(std::numbers::e_v<FPType>, -x[0][0].val * x[0][0].val / 2);
    }};
    VarParams varParams = {VarParam{1}};
    Bounds<1> bounds = {Bound{-10, 10}};
    auto pot{[](Positions<1, 1> x) { return x[0][0].val * x[0][0].val; }};
    auto kin{[&psi](Positions<1, 1> x, VarParams) {
        return std::pow(std::numbers::pi_v<FPType>, -0.25) *
               std::pow(std::numbers::e_v<FPType>, -x[0][0].val * x[0][0].val / 2) *
               (1 - x[0][0].val * x[0][0].val);
    }};
    RandomGenerator gen{(std::random_device())()};

    std::vector<Energy> energies = VMCEnergies<1, 1>(psi, varParams, kin, pot, bounds, gen);
    std::cout << "Done.\n";
    for (Energy energy : energies) {
        std::cout << energy.val << '\n';
    }
}
