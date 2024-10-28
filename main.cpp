#include "vmcp.hpp"

#include <iostream>
#include <numbers>

using namespace vmcp;

FPType wavefHO(Positions<1, 1> x, VarParams<0>) {
    return std::pow(std::numbers::pi_v<FPType>, -0.25) *
           std::pow(std::numbers::e_v<FPType>, -x[0][0].val * x[0][0].val / 2);
}
FPType potHO(Positions<1, 1> x) { return x[0][0].val * x[0][0].val; }
FPType kinHO(Positions<1, 1> x, VarParams<0>) {
    return (1 - x[0][0].val * x[0][0].val) * wavefHO(x, VarParams<0>{});
}

int main() {
    int numberEnergies = 100;
    Bounds<1> bounds = {Bound{-10, 10}};
    RandomGenerator gen{(std::random_device())()};
    std::vector<Energy> energies =
        VMCEnergies<1, 1, 0>(wavefHO, VarParams<0>{}, kinHO, potHO, bounds, numberEnergies, gen);

    std::cout << "Calculated energies:\n";
    for (Energy energy : energies) {
        std::cout << energy.val << ' ';
    }
    std::cout << '\n';
}
