#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <string>

constexpr int loops = 10;

template <typename Prng>
struct PrngStruct {
    Prng prng;
    std::array<int, loops> pancaNumbers;
    std::string name;
};

template <typename Prng>
void CheckProblems(PrngStruct<Prng> &rngSt) {
    std::uniform_int_distribution intDist(0, 1000);
    if (std::any_of(rngSt.pancaNumbers.begin(), rngSt.pancaNumbers.end(), [&](int i) {
            int rand = intDist(rngSt.prng);
            // std::cout << i << '\t' << rand << '\n';
            return i != rand;
        })) {
        std::cout << "Problem with " << rngSt.name << ".\n";
    } else {
        std::cout << "No problem with " << rngSt.name << ".\n";
    }
}

int main() {
    PrngStruct<std::mt19937> mersTw{
        std::mt19937(156239), {914, 542, 801, 209, 589, 795, 419, 869, 65, 515}, "Mersenne twister"};
    CheckProblems(mersTw);
    PrngStruct<std::default_random_engine> defRandEng{std::default_random_engine(156239),
                                                      {223, 332, 19, 37, 20, 965, 203, 144, 591, 483},
                                                      "Default random engine"};
    CheckProblems(defRandEng);
}
