#include "ramond/fermion.hpp"
#include <iostream>
using namespace ramond;

int main() {
    const auto series = fermion_series(2, false);
    for (const auto &[key, row] : series)
        for (int parity = 0; parity < 8; parity++)
            if (row[parity] != 0)
                std::cout << key[0] << ' ' << 2 * key[1] << ' '
                          << key[3] << ' ' << parity << ' '
                          << row[parity] << '\n';
}
