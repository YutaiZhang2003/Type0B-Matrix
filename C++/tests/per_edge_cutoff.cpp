#include "ramond/pipeline.hpp"
#include <iostream>
using namespace ramond;
int main() {
    require(physical_indices(20, true).size() == 2541, "wrong per-edge output count");
    require(physical_indices(20).size() == 506, "total-level output count changed");
    for (int level : {2, 3}) {
        SeriesDomain domain(Index{level, level, level, 0});
        auto box = schottky_vacuum(domain), total = schottky_vacuum(3 * level);
        for (auto k : virasoro_box_indices(domain.limits))
            require(get(box, k) == get(total, k), "Schottky box differs from total projection");
        for (auto &[k, v] : box)
            require(domain.contains(k), "Schottky leaked outside box");
    }
    for (bool inserted : {false, true}) {
        auto box = fermion_series(2, inserted, false, true);
        auto total = fermion_series(6, inserted);
        for (auto k : physical_indices(4, true))
            for (int p = 0; p < 8; p++)
                require(box[k][p] == total[k][p], "fermion box differs from total projection");
    }
    Index limits{2, 2, 1, 1};
    Series<Rational> a, b;
    for (auto k : virasoro_box_indices(limits)) {
        a[k] = Rational(1 + degree(k));
        b[k] = Rational(1 + k[1] + 3 * k[2]);
    }
    auto all = scalar_product(a, b, SeriesDomain(limits));
    auto diagonal = diagonal_product(a, b, 2, 1, 2, 1);
    int count = 0;
    for (auto &[k, v] : all)
        if (k[1] - k[2] == 1) {
            require(get(diagonal, Index{2 * k[0], k[1], k[2], 2 * k[3]}) == v,
                    "shifted diagonal product is incomplete");
            count++;
        }
    require(diagonal.size() == std::size_t(count), "wrong diagonal product domain");
    std::cout << "per-edge domains, exact Schottky and fermion projection, shifted products: passed\n";
}
