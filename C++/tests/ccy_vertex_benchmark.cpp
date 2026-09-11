#include "ramond/ccy.hpp"
#include "ramond/free_field.hpp"
#include <fstream>
#include <iostream>
using namespace ramond;
int main(int argc, char **argv) {
    require(argc == 4, "usage: ccy_vertex_benchmark LEVEL COPY OUTPUT");
    const int level = std::stoi(argv[1]), copy = std::stoi(argv[2]);
    require(copy == 0 || copy == 1, "copy must be zero or one");
    MP::precision(40);
    MP b = parse<MP>("7/5"), b2 = b * b;
    std::array<MP, 3> p{parse<MP>("11/23"), parse<MP>("13/29"), parse<MP>("17/31")};
    std::vector<MP> h{branch_weight(copy, 0, b, p[0]), branch_weight(copy, -1, b, p[1]),
                      branch_weight(copy, 1, b, p[1]), branch_weight(copy, 1, b, p[2])};
    MP external = copy == 0 ? -(MP(1) + MP(2) * b2) / (MP(2) * (MP(1) - b2))
                            : (b2 + MP(2)) / (MP(2) * (MP(1) - b2));
    auto indices = virasoro_box_indices({level, level, level, level});
    CCY<MP> engine(4, branch_central(copy, b), h, external);
    double start = seconds();
    auto result = engine.reduced(indices);
    double elapsed = seconds() - start;
    std::ofstream out(argv[3]);
    out << std::setprecision(17) << "{\"level\":" << level << ",\"copy\":" << copy
        << ",\"seconds\":" << elapsed << ",\"seed_terms\":" << engine.seed_terms
        << ",\"transitions\":" << engine.transitions;
#ifdef RAMOND_VERTEX_CACHE_STATS
    out << ",\"vertex_factor_evaluations\":" << engine.vertex_factor_evaluations
        << ",\"vertex_factor_hits\":" << engine.vertex_factor_hits;
#endif
    out << ",\"coefficients\":[";
    bool comma = false;
    for (auto k : indices) {
        if (comma) out << ',';
        comma = true;
        out << "{\"exponents\":[" << k[0] << ',' << k[1] << ',' << k[2] << ',' << k[3]
            << "],\"value\":" << json_number(result.at(k)) << '}';
    }
    out << "]}\n";
    require(bool(out), "failed to save CCY benchmark");
    std::cout << "CCY box " << level << " copy " << copy << ": " << elapsed << " s, "
              << engine.seed_terms << " seed summands\n";
}
