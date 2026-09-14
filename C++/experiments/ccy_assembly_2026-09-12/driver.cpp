#include "ccy_prototype.hpp"
#include "ramond/free_field.hpp"
#include <fstream>
#include <iostream>
using namespace ramond;
int main(int argc, char **argv) {
    require(argc == 8, "usage: driver LEVEL COPY NS4 LEFT4 RIGHT4 THIRD4 OUTPUT");
    int level = std::stoi(argv[1]), copy = std::stoi(argv[2]), ns = std::stoi(argv[3]),
        left = std::stoi(argv[4]), right = std::stoi(argv[5]), third = std::stoi(argv[6]);
    MP::precision(40);
    MP b = parse<MP>("7/5"), b2 = b * b;
    std::array<MP, 3> p{parse<MP>("11/23"), parse<MP>("13/29"), parse<MP>("17/31")};
    std::vector<MP> h{branch_weight(copy, ns, b, p[0]), branch_weight(copy, left, b, p[1]),
                      branch_weight(copy, right, b, p[1]), branch_weight(copy, third, b, p[2])};
    MP external = copy == 0 ? -(MP(1) + MP(2) * b2) / (MP(2) * (MP(1) - b2))
                            : (b2 + MP(2)) / (MP(2) * (MP(1) - b2));
    Index caps{level - (ns * ns + 7) / 8, level - (left * left - 1) / 8,
               level - (right * right - 1) / 8, level - (third * third - 1) / 8};
    for (auto n : caps) require(n >= 0, "negative branch budget");
    auto indices = virasoro_box_indices(caps);
    CCY<MP> engine(4, branch_central(copy, b), h, external);
    double start = seconds();
    auto result = engine.reduced(indices);
    double elapsed = seconds() - start;
    std::ofstream out(argv[7]);
    out << std::setprecision(17) << "{\"level\":" << level << ",\"copy\":" << copy
        << ",\"labels4\":[" << ns << ',' << left << ',' << right << ',' << third
        << "],\"caps\":[" << caps[0] << ',' << caps[1] << ',' << caps[2] << ',' << caps[3]
        << "],\"seconds\":" << elapsed << ",\"preparation\":" << engine.preparation_seconds
        << ",\"propagation\":" << engine.propagation_seconds << ",\"assembly\":" << engine.assembly_seconds
        << ",\"seed_terms\":" << engine.seed_terms << ",\"transitions\":" << engine.transitions
        << ",\"vertex_evaluations\":" << engine.vertex_factor_evaluations
        << ",\"vertex_hits\":" << engine.vertex_factor_hits << ",\"coefficients\":[";
    bool comma = false;
    for (auto k : indices) {
        if (comma) out << ',';
        comma = true;
        out << "{\"exponents\":[" << k[0] << ',' << k[1] << ',' << k[2] << ',' << k[3]
            << "],\"value\":" << json_number(result.at(k)) << '}';
    }
    out << "]}\n";
    require(bool(out), "failed to save benchmark");
    std::cout << "copy " << copy << " caps " << caps[0] << ',' << caps[1] << ',' << caps[2] << ',' << caps[3]
              << ": " << elapsed << " s, propagation " << engine.propagation_seconds
              << " s, assembly " << engine.assembly_seconds << " s\n";
}
