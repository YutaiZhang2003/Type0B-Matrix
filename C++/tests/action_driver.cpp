#include "ramond/free_field.hpp"
#include <iostream>
using namespace ramond;
template <class S> void run(int n, int dps) {
    (void)dps;
    FreeField<S> module(true, parse<S>("7/5"), parse<S>("13/29"));
    double start = seconds();
    auto result = ramond_actions(module, n, true);
    std::cout << "{\"seconds\":" << seconds() - start
              << ",\"residual\":" << result.minus_fit.relative_residual << ",\"coefficients\":[";
    bool comma = false;
    for (const auto &t : result.minus) {
        if (comma)
            std::cout << ',';
        comma = true;
        std::cout << "{\"n4\":" << t.label << ",\"first\":[";
        for (size_t i = 0; i < t.first.size(); i++) {
            if (i)
                std::cout << ',';
            std::cout << t.first[i];
        }
        std::cout << "],\"second\":[";
        for (size_t i = 0; i < t.second.size(); i++) {
            if (i)
                std::cout << ',';
            std::cout << t.second[i];
        }
        std::cout << "],\"value\":" << json_number(t.coefficient) << '}';
    }
    std::cout << "]}\n";
}
int main(int argc, char **argv) {
    try {
        int n = argc > 1 ? std::stoi(argv[1]) : 7, d = argc > 2 ? std::stoi(argv[2]) : 0;
        if (d) {
            MP::precision(d);
            run<MP>(n, d);
        } else
            run<Machine>(n, 0);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
