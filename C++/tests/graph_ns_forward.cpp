#include "scblocks/ns_recursion.hpp"
#include <iostream>

using namespace scblocks;

int main() {
    try {
        MP::precision(40);
        Setup setup("theta_ns");
        setup.per_edge = true;
        S q = setup.b + S(1) / setup.b;
        std::vector<S> weights;
        for (const auto &p : setup.p)
            weights.push_back(q * q / S(8) - p * p / S(2));
        NSRecursion engine(setup.graph, rational<S>(3, 2) + S(3) * q * q, weights, setup.domain(3));
        auto targets = indices(setup, 3);
        auto batch = engine.coefficients(targets);
        double maximum = 0, absolute = 0;
        for (const auto &k : targets) {
            auto a = batch.at(k), b = engine.coefficient(k);
            double error = magnitude(a - b);
            absolute = std::max(absolute, error);
            maximum = std::max(maximum, error / std::max({1., magnitude(a), magnitude(b)}));
        }
        bool passed = maximum < 1e-25;
        std::cout << std::setprecision(17) << "{\"passed\":" << (passed ? "true" : "false")
                  << ",\"channel\":\"theta-ns\",\"truncation\":\"per-edge\",\"level\":3,\"dps\":40,"
                     "\"multidegrees\":"
                  << targets.size() << ",\"max_scaled\":" << maximum
                  << ",\"max_absolute\":" << absolute << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
