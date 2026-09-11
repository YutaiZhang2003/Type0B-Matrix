#include "ramond/branching.hpp"
#include <fstream>
#include <iostream>
using namespace ramond;
template <class S> int run(int level, bool inserted, bool equal_ramond_momenta,
                          const std::string &output, bool benchmark) {
    S b = S(7) / S(5);
    std::array<S, 3> p{S(11) / S(23), S(13) / S(29), S(17) / S(31)};
    if (equal_ramond_momenta)
        p[2] = p[1];
    double start = seconds();
    OuterBranching<S> outer(b, p, level, 0, 0, inserted);
    std::cerr << "actions completed in " << outer.action_seconds << " s; support="
              << outer.support_size() << std::endl;
    std::exception_ptr failure;
    try {
        outer.prepare(1, benchmark && !inserted ? 1 : -1);
    } catch (...) {
        failure = std::current_exception();
    }
    double prepared = seconds() - start;
    double middle_seconds = 0;
    int middle_count = 0;
    if (benchmark && inserted && !failure) {
        double tick = seconds();
        MiddleBranching<S> middle(b, p[1], *outer.actions[0]);
        for (auto [incoming, outgoing] : middle_pairs(level))
            for (int a : {0, 1}) {
                middle.raw(outgoing, incoming, a);
                middle_count++;
            }
        middle_seconds = seconds() - tick;
    }
    if (!output.empty()) {
        std::ofstream out(output);
        require(bool(out), "cannot open branching coefficient output");
        out << "{\"level\":" << level << ",\"dps\":" << digits<S>()
            << ",\"status\":\"" << (failure ? "partial" : "complete") << '"'
            << ",\"production_signs\":" << (benchmark ? "true" : "false")
            << ",\"inserted\":" << (inserted ? "true" : "false")
            << ",\"seconds\":" << prepared << ",\"actions_seconds\":" << outer.action_seconds
            << ",\"ward_seconds\":" << outer.ward_seconds
            << ",\"middle_seconds\":" << middle_seconds
            << ",\"middle_count\":" << middle_count
            << ",\"total_branching_seconds\":" << prepared + middle_seconds
            << ",\"direct_boundary_values\":" << outer.direct_boundary_values
            << ",\"recursive_values\":" << outer.recursive_values
            << ",\"recursive_groups\":" << outer.recursive_groups
            << ",\"maximum_local_unknowns\":" << outer.maximum_local_unknowns
            << ",\"second_ward_rows\":" << outer.second_ward_rows
            << ",\"vanishing_first_pivots\":" << outer.vanishing_first_pivots
            << ",\"maximum_ward_residual\":" << outer.maximum_ward_residual
            << ",\"coefficients\":[";
        bool comma = false;
        for (int n : outer.ns)
            for (int r2 : outer.r)
                for (int r3 : outer.r)
                    for (int a : {0, 1})
                        for (int g : {0, 1})
                            if ((n / 2 + a + g) % 2 == 0)
                                for (int eta : {-1, 1}) {
                                    const S *value = nullptr;
                                    try {
                                        value = &outer.raw(n, r2, r3, a, g, eta);
                                    } catch (const std::out_of_range &) {
                                        continue; // Triple outside the closed support.
                                    }
                                    if (comma)
                                        out << ',';
                                    comma = true;
                                    out << "{\"labels\":[" << n << ',' << r2 << ',' << r3 << ','
                                        << a << ',' << g << ',' << eta << "],\"value\":"
                                        << json_number(*value) << '}';
                                }
        out << "]}\n";
    }
    if (failure)
        std::rethrow_exception(failure);
    if (level <= 5 && !benchmark) {
        double maximum = 0;
        int checked = 0;
        for (int n : outer.ns)
            for (int r2 : outer.r)
                for (int r3 : outer.r)
                    if (n * n / 4 + 2 * ramond_level(r2) + 2 * ramond_level(r3) <= 2 * level)
                        for (int a : {0, 1})
                            for (int g : {0, 1})
                                if ((n / 2 + a + g) % 2 == 0)
                                    for (int eta : {-1, 1}) {
                                        maximum = std::max(maximum, outer.second_ward_residual(
                                            {n, r2, r3}, a, g, eta));
                                        checked++;
                                    }
        std::cout << "second Ward identities checked=" << checked
                  << " maximum residual=" << maximum << std::endl;
        require(maximum < (std::is_same_v<S, Machine> ? 1e-8 : 1e-20),
                "second Ward identity residual exceeds tolerance");
    }
    std::cout << "level=" << level << " inserted=" << inserted
              << " total_seconds=" << seconds() - start
              << " actions_seconds=" << outer.action_seconds
              << " ward_seconds=" << outer.ward_seconds
              << " action_residual=" << outer.maximum_action_residual
              << " ward_residual=" << outer.maximum_ward_residual
              << " second_ward_rows=" << outer.second_ward_rows
              << " vanishing_first_pivots=" << outer.vanishing_first_pivots
              << " boundary_values=" << outer.direct_boundary_values
              << " recursive_values=" << outer.recursive_values
              << " recursive_groups=" << outer.recursive_groups
              << " max_local_unknowns=" << outer.maximum_local_unknowns
              << " middle_seconds=" << middle_seconds << std::endl;
    return 0;
}
int main(int argc, char **argv) {
    try {
        int level = argc > 1 ? std::stoi(argv[1]) : 3;
        int dps = argc > 2 ? std::stoi(argv[2]) : 40;
        bool inserted = argc > 3 && std::stoi(argv[3]);
        bool equal_ramond_momenta = argc > 4 && std::stoi(argv[4]);
        std::string output = argc > 5 ? argv[5] : "";
        bool benchmark = argc > 6 && std::stoi(argv[6]);
        if (!dps)
            return run<Machine>(level, inserted, equal_ramond_momenta, output, benchmark);
        MP::precision(dps);
        return run<MP>(level, inserted, equal_ramond_momenta, output, benchmark);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
