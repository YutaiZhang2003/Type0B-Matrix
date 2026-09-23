// Physical graph blocks without a physical-PBW reference calculation.
int graph_dv(const Setup &setup, int level, const fs::path &directory) {
    double start = seconds();
    auto targets = indices(setup, level);
    int bits = setup.graph.edges;
    DoubleVirasoro dv(setup, level);
    Sewing fermion(setup, true);
    std::map<std::vector<int>, std::vector<int>> groups;
    for (int ci = 0; ci < int(setup.cases.size()); ci++)
        groups[setup.cases[ci].marked].push_back(ci);
    std::ofstream output(directory / "coefficients.jsonl.partial");
    require(bool(output), "cannot open coefficient output");
    double numerator_seconds = 0, fermion_seconds = 0, recovery_seconds = 0;
    for (const auto &[marked, cases] : groups) {
        double tick = seconds();
        auto numerators = dv.compute(marked, cases);
        numerator_seconds += seconds() - tick;
        // For a diagonal target, subtracting a physical monomial leaves a
        // diagonal auxiliary monomial. Nonzero fermion modes are still
        // retained inside the expanded double-Virasoro numerator.
        tick = seconds();
        Block auxiliary;
        for (auto k : targets) {
            Key split = k;
            for (int j = 0; j < int(marked.size()); j++)
                split[bits + j] = k[marked[j]];
            auxiliary[k] = fermion.coefficient(split, setup.cases[cases.front()], marked);
        }
        fermion_seconds += seconds() - tick;
        tick = seconds();
        S scalar = power(S(2), setup.loop_count()) *
                   power((setup.b + S(1) / setup.b) / root(S(2)), marked.size());
        for (int ci : cases) {
            Block recovered;
            for (auto k : targets) {
                Key split = k;
                for (int j = 0; j < int(marked.size()); j++)
                    split[bits + j] = k[marked[j]];
                Row remainder = numerators[ci][split];
                for (const auto &[r, row] : recovered)
                    if (leq(r, k) && r != k) {
                        auto found = auxiliary.find(minus(k, r));
                        if (found != auxiliary.end())
                            addstar(remainder, row, found->second, setup.graph, S(-1));
                    }
                for (auto &[mask, x] : remainder)
                    x /= scalar;
                recovered[k] = remainder;
                output << "{\"case\":" << ci << ",\"level2\":[";
                for (int e = 0; e < bits; e++) {
                    if (e)
                        output << ',';
                    output << k[e];
                }
                output << "],\"double_virasoro\":";
                row_json(output, remainder);
                output << "}\n";
            }
        }
        recovery_seconds += seconds() - tick;
    }
    output.close();
    require(bool(output), "coefficient write failed");
    std::ofstream summary(directory / "summary.json.partial");
    require(bool(summary), "cannot open summary output");
    summary << std::setprecision(17)
            << "{\"status\":\"computed\",\"physical_pbw_used\":false,\"level\":" << level
            << ",\"truncation\":\"" << (setup.per_edge ? "per-edge" : "total")
            << "\",\"dps\":" << MP::digits << ",\"cases\":" << setup.cases.size()
            << ",\"multidegrees_per_case\":" << targets.size()
            << ",\"dv_seconds\":" << numerator_seconds << ",\"fermion_seconds\":" << fermion_seconds
            << ",\"recovery_seconds\":" << recovery_seconds
            << ",\"total_seconds\":" << seconds() - start << ",\"branches\":" << dv.branches
            << ",\"ccy_transitions\":" << dv.ccy_transitions << "}\n";
    summary.close();
    require(bool(summary), "summary write failed");
    fs::rename(directory / "coefficients.jsonl.partial", directory / "coefficients.jsonl");
    fs::rename(directory / "summary.json.partial", directory / "summary.json");
    return 0;
}
