// Complete physical-PBW comparison, including split sewing and spin evaluations.
int validate_pbw(const Setup &setup, int level, const std::filesystem::path &directory) {
    double start = seconds();
    auto targets = indices(setup, level);
    int bits = setup.graph.edges;
    std::cerr << "physical multidegrees=" << targets.size() << " total level=" << level
              << " cases=" << setup.cases.size() << "\n";
    DoubleVirasoro dv(setup, level);
    std::map<std::vector<int>, std::vector<int>> groups;
    std::vector<int> all;
    for (int ci = 0; ci < int(setup.cases.size()); ci++) {
        groups[setup.cases[ci].marked].push_back(ci);
        all.push_back(ci);
    }
    std::map<std::vector<int>, std::vector<Block>> numerators;
    numerators[{}] = dv.compute({}, all);
    for (auto [marked, cases] : groups)
        if (!marked.empty())
            numerators[marked] = dv.compute(marked, cases);
    double dv_seconds = seconds() - start;
    std::cerr << "DV done " << dv_seconds << " seconds\n";
    Sewing physical(setup, false), fermion(setup, true);
    std::map<std::vector<int>, Block> aux;
    for (auto [marked, cases] : groups)
        for (auto k : indices(setup, level, marked))
            aux[marked][k] = fermion.coefficient(k, setup.cases[cases[0]], marked);
    double fermion_seconds = seconds() - start - dv_seconds;
    std::ofstream coeff(directory / "coefficients.jsonl.partial");
    require(bool(coeff), "cannot open coefficient output");
    std::ofstream out(directory / "summary.json.partial");
    require(bool(out), "cannot open summary output");
    out << std::setprecision(17) << "{\"level\":" << level << ",\"truncation\":\""
        << (setup.per_edge ? "per-edge" : "total") << "\",\"dps\":" << MP::digits
        << ",\"physical_multidegrees\":" << targets.size() << ",\"cases\":[";
    int failures = 0;
    for (int ci = 0; ci < int(setup.cases.size()); ci++) {
        auto c = setup.cases[ci];
        double t = seconds();
        Block ph, rec;
        for (auto k : targets)
            ph[k] = physical.coefficient(k, c);
        double pbw_seconds = seconds() - t;
        Error forward, recovery, vanishing, unequal, spin, sector;
        auto expanded = indices(setup, level, c.marked);
        auto &hat = numerators[c.marked][ci];
        S scalar = power(S(2), setup.loop_count()) *
                   power((setup.b + S(1) / setup.b) / root(S(2)), c.marked.size());
        for (auto k : expanded) {
            Row expected;
            for (auto &[r, row] : ph) {
                Key pull = r;
                for (int j = 0; j < int(c.marked.size()); j++)
                    pull[bits + j] = r[c.marked[j]];
                if (!leq(pull, k))
                    continue;
                auto it = aux[c.marked].find(minus(k, pull));
                if (it == aux[c.marked].end())
                    continue;
                addstar(expected, row, it->second, setup.graph);
            }
            bool diagonal = true;
            for (int j = 0; j < int(c.marked.size()); j++)
                if (k[c.marked[j]] != k[bits + j])
                    diagonal = false;
            if (diagonal)
                forward.check(hat[k], expected, k, bits);
            else
                unequal.check(hat[k], expected, k, bits);
        }
        for (auto k : targets) {
            Key pull = k;
            for (int j = 0; j < int(c.marked.size()); j++)
                pull[bits + j] = k[c.marked[j]];
            Row remainder = hat[pull];
            for (auto &[r, row] : rec)
                if (leq(r, k) && r != k) {
                    Key sk = minus(k, r);
                    for (int j = 0; j < int(c.marked.size()); j++)
                        sk[bits + j] = sk[c.marked[j]];
                    auto it = aux[c.marked].find(sk);
                    if (it != aux[c.marked].end())
                        addstar(remainder, row, it->second, setup.graph, S(-1));
                }
            for (auto &[mask, x] : remainder)
                x /= scalar;
            rec[k] = remainder;
            Row recovered = remainder;
            recovery.check(recovered, ph[k], k, bits);
            Row sx, sy;
            for (int lift = 0; lift < (1 << bits); lift++) {
                for (auto [p, v] : recovered)
                    sx[lift] += S(sign(__builtin_popcount(unsigned(lift & p)))) * v;
                for (auto [p, v] : ph[k])
                    sy[lift] += S(sign(__builtin_popcount(unsigned(lift & p)))) * v;
            }
            spin.check(sx, sy, k, bits);
            if (!c.marked.empty())
                vanishing.check(numerators[{}][ci][k], {}, k, bits);
            coeff << "{\"case\":" << ci << ",\"level2\":[";
            for (int e = 0; e < bits; e++) {
                if (e)
                    coeff << ',';
                coeff << k[e];
            }
            coeff << "],\"physical\":";
            row_json(coeff, ph[k]);
            coeff << ",\"recovered\":";
            row_json(coeff, recovered);
            coeff << ",\"enlarged\":";
            row_json(coeff, hat[pull]);
            coeff << "}\n";
        }
        bool bad =
            forward.failed || recovery.failed || vanishing.failed || unequal.failed || spin.failed;
        failures += bad;
        if (ci)
            out << ',';
        out << "{\"case\":" << ci << ",\"insertions\":" << c.marked.size() << ",\"forward\":";
        forward.json(out);
        out << ",\"recovery\":";
        recovery.json(out);
        out << ",\"vanishing\":";
        vanishing.json(out);
        out << ",\"unequal_split\":";
        unequal.json(out);
        out << ",\"spin_evaluations\":";
        spin.json(out);
        out << ",\"physical_pbw_seconds\":" << pbw_seconds << ",\"case_seconds\":" << seconds() - t
            << '}';
        out.flush();
        coeff.flush();
        std::cerr << "case=" << ci << " recovery=" << recovery.scaled
                  << " forward=" << forward.scaled << " unequal=" << unequal.scaled
                  << " vanish=" << vanishing.absolute << " seconds=" << seconds() - t << "\n";
    }
    out << "],\"failed_cases\":" << failures << ",\"dv_seconds\":" << dv_seconds
        << ",\"fermion_seconds\":" << fermion_seconds
        << ",\"runtime_seconds\":" << seconds() - start
        << ",\"ccy_transitions\":" << dv.ccy_transitions << ",\"branch_tuples\":" << dv.branches
        << "}\n";
    out.close();
    coeff.close();
    require(bool(out) && bool(coeff), "output write failed");
    std::filesystem::rename(directory / "coefficients.jsonl.partial",
                            directory / "coefficients.jsonl");
    std::filesystem::rename(directory / "summary.json.partial", directory / "summary.json");
    return failures ? 1 : 0;
}
