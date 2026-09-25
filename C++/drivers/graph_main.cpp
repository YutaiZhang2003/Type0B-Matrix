#include "scblocks/ns_recursion.hpp"
#include <filesystem>
#include <numeric>

using namespace scblocks;
namespace fs = std::filesystem;

// The Section 6 reference parameters. Theta NS-R-R uses bin/ramond and bin/pbw.
Setup setup_for(const std::string &channel) {
    if (channel == "theta-ns")
        return Setup("theta_ns");
    if (channel == "glasses-ns")
        return Setup("glasses_nn");
    if (channel == "glasses-ns-r")
        return Setup("glasses_nr");
    if (channel == "glasses-r")
        return Setup("glasses_rr");
    require(channel == "tetrahedron-r" || channel == "tetrahedron-ns", "unknown channel");
    Setup s("mercedes");
    if (channel == "tetrahedron-ns") {
        s.r.assign(6, false);
        s.cases.clear();
        for (int fm = 0; fm < 8; fm++)
            s.cases.push_back(
                {{__builtin_popcount(unsigned(fm)) % 2, fm & 1, (fm >> 1) & 1, (fm >> 2) & 1},
                 {1, 1, 1, 1},
                 {}});
    }
    return s;
}

int case_for(const Setup &s, int mask) {
    for (int ci = 0; ci < int(s.cases.size()); ci++) {
        bool match = true;
        for (int v = 0; v < int(s.graph.slots.size()); v++) {
            int parity = 0;
            for (int e : s.graph.slots[v])
                parity ^= (mask >> e) & 1;
            match = match && parity == s.cases[ci].f[v];
        }
        if (match)
            return ci;
    }
    throw std::runtime_error("no form parity matches the coefficient");
}

void scalar_json(std::ostream &out, const Setup &s, const Key &k, const S &value,
                 const char *field) {
    out << "{\"case\":" << case_for(s, parity_mask(k)) << ",\"level2\":[";
    for (int e = 0; e < s.graph.edges; e++) {
        if (e)
            out << ',';
        out << k[e];
    }
    out << "],\"" << field << "\":";
    row_json(out, {{parity_mask(k), value}});
    out << "}\n";
}

void metadata(const Setup &s, const std::string &channel, const std::string &method, int level,
              const fs::path &dir) {
    std::ofstream out(dir / "metadata.json.partial");
    require(bool(out), "cannot open metadata");
    out << "{\"schema_version\":1,\"conventions\":\"ordered_yutai_bpz_sewing_2026-09-25\",\"channel\":\""
        << channel << "\",\"method\":\"" << method << "\",\"level\":" << level
        << ",\"truncation\":\"" << (s.per_edge ? "per-edge" : "total")
        << "\",\"dps\":" << MP::digits << ",\"precision_bits\":" << MP::bits
        << ",\"b\":" << json_number(s.b) << ",\"momenta\":[";
    for (size_t e = 0; e < s.p.size(); e++) {
        if (e)
            out << ',';
        out << json_number(s.p[e]);
    }
    out << "],\"ramond_edges\":[";
    for (int e = 0; e < s.graph.edges; e++) {
        if (e)
            out << ',';
        out << (s.r[e] ? "true" : "false");
    }
    out << "],\"vertex_slots\":[";
    for (size_t v = 0; v < s.graph.slots.size(); v++) {
        if (v)
            out << ',';
        auto e = s.graph.slots[v];
        out << '[' << e[0] << ',' << e[1] << ',' << e[2] << ']';
    }
    out << "],\"edge_ends\":[";
    for (int e = 0; e < s.graph.edges; e++) {
        if (e)
            out << ',';
        out << '[' << s.graph.ends[e][0] << ',' << s.graph.ends[e][1] << ']';
    }
    out << "],\"cases\":[";
    for (size_t ci = 0; ci < s.cases.size(); ci++) {
        if (ci)
            out << ',';
        out << '{';
        int n = 0;
        for (auto item :
             {std::make_pair("f", s.cases[ci].f), std::make_pair("eta", s.cases[ci].eta),
              std::make_pair("marked_edges", s.cases[ci].marked)}) {
            if (n++)
                out << ',';
            out << '"' << item.first << "\":[";
            for (size_t j = 0; j < item.second.size(); j++) {
                if (j)
                    out << ',';
                out << item.second[j];
            }
            out << ']';
        }
        out << '}';
    }
    out << "]}\n";
    out.close();
    require(bool(out), "metadata write failed");
    fs::rename(dir / "metadata.json.partial", dir / "metadata.json");
}

#include "graph_dv.inc"
#include "graph_pbw_check.inc"

int all_ns(const Setup &s, int level, const std::string &method, const fs::path &dir) {
    require(std::none_of(s.r.begin(), s.r.end(), [](bool r) { return r; }),
            "dv/ns scalar commands require all-NS edges");
    require(s.name == "theta_ns" || s.name == "mercedes",
            "NS recursion supports theta and tetrahedron");
    auto targets = indices(s, level);
    auto domain = s.domain(level).doubled();
    double start = seconds();
    std::ofstream out(dir / "coefficients.jsonl.partial"), timing(dir / "summary.json.partial");
    require(bool(out) && bool(timing), "cannot open output");
    timing << std::setprecision(17);
    timing << "{\"level\":" << level << ",\"truncation\":\"" << (s.per_edge ? "per-edge" : "total")
           << "\",\"dps\":" << MP::digits << ",\"multidegrees\":" << targets.size();
    if (method == "dv") {
        DoubleVirasoro dv(s, level);
        std::vector<int> cases(s.cases.size());
        std::iota(cases.begin(), cases.end(), 0);
        auto hats = dv.compute({}, cases);
        double dt = seconds() - start;
        std::cerr << "double Virasoro complete: " << dt << " s\n";
        Sewing ff(s, true);
        std::vector<std::pair<Key, S>> aux;
        for (const auto &k : targets) {
            Row r = ff.coefficient(k, s.cases[0]);
            if (!total(k)) {
                require(getrow(r, 0) == S(1), "auxiliary constant is not one");
                continue;
            }
            for (const auto &[mask, x] : r) {
                require(mask == parity_mask(k), "auxiliary parity mismatch");
                if (x != S(0))
                    aux.emplace_back(k, x);
            }
        }
        double ft = seconds() - start - dt;
        Poly<S> remainder;
        for (int ci : cases)
            for (const auto &[k, row] : hats[ci])
                for (const auto &[mask, x] : row) {
                    require(mask == parity_mask(k) && ci == case_for(s, mask),
                            "enlarged sector mismatch");
                    remainder[k] += x;
                }
        hats.clear();
        int kernels[64][64];
        for (int a = 0; a < (1 << s.graph.edges); a++)
            for (int b = 0; b < (1 << s.graph.edges); b++)
                kernels[a][b] = s.graph.kernel(a, b);
        double recovery_start = seconds();
        int count = 0;
        for (const auto &k : targets) {
            S value = remainder[k];
            int km = parity_mask(k);
            scalar_json(out, s, k,
                        value * plumbing_bpz_weight<S>(s.graph, parity_mask(k)),
                        "double_virasoro");
            if (value != S(0))
                for (const auto &[a, f] : aux) {
                    if (total(k) + total(a) > int(domain))
                        break;
                    auto key = plus(k, a);
                    if (domain.contains(key))
                        remainder[key] -= S(kernels[km][parity_mask(a)]) * value * f;
                }
            if (++count % 1000 == 0)
                std::cerr << "recovery " << count << '/' << targets.size() << "\n";
        }
        timing << ",\"dv_seconds\":" << dt << ",\"fermion_seconds\":" << ft
               << ",\"recovery_and_output_seconds\":" << seconds() - recovery_start
               << ",\"branches\":" << dv.branches << ",\"ccy_transitions\":" << dv.ccy_transitions;
    } else {
        S q = s.b + S(1) / s.b;
        std::vector<S> h;
        for (const auto &p : s.p)
            h.push_back(q * q / S(8) - p * p / S(2));
        NSRecursion engine(s.graph, rational<S>(3, 2) + S(3) * q * q, h, s.domain(level));
        double seed = seconds() - start;
        auto coefficients = engine.coefficients(targets);
        int count = 0;
        for (const auto &k : targets) {
            scalar_json(out, s, k, coefficients.at(k), "c_recursion");
            if (++count % 1000 == 0)
                std::cerr << "NS recursion " << count << '/' << targets.size()
                          << " elapsed=" << seconds() - start << " s\n";
        }
        timing << ",\"schottky_seed_seconds\":" << seed
               << ",\"recursion_algorithm\":\"forward by output parity; Schottky factor restored "
                  "once\""
               << ",\"recursion_transitions\":" << engine.transitions
               << ",\"global_seed_terms\":" << engine.seed_terms
               << ",\"recursion_and_output_seconds\":" << seconds() - start - seed;
    }
    timing << ",\"total_seconds\":" << seconds() - start << "}\n";
    out.close();
    timing.close();
    require(bool(out) && bool(timing), "output write failed");
    fs::rename(dir / "coefficients.jsonl.partial", dir / "coefficients.jsonl");
    fs::rename(dir / "summary.json.partial", dir / "summary.json");
    return 0;
}

int main(int argc, char **argv) {
    try {
        int level = 5, dps = 40, selected_case = -1;
        std::string channel, method = "pbw-check", truncation = "total", output, b = "7/5";
        std::vector<std::string> p;
        for (int i = 1; i < argc; i += 2) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout
                    << "graph_blocks --channel "
                       "theta-ns|glasses-ns|glasses-ns-r|glasses-r|tetrahedron-r|tetrahedron-ns\n"
                    << "  --method pbw-check|dv|ns --level N --truncation total|per-edge --dps D "
                       "--output DIR\n"
                    << "  [--b B] [--P1 P ... --P6 P] [--case-index K for pbw-check]\n";
                return 0;
            }
            require(i + 1 < argc, "missing option value");
            std::string value = argv[i + 1];
            if (key == "--channel")
                channel = value;
            else if (key == "--method")
                method = value;
            else if (key == "--level")
                level = std::stoi(value);
            else if (key == "--dps")
                dps = std::stoi(value);
            else if (key == "--case-index")
                selected_case = std::stoi(value);
            else if (key == "--truncation")
                truncation = value;
            else if (key == "--output")
                output = value;
            else if (key == "--b")
                b = value;
            else if (key.size() == 4 && key.substr(0, 3) == "--P" && key[3] >= '1' &&
                     key[3] <= '6') {
                p.resize(6);
                p[key[3] - '1'] = value;
            } else
                throw std::invalid_argument("unknown option: " + key);
        }
        require(level >= 0 && level <= 100, "level must be in [0,100]");
        require(!output.empty(), "--output is required");
        require(truncation == "total" || truncation == "per-edge", "invalid truncation");
        require(method == "pbw-check" || method == "dv" || method == "ns", "invalid method");
        require(dps >= 30, "graph arithmetic requires --dps >=30");
        MP::precision(dps);
        Setup s = setup_for(channel);
        require(selected_case >= -1, "invalid case index");
        if (selected_case >= 0) {
            require(method == "pbw-check" && selected_case < int(s.cases.size()),
                    "invalid PBW case selection");
            Case chosen = s.cases[selected_case];
            s.cases = {chosen};
        }
        s.per_edge = truncation == "per-edge";
        s.b = parse<S>(b);
        for (size_t e = 0; e < p.size(); e++)
            if (!p[e].empty()) {
                require(e < s.p.size(), "momentum index exceeds graph edges");
                s.p[e] = parse<S>(p[e]);
            }
        fs::path dir(output);
        fs::create_directories(dir);
        require(!fs::exists(dir / "summary.json") && !fs::exists(dir / "coefficients.jsonl"),
                "output already contains a completed run; choose a new directory");
        metadata(s, channel, method, level, dir);
        if (method == "pbw-check")
            return validate_pbw(s, level, dir);
        bool all_ns_scalar = std::none_of(s.r.begin(), s.r.end(), [](bool r) { return r; }) &&
                             (s.name == "theta_ns" || s.name == "mercedes");
        if (method == "dv" && !all_ns_scalar)
            return graph_dv(s, level, dir);
        return all_ns(s, level, method, dir);
    } catch (const std::exception &e) {
        std::cerr << "graph_blocks: " << e.what() << '\n';
        return 2;
    }
}
