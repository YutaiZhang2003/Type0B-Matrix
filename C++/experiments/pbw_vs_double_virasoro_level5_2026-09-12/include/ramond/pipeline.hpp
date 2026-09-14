#pragma once
#include "branching.hpp"
#include "schottky.hpp"
#include <iostream>
namespace ramond {
struct Settings {
    int level = 3, dps = 0, p = 0, f = 0, eta = 1;
    bool inserted = true, record_sector = false, box = false;
    std::string b = "7/5";
    std::array<std::string, 3> momenta{"11/23", "13/29", "17/31"};
};
struct Timings {
    double branching = 0, actions = 0, outer_ward = 0, middle = 0, ccy = 0, products = 0,
           assembly = 0, auxiliary = 0, division = 0, schottky = 0, restoration = 0, total = 0;
};
template <class S> struct Result {
    ParitySeries<S> numerator, auxiliary, physical;
    Timings timing;
    size_t branch_cases = 0, virasoro_blocks = 0, transpose_reuse = 0,
           ccy_transitions = 0, ccy_seed_terms = 0;
    int direct_boundary_values = 0, recursive_branching_values = 0, recursive_groups = 0,
        maximum_local_unknowns = 0, second_ward_rows = 0, vanishing_first_pivots = 0;
    double action_residual = 0, ward_residual = 0, sector_residual = 0;
};
inline std::vector<Index> physical_indices(int cutoff, bool box = false) {
    std::vector<Index> out;
    if (box) {
        for (int a = 0; a <= cutoff; a++)
            for (int b = 0; b <= cutoff / 2; b++)
                for (int d = 0; d <= cutoff; d += 2)
                    out.push_back({a, b, b, d});
        std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
            return degree(a) != degree(b) ? degree(a) < degree(b) : a < b;
        });
        return out;
    }
    for (int d = 0; d <= cutoff; d++)
        for (int a = 0; a <= d; a++)
            for (int c = 0; c <= d - a; c += 2) {
                int rem = d - a - c;
                if (rem % 2 == 0)
                    out.push_back({a, rem / 2, rem / 2, c});
            }
    return out;
}
template <class S>
ParitySeries<S> recover(ParitySeries<S> numerator, const ParitySeries<S> &auxiliary, int cutoff,
                        bool inserted, bool record, double &maximum, bool box = false) {
    // Triangular star division in the recoverable parity sector. This is a
    // restricted inverse, not an inverse of the auxiliary in the full algebra.
    auto constant = auxiliary.find(Index{});
    require(constant != auxiliary.end(), "auxiliary constant is missing");
    const auto &ground = constant->second;
    int sigma = inserted ? -1 : 1;
    require(ground[0] != S(0), "auxiliary constant vanishes");
    for (int i = 0; i < 8; i++) {
        S expected = i == 0 ? ground[0] : i == 6 ? S(sigma) * ground[0] : S(0);
        require(small(ground[i] - expected, 1e-12, -std::max(20, digits<S>() - 10)),
                "invalid auxiliary constant sector");
    }
    S scalar = S(2) * ground[0];
    std::vector<Index> positive;
    for (const auto &[key, row] : auxiliary)
        if (degree(key) > 0 && physical_contains(key, cutoff / 2, box))
            positive.push_back(key);
    std::sort(positive.begin(), positive.end(), [](const auto &a, const auto &b) {
        return degree(a) != degree(b) ? degree(a) < degree(b) : a < b;
    });
    ParitySeries<S> answer;
    for (auto key : physical_indices(cutoff, box)) {
        auto &row = answer[key];
        auto found = numerator.find(key);
        if (found != numerator.end()) {
            for (int i = 0; i < 8; i++)
                row[i] = found->second[i] / scalar;
            numerator.erase(found);
        }
        std::array<S, 8> projected;
        double error = 0, scale = 1;
        for (int i = 0; i < 8; i++) {
            projected[i] = (row[i] + S(sigma * star_sign(6, i ^ 6)) * row[i ^ 6]) / S(2);
            require(finite(row[i]), "nonfinite recovered coefficient");
            error = std::max(error, magnitude(row[i] - projected[i]));
            scale = std::max(scale, magnitude(row[i]));
        }
        maximum = std::max(maximum, error / scale);
        require(record || error <= 1e-8 * scale,
                "numerator leaves the recoverable sector; use higher precision or explicit "
                "--sector-policy record");
        if (!inserted && !record)
            row = std::move(projected);
        for (auto shift : positive) {
            if (degree(key) + degree(shift) > (box ? 3 * cutoff : cutoff))
                break;
            if (!physical_contains(key + shift, cutoff / 2, box))
                continue;
            auto &future = numerator[key + shift];
            const auto &a = auxiliary.at(shift);
            for (int i = 0; i < 8; i++)
                if (a[i] != S(0))
                    for (int j = 0; j < 8; j++)
                        if (row[j] != S(0))
                            future[i ^ j] -= S(star_sign(i, j)) * a[i] * row[j];
        }
    }
    return answer;
}
template <class S> Result<S> pipeline(const Settings &settings) {
    Result<S> result;
    const int level = settings.level, cutoff = 2 * level;
    S b = parse<S>(settings.b);
    std::array<S, 3> p;
    for (int j = 0; j < 3; j++)
        p[j] = parse<S>(settings.momenta[j]);
    require(b != S(0) && b * b != S(1) && b == conjugate(b),
            "b must be real and different from 0,+/-1");
    double start = seconds(), tick = start;
    int eta = settings.eta, eta_prime = settings.inserted ? -eta : eta;
    std::cerr << "branching: starting at " << (settings.box ? "per-edge" : "total")
              << " level " << level << '\n';
    OuterBranching<S> outer(b, p, level, settings.f, settings.p, settings.inserted, settings.box);
    outer.prepare(eta, eta_prime);
    result.timing.branching = seconds() - tick;
    result.timing.actions = outer.action_seconds;
    result.timing.outer_ward = outer.ward_seconds;
    result.action_residual = outer.maximum_action_residual;
    result.ward_residual = outer.maximum_ward_residual;
    result.direct_boundary_values = outer.direct_boundary_values;
    result.recursive_branching_values = outer.recursive_values;
    result.recursive_groups = outer.recursive_groups;
    result.maximum_local_unknowns = outer.maximum_local_unknowns;
    result.second_ward_rows = outer.second_ward_rows;
    result.vanishing_first_pivots = outer.vanishing_first_pivots;
    std::cerr << "branching: " << result.timing.branching << " s; direct boundary values="
              << result.direct_boundary_values << ", recursive values="
              << result.recursive_branching_values << ", largest local relation="
              << result.maximum_local_unknowns << ", second Ward rows="
              << result.second_ward_rows << '\n';
    std::unique_ptr<MiddleBranching<S>> middle;
    std::map<std::array<int, 3>, S> middle_values;
    auto pairs = middle_pairs(level, settings.box);
    if (settings.inserted) {
        tick = seconds();
        middle = std::make_unique<MiddleBranching<S>>(b, p[1], *outer.actions[0]);
        for (auto [incoming, outgoing] : pairs)
            for (int a = 0; a < 2; a++)
                middle_values[{outgoing, incoming, a}] = middle->raw(outgoing, incoming, a);
        result.timing.middle = seconds() - tick;
    }
    S q = b + S(1) / b, b2 = b * b;
    std::array<S, 2> central{branch_central(0, b), branch_central(1, b)},
        external{-(S(1) + S(2) * b2) / (S(2) * (S(1) - b2)), (b2 + S(2)) / (S(2) * (S(1) - b2))};
    double last = seconds();
    for (int n1 : outer.ns) {
        std::map<std::array<int, 3>, Series<S>> products;
        std::vector<std::pair<int, int>> work = pairs;
        if (!settings.inserted) {
            work.clear();
            for (int n : outer.r)
                work.emplace_back(n, n);
        }
        for (auto [incoming, outgoing] : work)
            for (int n3 : outer.r) {
                Index shift{n1 * n1 / 4, ramond_level(incoming), ramond_level(outgoing),
                            2 * ramond_level(n3)};
                int remaining = cutoff - degree(shift);
                if (settings.box ? shift[0] > cutoff || shift[1] > level ||
                                       shift[2] > level || shift[3] > cutoff
                                 : remaining < 0)
                    continue;
                int budget = settings.box ? level : (cutoff - shift[0] - shift[3]) / 2,
                    left = budget - shift[1],
                    right = budget - shift[2];
                int first = (cutoff - shift[0]) / 2, third = (cutoff - shift[3]) / 2;
                if (std::min(left, right) < 0)
                    continue;
                tick = seconds();
                std::array<std::pair<int, S>, 2> factors;
                for (int alpha = 0; alpha < 2; alpha++) {
                    int gamma = ((settings.f - n1 / 2 - alpha) % 2 + 2) % 2,
                        index = ((n1 / 2 + settings.p) % 2 + 2) % 2 | (alpha << 1) | (gamma << 2);
                    S denominator = ns_norm(n1, b, p[0]) * r_norm(incoming, alpha, b, p[1]) *
                                    r_norm(n3, gamma, b, p[2]);
                    S factor = outer.raw(n1, incoming, n3, alpha, gamma, eta) *
                               outer.raw(n1, outgoing, n3, alpha, gamma, eta_prime);
                    if (settings.inserted) {
                        denominator *= r_norm(outgoing, alpha, b, p[1]);
                        factor *= middle_values.at({outgoing, incoming, alpha});
                    }
                    factor *= S(sign(n1 / 2) * theta_sign(index));
                    factor /= denominator;
                    factors[alpha] = {index, factor};
                }
                result.timing.assembly += seconds() - tick;
                if (factors[0].second == S(0) && factors[1].second == S(0))
                    continue;
                Series<S> product;
                std::array<int, 3> key{std::min(incoming, outgoing), std::max(incoming, outgoing),
                                       n3};
                auto stored = products.find(key);
                if (settings.inserted && stored != products.end()) {
                    tick = seconds();
                    for (const auto &[k, v] : stored->second)
                        product[{k[0], k[2], k[1], k[3]}] = v;
                    products.erase(stored);
                    result.transpose_reuse++;
                    result.timing.products += seconds() - tick;
                } else {
                    Index limits = settings.inserted ? Index{first, left, right, third}
                                                     : Index{first, left, third, 0};
                    auto indices = settings.box ? virasoro_box_indices(limits)
                                   : virasoro_indices(settings.inserted ? 4 : 3,
                                                    settings.inserted ? left : remaining / 2,
                                                    settings.inserted ? right : remaining / 2);
                    std::array<Series<S>, 2> blocks;
                    for (int copy = 0; copy < 2; copy++) {
                        tick = seconds();
                        std::vector<S> weights{branch_weight(copy, n1, b, p[0]),
                                               branch_weight(copy, incoming, b, p[1])};
                        if (settings.inserted)
                            weights.push_back(branch_weight(copy, outgoing, b, p[1]));
                        weights.push_back(branch_weight(copy, n3, b, p[2]));
                        CCY<S> engine(settings.inserted ? 4 : 3, central[copy], weights,
                                      external[copy]);
                        blocks[copy] = engine.reduced(indices);
                        result.virasoro_blocks++;
                        result.ccy_transitions += engine.transitions;
                        result.ccy_seed_terms += engine.seed_terms;
                        result.timing.ccy += seconds() - tick;
                    }
                    tick = seconds();
                    if (settings.inserted) {
                        product = diagonal_product(blocks[0], blocks[1], left, right,
                                                   settings.box ? first : -1,
                                                   settings.box ? third : -1);
                        products[key] = product;
                    } else {
                        auto combined = scalar_product(blocks[0], blocks[1],
                            settings.box ? SeriesDomain(limits) : SeriesDomain(remaining / 2));
                        for (const auto &[k, v] : combined)
                            product[{2 * k[0], k[1], k[1], 2 * k[2]}] = v;
                    }
                    result.timing.products += seconds() - tick;
                }
                tick = seconds();
                for (const auto &[k, v] : product) {
                    auto target = k + shift;
                    require(physical_contains(target, level, settings.box),
                            "assembled coefficient lies outside physical truncation");
                    auto &row = result.numerator[target];
                    for (const auto &[index, factor] : factors)
                        row[index] += factor * v;
                }
                result.timing.assembly += seconds() - tick;
                result.branch_cases++;
                if (seconds() - last > 15) {
                    std::cerr << "numerator: " << result.branch_cases << " branches, "
                              << seconds() - start << " s\n";
                    last = seconds();
                }
            }
    }
    tick = seconds();
    std::cerr << "numerator complete: " << result.branch_cases << " branches, "
              << result.timing.ccy << " s CCY; computing auxiliary and recovery\n";
    result.auxiliary = numerical_fermion(level, settings.inserted, q, false, settings.box);
    result.timing.auxiliary = seconds() - tick;
    tick = seconds();
    auto quotient = recover(result.numerator, result.auxiliary, cutoff, settings.inserted,
                            settings.record_sector, result.sector_residual, settings.box);
    result.timing.division = seconds() - tick;
    tick = seconds();
    SeriesDomain vacuum_domain = settings.box ? SeriesDomain(Index{level, level, level, 0})
                                               : SeriesDomain(level);
    auto vacuum = schottky_vacuum(vacuum_domain),
         squared = qmultiply(vacuum, vacuum, vacuum_domain);
    // Each of the two reduced Virasoro factors omitted one universal vacuum.
    result.timing.schottky = seconds() - tick;
    tick = seconds();
    for (const auto &[k, exact] : squared) {
        Index shift{2 * k[0], k[1], k[1], 2 * k[2]};
        S scalar = from_rational<S>(exact);
        for (const auto &[n, row] : quotient)
            if (physical_contains(n + shift, level, settings.box)) {
                auto &out = result.physical[n + shift];
                for (int i = 0; i < 8; i++)
                    out[i] += scalar * row[i];
            }
    }
    result.timing.restoration = seconds() - tick;
    result.timing.total = seconds() - start;
    return result;
}
template <class S> void encode_series(std::ostream &out, const ParitySeries<S> &series) {
    std::vector<Index> keys;
    for (const auto &[k, v] : series)
        keys.push_back(k);
    std::sort(keys.begin(), keys.end(), [](const auto &a, const auto &b) {
        return degree(a) != degree(b) ? degree(a) < degree(b) : a < b;
    });
    out << '[';
    bool comma = false;
    for (auto k : keys) {
        if (comma)
            out << ',';
        comma = true;
        out << "{\"exponents\":[" << k[0] << ',' << k[1] << ',' << k[2] << ',' << k[3]
            << "],\"values\":[";
        for (int j = 0; j < 8; j++) {
            if (j)
                out << ',';
            out << json_number(series.at(k)[j]);
        }
        out << "]}";
    }
    out << ']';
}
template <class S> void encode_result(std::ostream &out, const Settings &s, const Result<S> &r) {
    const auto &t = r.timing;
    out << std::setprecision(17)
        << "{\"status\":\"computed\",\"implementation\":\"C++17\",\"mode\":\""
        << (s.inserted ? "inserted" : "ordinary") << "\",\"truncation\":\""
        << (s.box ? "per-edge" : "total") << "\",";
    if (s.box)
        out << "\"q_level_cutoffs\":[" << s.level << ',' << s.level << ',' << s.level << ']';
    else
        out << "\"total_q_level\":" << s.level;
    out << ",\"dps\":" << s.dps << ",\"precision_bits\":" << (s.dps ? MP::bits : 53) << ",\"b\":\""
        << s.b << "\",\"momenta\":[\"" << s.momenta[0] << "\",\"" << s.momenta[1] << "\",\""
        << s.momenta[2] << "\"],\"p\":" << s.p << ",\"f\":" << s.f << ",\"etas\":[" << s.eta << ','
        << (s.inserted ? -s.eta : s.eta) << "],\"sector_policy\":\""
        << (s.record_sector ? "record" : "error")
        << "\",\"branching_method\":\"stored_recursion\""
        << ",\"exponent_convention\":\"q1^(a/2) q2^l q3^(d/2), stored as "
           "(a,l,l,d)\",\"timing_seconds\":{\"branching\":"
        << t.branching << ",\"actions\":" << t.actions << ",\"outer_ward\":" << t.outer_ward
        << ",\"middle\":" << t.middle << ",\"ccy\":" << t.ccy << ",\"products\":" << t.products
        << ",\"assembly\":" << t.assembly << ",\"auxiliary\":" << t.auxiliary
        << ",\"division\":" << t.division << ",\"schottky\":" << t.schottky
        << ",\"restoration\":" << t.restoration << ",\"total\":" << t.total
        << "},\"counts\":{\"branches\":" << r.branch_cases
        << ",\"virasoro_blocks\":" << r.virasoro_blocks
        << ",\"ccy_transitions\":" << r.ccy_transitions
        << ",\"ccy_seed_terms\":" << r.ccy_seed_terms
        << ",\"transposed_products_reused\":" << r.transpose_reuse
        << ",\"direct_boundary_values\":" << r.direct_boundary_values
        << ",\"recursive_branching_values\":" << r.recursive_branching_values
        << ",\"recursive_groups\":" << r.recursive_groups
        << ",\"maximum_local_unknowns\":" << r.maximum_local_unknowns
        << ",\"second_ward_rows\":" << r.second_ward_rows
        << ",\"vanishing_first_pivots\":" << r.vanishing_first_pivots
        << "},\"diagnostics\":{\"maximum_action_residual\":" << r.action_residual
        << ",\"maximum_ward_residual\":" << r.ward_residual
        << ",\"maximum_sector_residual\":" << r.sector_residual << "},\"reduced_numerator\":";
    encode_series(out, r.numerator);
    out << ",\"full_auxiliary\":";
    encode_series(out, r.auxiliary);
    out << ",\"coefficients\":";
    encode_series(out, r.physical);
    out << "}\n";
}
} // namespace ramond
