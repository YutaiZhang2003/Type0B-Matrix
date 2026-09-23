#pragma once
#include "dv_engine.hpp"
namespace scblocks {
// Full primitive-product NS seed in the ordered graph's parity algebra.
inline int parity_mask(const Key &k) {
    int m = 0;
    for (int e = 0; e < 8; e++)
        m |= (k[e] & 1) << e;
    return m;
}
inline QPoly embed_twice(const QPoly &a) {
    QPoly b;
    for (const auto &[k, v] : a) {
        Key n{};
        for (int e = 0; e < 8; e++)
            n[e] = 2 * k[e];
        b.emplace(n, v);
    }
    return b;
}
inline QPoly star_q(const QPoly &a, const QPoly &b, const Graph &g, Cutoff cutoff) {
    QPoly c;
    for (const auto &[ka, va] : a)
        for (const auto &[kb, vb] : b) {
            auto k = plus(ka, kb);
            if (cutoff.contains(k))
                c[k] += Rational(g.kernel(parity_mask(ka), parity_mask(kb))) * va * vb;
        }
    for (auto it = c.begin(); it != c.end();)
        if (it->second == 0)
            it = c.erase(it);
        else
            ++it;
    return c;
}
inline QPoly ns_schottky(const Graph &g, Cutoff level) {
    require((g.edges == 6 && g.slots.size() == 4) || (g.edges == 3 && g.slots.size() == 2),
            "NS seed supports theta and tetrahedron");
    auto twice = level.doubled();
    auto one = constant(1);
    std::array<QMat, 3> coord{QMat{QPoly{}, one, one, {}}, QMat{one, constant(-1), {}, one},
                              QMat{one, {}, {}, one}};
    auto back = coord;
    back[1] = {one, one, {}, one};
    std::vector<QMat> maps;
    std::vector<int> det_sign;
    for (int e = 0; e < g.edges; e++) {
        int from = g.ends[e][0] % 3, to = g.ends[e][1] % 3;
        det_sign.push_back(sign(1 + (from == 0) + (to == 0)));
        for (int d = 0; d < 2; d++) {
            Key k{};
            k[e] = 1;
            QMat inv{QPoly{}, QPoly{{k, Rational(1)}}, one, {}};
            maps.push_back(multiply(back[g.ends[e][1 - d] % 3],
                                    multiply(inv, coord[g.ends[e][d] % 3], level), level));
        }
    }
    std::set<Part> classes;
    Part word;
    int maximum = 2 * level / 3;
    std::function<void(int, int)> visit = [&](int start, int v) {
        int n = word.size();
        if (n && v == start && word.back() != (word.front() ^ 1)) {
            bool primitive = true;
            for (int p = 1; p < n; p++)
                if (n % p == 0) {
                    bool same = true;
                    for (int j = 0; j < n; j++)
                        if (word[j] != word[j % p])
                            same = false;
                    if (same)
                        primitive = false;
                }
            if (primitive) {
                Part inv;
                for (auto it = word.rbegin(); it != word.rend(); ++it)
                    inv.push_back(*it ^ 1);
                Part best = word;
                for (auto w : {word, inv})
                    for (int r = 0; r < n; r++) {
                        Part rot(w.begin() + r, w.end());
                        rot.insert(rot.end(), w.begin(), w.begin() + r);
                        best = std::min(best, rot);
                    }
                classes.insert(best);
            }
        }
        if (n == maximum)
            return;
        for (int e = 0; e < g.edges; e++)
            for (int d = 0; d < 2; d++)
                if (g.ends[e][d] / 3 == v) {
                    int arc = 2 * e + d;
                    if (n && arc == (word.back() ^ 1))
                        continue;
                    word.push_back(arc);
                    Key visits{};
                    for (int a : word)
                        visits[a / 2] += 3;
                    if (twice.contains(visits))
                        visit(start, g.ends[e][1 - d] / 3);
                    word.pop_back();
                }
    };
    if (maximum)
        for (int v = 0; v < int(g.slots.size()); v++)
            visit(v, v);
    // i^tau maps the commuting closed-cycle star algebra to scalar SL(2) lifts.
    std::vector<int> basis = g.edges == 6 ? std::vector<int>{11, 22, 37} : std::vector<int>{3, 5};
    std::array<int, 64> tau;
    tau.fill(-1);
    for (int bits = 0; bits < (1 << basis.size()); bits++) {
        int m = 0, p = 0;
        for (int j = 0; j < int(basis.size()); j++)
            if (bits & (1 << j)) {
                p += (g.kernel(basis[j], basis[j]) < 0 ? 1 : 0) +
                     (g.kernel(m, basis[j]) < 0 ? 2 : 0);
                m ^= basis[j];
            }
        tau[m] = p % 4;
    }
    for (int a = 0; a < 64; a++)
        if (tau[a] >= 0)
            for (int b = 0; b < 64; b++)
                if (tau[b] >= 0) {
                    require(g.kernel(a, b) == g.kernel(b, a),
                            "closed cycle algebra is not commutative");
                    require((tau[a] + tau[b] - tau[a ^ b] + (g.kernel(a, b) < 0 ? 2 : 0) + 8) % 4 ==
                                0,
                            "inconsistent cycle lift character");
                }
    auto matrix_of = [&](const Part &w, Cutoff cutoff) {
        QMat m{one, {}, {}, one};
        for (int arc : w)
            m = multiply(maps[arc], m, cutoff);
        return m;
    };
    auto raw_phase = [&](const Part &w) {
        int p = 0;
        for (int arc : w)
            if (det_sign[arc / 2] < 0)
                p += 1 + 2 * (arc % 2);
        return p;
    };
    // A closed fermion loop gives -Tr(T^n)/(2n) in log Pf(J+C).
    // Thus the SL(2) half-multiplier enters 1-u*k^(m-1).
    // Ground supercurrent two-point forms fix -u*k, including that loop sign.
    std::array<int, 6> edge_phase{};
    std::vector<Part> basis_words = g.edges == 6
                                        ? std::vector<Part>{{0, 6, 3}, {2, 8, 5}, {0, 11, 5}}
                                        : std::vector<Part>{{0, 3}, {0, 5}};
    for (int j = 0; j < int(basis.size()); j++) {
        auto w = basis_words[j];
        auto m = matrix_of(w, level);
        auto tr = add(m[0], m[3]);
        int p = raw_phase(w) + (tr.at(Key{}) < 0 ? 2 : 0);
        int d = 1;
        for (int arc : w)
            d *= det_sign[arc / 2];
        int needed =
            ((g.sign_of(basis[j]) < 0 ? 2 : 0) + 2 + tau[basis[j]] - p - (d < 0 ? 2 : 0) + 24) % 4;
        require(needed % 2 == 0, "spin lift cannot be fixed by a sign");
        edge_phase[j + (g.edges == 6 ? 3 : 1)] = needed;
    }
    QPoly fermion = one;
    int factors = 0;
    for (const auto &w : classes) {
        Key visits{}, remove{};
        for (int arc : w)
            visits[arc / 2]++;
        // The first NS oscillator has half-level monomial 3*visits.
        // Unit corrections have even half-levels, hence these ceilings.
        if (level.per_edge) {
            for (int e = 0; e < g.edges; e++)
                remove[e] = (3 * visits[e] + 1) / 2;
        } else
            remove[0] = (3 * total(visits) + 1) / 2;
        Cutoff residual = level.shifted(remove);
        auto m = matrix_of(w, residual);
        auto trace = add(m[0], m[3]);
        Rational tr0 = trace.at(Key{});
        require(tr0 == 1 || tr0 == -1, "unexpected cusp trace");
        auto inv = inverse(trace, residual), normal = scale(inv, tr0);
        int mask = 0, d = 1, phase = raw_phase(w) + (tr0 < 0 ? 2 : 0);
        for (int arc : w) {
            mask ^= 1 << (arc / 2);
            d *= det_sign[arc / 2];
            phase += edge_phase[arc / 2];
        }
        require(tau[mask] >= 0, "primitive word is not a cycle");
        int physical_phase = (phase - tau[mask] + 64) % 4;
        require(physical_phase % 2 == 0, "non-real PBW lift");
        require(g.kernel(mask, mask) == d, "determinant and cycle square disagree");
        auto inverse_squared = multiply(inv, inv, residual);
        auto ratio = multiply(QPoly{{visits, Rational(d)}}, inverse_squared, residual);
        QPoly catalan = one, p = one;
        int o = order(ratio, residual);
        for (int n = 1; n <= residual / o; n++) {
            p = multiply(p, ratio, residual);
            mpz_class c;
            mpz_bin_uiui(c.get_mpz_t(), 2 * n, n);
            c /= n + 1;
            catalan = add(catalan, scale(p, Rational(c)));
        }
        auto unit = multiply(normal, catalan, residual);
        auto multiplier_unit =
            multiply(inverse_squared, multiply(catalan, catalan, residual), residual);
        require(multiply(unit, unit, residual) == multiplier_unit,
                "lift square is not the Schottky multiplier");
        auto k = embed_twice(multiply(QPoly{{visits, Rational(d)}}, multiplier_unit, level));
        QPoly u;
        for (auto [n, x] : unit) {
            Key a = visits;
            for (int e = 0; e < g.edges; e++)
                a[e] += 2 * n[e];
            if (twice.contains(a))
                u[a] = Rational(physical_phase == 0 ? 1 : -1) * x;
        }
        QPoly kp = one;
        for (int mode = 2; int(w.size()) * (2 * mode - 1) <= 2 * level; mode++) {
            kp = multiply(kp, k, twice);
            auto term = multiply(u, kp, twice);
            fermion = star_q(fermion, add(one, scale(term, Rational(-1))), g, twice);
            factors++;
        }
    }
    auto bosonic = embed_twice(schottky(g, level));
    auto result = multiply(bosonic, fermion, twice);
    std::cerr << "NS Schottky: " << classes.size() << " primitives, " << factors
              << " fermion oscillator factors, " << result.size() << " coefficients\n";
    return result;
}

} // namespace scblocks
