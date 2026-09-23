// Directed check of the primitive-cycle remainder cutoff against the untrimmed product.
#include "scblocks/graph_ccy.hpp"
#include <iostream>
using namespace scblocks;
inline QPoly schottky_untrimmed(const Graph &g, Cutoff cutoff) {
    auto one = constant(1);
    std::array<QMat, 3> coordinate{QMat{QPoly{}, one, one, QPoly{}},
                                   QMat{one, constant(-1), {}, one}, QMat{one, {}, {}, one}};
    std::array<QMat, 3> back = coordinate;
    back[1] = {one, one, {}, one};
    std::vector<QMat> maps;
    for (int e = 0; e < g.edges; e++)
        for (int d = 0; d < 2; d++) {
            Key k{};
            k[e] = 1;
            QPoly q{{k, Rational(1)}};
            QMat inv{QPoly{}, q, one, QPoly{}};
            maps.push_back(multiply(back[g.ends[e][1 - d] % 3],
                                    multiply(inv, coordinate[g.ends[e][d] % 3], cutoff), cutoff));
        }
    std::set<Part> classes;
    Part word;
    std::function<void(int, int)> visit = [&](int start, int vertex) {
        int n = word.size();
        if (n && vertex == start && word.back() != (word.front() ^ 1)) {
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
        if (n == cutoff / 2)
            return;
        for (int e = 0; e < g.edges; e++)
            for (int d = 0; d < 2; d++)
                if (g.ends[e][d] / 3 == vertex) {
                    int arc = 2 * e + d;
                    if (n && arc == (word.back() ^ 1))
                        continue;
                    word.push_back(arc);
                    Key visits{};
                    for (int a : word)
                        visits[a / 2] += 2;
                    if (cutoff.contains(visits))
                        visit(start, g.ends[e][1 - d] / 3);
                    word.pop_back();
                }
    };
    for (int v = 0; v < int(g.slots.size()); v++)
        visit(v, v);
    QPoly logarithm;
    for (auto w : classes) {
        QMat m{one, {}, {}, one};
        QPoly det = one;
        for (int arc : w) {
            m = multiply(maps[arc], m, cutoff);
            Key k{};
            k[arc / 2] = 1;
            int from = g.ends[arc / 2][arc % 2] % 3, to = g.ends[arc / 2][1 - arc % 2] % 3;
            int ds = sign(1 + (from == 0) + (to == 0));
            det = multiply(det, QPoly{{k, Rational(ds)}}, cutoff);
        }
        auto inv = inverse(add(m[0], m[3]), cutoff),
             ratio = multiply(det, multiply(inv, inv, cutoff), cutoff);
        int o = order(ratio, cutoff);
        QPoly multiplier, p = one;
        for (int j = 1; j <= cutoff / o; j++) {
            p = multiply(p, ratio, cutoff);
            mpz_class c;
            mpz_bin_uiui(c.get_mpz_t(), 2 * j, j);
            c /= j + 1;
            multiplier = add(multiplier, scale(p, Rational(c)));
        }
        p = multiplier;
        for (int s = 2; s <= cutoff / o; s++) {
            p = multiply(p, multiplier, cutoff);
            int d = 0;
            for (int a = 2; a <= s; a++)
                if (s % a == 0)
                    d += a;
            logarithm = add(logarithm, scale(p, Rational(d, s)));
        }
    }
    auto out = one, p = one;
    for (int j = 1; j <= cutoff / order(logarithm, cutoff); j++) {
        p = scale(multiply(p, logarithm, cutoff), Rational(1, j));
        out = add(out, p);
    }
    return out;
}
int main() {
    try {
        Graph theta(3, {{0, 1, 2}, {0, 1, 2}}, {{0, 3}, {1, 4}, {2, 5}});
        Graph glasses(3, {{0, 1, 1}, {0, 2, 2}}, {{0, 3}, {1, 2}, {4, 5}});
        Graph tetrahedron(6, {{0, 1, 2}, {0, 3, 5}, {1, 4, 3}, {2, 5, 4}},
                          {{0, 3}, {1, 6}, {2, 9}, {4, 8}, {7, 11}, {10, 5}});
        size_t coefficients = 0;
        for (auto item :
             std::vector<std::pair<Graph, Cutoff>>{{theta, Cutoff(5, true, 3)},
                                                   {glasses, Cutoff(5, true, 3)},
                                                   {tetrahedron, Cutoff(10, false, 6)}}) {
            auto expected = schottky_untrimmed(item.first, item.second),
                 actual = schottky(item.first, item.second);
            require(actual == expected,
                    "trimmed Schottky expansion differs from untrimmed primitive product");
            coefficients += actual.size();
        }
        std::cout << "{\"passed\":true,\"arithmetic\":\"exact "
                     "rational\",\"domains\":3,\"nonzero_coefficients\":"
                  << coefficients << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
