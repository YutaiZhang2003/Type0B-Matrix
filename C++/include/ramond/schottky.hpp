#pragma once
#include "ccy.hpp"
#include "fermion.hpp"
namespace ramond {
using QSeries = Series<Rational>;
inline QSeries qconstant(const Rational &v) {
    QSeries p;
    if (v != 0)
        p[Index{}] = v;
    return p;
}
inline QSeries qadd(QSeries a, const QSeries &b) {
    for (const auto &[k, v] : b) {
        a[k] += v;
        if (a[k] == 0)
            a.erase(k);
    }
    return a;
}
inline QSeries qscale(QSeries a, const Rational &v) {
    if (v == 0)
        return {};
    for (auto &[k, x] : a)
        x *= v;
    return a;
}
inline QSeries qmultiply(const QSeries &a, const QSeries &b, int cutoff) {
    auto r = scalar_product(a, b, cutoff);
    for (auto it = r.begin(); it != r.end();)
        if (it->second == 0)
            it = r.erase(it);
        else
            ++it;
    return r;
}
inline int qorder(const QSeries &a, int cutoff) {
    int order = cutoff + 1;
    for (const auto &[k, v] : a)
        if (v != 0)
            order = std::min(order, degree(k));
    return order;
}
inline QSeries qinverse(const QSeries &a, int cutoff) {
    Rational c = get(a, Index{});
    require(c != 0, "Schottky cycle has nonunit trace at the cusp");
    auto one = qconstant(1), rem = qadd(one, qscale(a, -1 / c));
    auto result = one, p = one;
    for (int j = 0; j < cutoff / qorder(rem, cutoff); j++) {
        p = qmultiply(p, rem, cutoff);
        result = qadd(std::move(result), p);
    }
    return qscale(std::move(result), 1 / c);
}
using QMatrix = std::array<QSeries, 4>;
inline QMatrix qmatrix(const QMatrix &a, const QMatrix &b, int cutoff) {
    return {qadd(qmultiply(a[0], b[0], cutoff), qmultiply(a[1], b[2], cutoff)),
            qadd(qmultiply(a[0], b[1], cutoff), qmultiply(a[1], b[3], cutoff)),
            qadd(qmultiply(a[2], b[0], cutoff), qmultiply(a[3], b[2], cutoff)),
            qadd(qmultiply(a[2], b[1], cutoff), qmultiply(a[3], b[3], cutoff))};
}
inline void cycles_visit(int start, int vertex, Part &word, int maximum, std::set<Part> &classes) {
    if (!word.empty() && vertex == start && word.back() != (word.front() ^ 1)) {
        bool primitive = true;
        int n = word.size();
        for (int p = 1; p < n; p++)
            if (n % p == 0) {
                bool same = true;
                for (int j = 0; j < n; j++)
                    if (word[j] != word[j % p])
                        same = false;
                if (same) {
                    primitive = false;
                    break;
                }
            }
        if (primitive) {
            Part inverse;
            for (auto it = word.rbegin(); it != word.rend(); ++it)
                inverse.push_back(*it ^ 1);
            Part best = word;
            for (const auto &w : {word, inverse})
                for (int j = 0; j < n; j++) {
                    Part rotated(w.begin() + j, w.end());
                    rotated.insert(rotated.end(), w.begin(), w.begin() + j);
                    best = std::min(best, rotated);
                }
            classes.insert(std::move(best));
        }
    }
    if (int(word.size()) == maximum)
        return;
    for (int edge = 0; edge < 3; edge++) {
        int arc = 2 * edge + vertex;
        if (!word.empty() && arc == (word.back() ^ 1))
            continue;
        word.push_back(arc);
        cycles_visit(start, 1 - vertex, word, maximum, classes);
        word.pop_back();
    }
}
inline QSeries schottky_vacuum(int cutoff) {
    auto one = qconstant(1);
    if (cutoff < 4)
        return one;
    std::array<QSeries, 3> q;
    for (int j = 0; j < 3; j++) {
        Index key{};
        key[j] = 1;
        q[j][key] = 1;
    }
    std::array<QMatrix, 3> maps{QMatrix{QSeries{}, one, q[0], QSeries{}},
                                QMatrix{one, qadd(q[1], qconstant(-1)), one, qconstant(-1)},
                                QMatrix{QSeries{}, q[2], one, QSeries{}}};
    std::set<Part> classes;
    Part word;
    for (int vertex = 0; vertex < 2; vertex++)
        cycles_visit(vertex, vertex, word, cutoff / 2, classes);
    QSeries logarithm;
    for (const auto &w : classes) {
        QMatrix matrix{one, {}, {}, one};
        QSeries det = one;
        for (int arc : w) {
            matrix = qmatrix(maps[arc / 2], matrix, cutoff);
            det = qmultiply(det, qscale(q[arc / 2], -1), cutoff);
        }
        auto inv = qinverse(qadd(matrix[0], matrix[3]), cutoff);
        auto ratio = qmultiply(det, qmultiply(inv, inv, cutoff), cutoff);
        int order = qorder(ratio, cutoff);
        QSeries multiplier, p = one;
        for (int j = 1; j <= cutoff / order; j++) {
            p = qmultiply(p, ratio, cutoff);
            mpz_class catalan;
            mpz_bin_uiui(catalan.get_mpz_t(), 2 * j, j);
            catalan /= (j + 1);
            multiplier = qadd(std::move(multiplier), qscale(p, Rational(catalan)));
        }
        p = multiplier;
        for (int s = 2; s <= cutoff / order; s++) {
            p = qmultiply(p, multiplier, cutoff);
            int divisor_sum = 0;
            for (int m = 2; m <= s; m++)
                if (s % m == 0)
                    divisor_sum += m;
            logarithm = qadd(std::move(logarithm), qscale(p, Rational(divisor_sum, s)));
        }
    }
    QSeries result = one, p = one;
    int order = qorder(logarithm, cutoff);
    for (int j = 1; j <= cutoff / order; j++) {
        p = qscale(qmultiply(p, logarithm, cutoff), Rational(1, j));
        result = qadd(std::move(result), p);
    }
    return result;
}
} // namespace ramond
