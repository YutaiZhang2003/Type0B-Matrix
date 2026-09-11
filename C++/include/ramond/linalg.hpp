#pragma once
#include "storage.hpp"
#include <set>
extern "C" {
void zgelss_(int *, int *, int *, std::complex<double> *, int *, std::complex<double> *, int *,
             double *, double *, int *, std::complex<double> *, int *, double *, int *);
void zgeqp3_(int *, int *, std::complex<double> *, int *, int *, std::complex<double> *,
             std::complex<double> *, int *, double *, int *);
void zgetrf_(int *, int *, std::complex<double> *, int *, int *, int *);
void zgetrs_(char *, int *, int *, std::complex<double> *, int *, int *, std::complex<double> *,
             int *, int *);
}
namespace ramond {
// LAPACK selects a full-rank restriction and supplies its reusable LU
// preconditioner. MP solves evaluate and refine residuals using the original
// multiprecision columns, then check every row, including the unselected rows.
template <class S> using Sparse = std::unordered_map<uint32_t, S>;
struct Fit {
    int rows = 0, columns = 0, rank = 0, iterations = 0;
    double relative_residual = 0, smallest_singular = 0;
};
inline std::vector<Machine> least_squares(std::vector<Machine> a, int m, int n,
                                          std::vector<Machine> b, Fit &fit, double rcond = 1e-11) {
    int one = 1, ldb = std::max(m, n), lwork = -1, info = 0, rank = 0;
    b.resize(ldb);
    std::vector<double> sv(std::min(m, n)), rw(5 * std::min(m, n));
    Machine query;
    zgelss_(&m, &n, &one, a.data(), &m, b.data(), &ldb, sv.data(), &rcond, &rank, &query, &lwork,
            rw.data(), &info);
    require(!info, "LAPACK least-squares workspace failure");
    lwork = static_cast<int>(query.real());
    std::vector<Machine> work(lwork);
    zgelss_(&m, &n, &one, a.data(), &m, b.data(), &ldb, sv.data(), &rcond, &rank, work.data(),
            &lwork, rw.data(), &info);
    require(!info, "LAPACK least-squares failure");
    fit.rank = rank;
    fit.smallest_singular = sv.back();
    b.resize(n);
    return b;
}
template <class S> class LinearSystem {
    std::vector<Sparse<S>> columns_;
    std::vector<uint32_t> keys_;
    std::unordered_map<uint32_t, int> row_;
    std::vector<S> norms_;
    std::vector<Machine> shadow_, lu_;
    std::vector<int> selected_, pivots_;
    bool restrict_;

  public:
    Fit fit;
    LinearSystem(std::vector<Sparse<S>> columns, const std::vector<uint32_t> &extra = {},
                 bool restriction = false)
        : columns_(std::move(columns)), restrict_(restriction || std::is_same_v<S, MP>) {
        std::set<uint32_t> keys(extra.begin(), extra.end());
        for (const auto &col : columns_)
            for (const auto &[key, v] : col)
                if (v != S(0))
                    keys.insert(key);
        keys_.assign(keys.begin(), keys.end());
        int m = keys_.size(), n = columns_.size();
        fit.rows = m;
        fit.columns = n;
        require(m >= n && n > 0, "invalid span dimensions");
        for (int i = 0; i < m; i++)
            row_[keys_[i]] = i;
        norms_.resize(n);
        shadow_.resize(static_cast<size_t>(m) * n);
        for (int j = 0; j < n; j++) {
            S norm(0);
            for (const auto &[key, v] : columns_[j])
                norm += conjugate(v) * v;
            norms_[j] = root(real_number(norm));
            require(norms_[j] != S(0), "vanishing span column");
            for (const auto &[key, v] : columns_[j])
                if (v != S(0))
                    shadow_[row_.at(key) + static_cast<size_t>(m) * j] = machine(v / norms_[j]);
        }
        if (restrict_) {
            std::vector<Machine> a(static_cast<size_t>(m) * n), tau(n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    a[j + static_cast<size_t>(n) * i] = shadow_[i + static_cast<size_t>(m) * j];
            std::vector<int> p(m);
            std::vector<double> rw(2 * m);
            int info = 0, lwork = -1;
            Machine query;
            zgeqp3_(&n, &m, a.data(), &n, p.data(), tau.data(), &query, &lwork, rw.data(), &info);
            require(!info, "QR workspace failure");
            lwork = static_cast<int>(query.real());
            std::vector<Machine> work(lwork);
            zgeqp3_(&n, &m, a.data(), &n, p.data(), tau.data(), work.data(), &lwork, rw.data(),
                    &info);
            require(!info, "pivoted QR failure");
            selected_.resize(n);
            lu_.resize(n * n);
            for (int i = 0; i < n; i++) {
                selected_[i] = p[i] - 1;
                for (int j = 0; j < n; j++)
                    lu_[i + n * j] = shadow_[selected_[i] + static_cast<size_t>(m) * j];
            }
            Fit rank_fit;
            least_squares(lu_, n, n, std::vector<Machine>(n), rank_fit);
            fit.rank = rank_fit.rank;
            fit.smallest_singular = rank_fit.smallest_singular;
            require(fit.rank == n && fit.smallest_singular > 1e-11, "pivoted span lost full rank");
            pivots_.resize(n);
            zgetrf_(&n, &n, lu_.data(), &n, pivots_.data(), &info);
            require(!info, "singular pivoted span");
            shadow_.clear();
            shadow_.shrink_to_fit();
        }
    }
    std::vector<S> solve(const Sparse<S> &target) {
        int m = keys_.size(), n = columns_.size();
        for (const auto &[k, v] : target)
            require(v == S(0) || row_.count(k), "target contains a row absent from the span");
        std::vector<S> values(n);
        fit.iterations = 0;
        if (!restrict_) {
            std::vector<Machine> b(m);
            for (int i = 0; i < m; i++)
                b[i] = machine(get(target, keys_[i]));
            auto x = least_squares(shadow_, m, n, std::move(b), fit);
            require(fit.rank == n, "native span lost full column rank");
            for (int j = 0; j < n; j++)
                values[j] = S(x[j]) / norms_[j];
        } else {
            auto lu_solve = [&](std::vector<Machine> &v) {
                char trans = 'N';
                int one = 1, info = 0;
                zgetrs_(&trans, &n, &one, lu_.data(), &n, pivots_.data(), v.data(), &n, &info);
                require(!info, "LU solve failure");
            };
            std::vector<Machine> b(n);
            S scale(0);
            for (int i = 0; i < n; i++) {
                S v = get(target, keys_[selected_[i]]);
                b[i] = machine(v);
                scale += conjugate(v) * v;
            }
            scale = root(real_number(scale));
            if (magnitude(scale) < 1)
                scale = S(1);
            lu_solve(b);
            for (int j = 0; j < n; j++)
                values[j] = S(b[j]);
            for (int iteration = 0; iteration < 30; iteration++) {
                S err(0);
                for (int i = 0; i < n; i++) {
                    auto key = keys_[selected_[i]];
                    S r = get(target, key);
                    for (int j = 0; j < n; j++)
                        r -= get(columns_[j], key) / norms_[j] * values[j];
                    b[i] = machine(r);
                    err += conjugate(r) * r;
                }
                if (small(root(real_number(err)) / scale, 1e-12, -std::max(20, digits<S>() - 15)))
                    break;
                require(iteration < 29, "multiprecision refinement did not converge");
                lu_solve(b);
                for (int j = 0; j < n; j++)
                    values[j] += S(b[j]);
                fit.iterations++;
            }
            for (int j = 0; j < n; j++)
                values[j] /= norms_[j];
        }
        Sparse<S> residual;
        for (const auto &[key, v] : target)
            residual[key] = -v;
        for (int j = 0; j < n; j++)
            for (const auto &[key, v] : columns_[j])
                residual[key] += v * values[j];
        S norm(0), error(0);
        for (const auto &[key, v] : target)
            norm += conjugate(v) * v;
        for (const auto &[key, v] : residual)
            error += conjugate(v) * v;
        S rel = root(real_number(error));
        if (norm != S(0))
            rel /= root(real_number(norm));
        fit.relative_residual = magnitude(rel);
        require(finite(rel) && small(rel, 1e-8, -std::max(15, digits<S>() - 20)),
                "full-row span residual exceeds tolerance: " +
                    std::to_string(fit.relative_residual));
        return values;
    }
};
template <class S>
std::vector<S> span_solve(const Sparse<S> &target, std::vector<Sparse<S>> columns, Fit &fit) {
    std::vector<uint32_t> extra;
    for (const auto &[k, v] : target)
        if (v != S(0))
            extra.push_back(k);
    LinearSystem<S> system(std::move(columns), extra);
    auto answer = system.solve(target);
    fit = system.fit;
    return answer;
}
} // namespace ramond
