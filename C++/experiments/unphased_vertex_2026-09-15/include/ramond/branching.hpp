#pragma once
#include "anchors.hpp"
namespace ramond {
// Internal integer labels are 4*n; external conventions and normalization
// coincide with the Python branching pipeline and the human notes.
inline int ramond_level(int n4) {
    require(n4 % 2 != 0, "Ramond labels must be odd quarter-integers");
    return (n4 * n4 - 1) / 8;
}
inline std::vector<int> ns_labels(int level) {
    int bound = static_cast<int>(std::sqrt(2 * level));
    std::vector<int> out;
    for (int n = -bound; n <= bound; n++)
        out.push_back(2 * n);
    return out;
}
inline std::vector<std::pair<int, int>> middle_pairs(int level, bool box = false) {
    int bound = 1;
    while (ramond_level(bound) <= 2 * level + 2)
        bound += 2;
    std::vector<std::pair<int, int>> out;
    for (int n = -bound; n <= bound; n += 2)
        for (int changed : {n - 2, n + 2})
            if (box ? std::max(ramond_level(n), ramond_level(changed)) <= level
                    : ramond_level(n) + ramond_level(changed) <= 2 * level)
                out.emplace_back(n, changed);
    return out;
}
template <class S> class RamondActions {
    S b_, p_;
    bool precompute_;
    std::array<std::unique_ptr<FreeField<S>>, 2> modules_;
    std::map<int, ActionPair<S>> pairs_;
    std::map<int, std::vector<ActionTerm<S>>> zeros_;
    std::map<std::array<int, 3>, std::vector<ActionTerm<S>>> parities_;

  public:
    double maximum_residual = 0;
    RamondActions(S b, S p, bool precompute) : b_(b), p_(p), precompute_(precompute) {}
    const std::vector<ActionTerm<S>> &get(int n4, int parity, bool plus = false) {
        std::array<int, 3> key{n4, parity, int(plus)};
        auto old = parities_.find(key);
        if (old != parities_.end())
            return old->second;
        if (plus && std::abs(n4) == 1)
            return parities_[key];
        if (!pairs_.count(n4)) {
            int reflected = n4 < 0;
            if (!modules_[reflected])
                modules_[reflected] =
                    std::make_unique<FreeField<S>>(true, b_, reflected ? -p_ : p_);
            ActionPair<S> pair;
            try {
                pair = ramond_actions(*modules_[reflected], std::abs(n4), precompute_ || plus);
            } catch (const std::exception &e) {
                throw std::runtime_error("Ramond mode action n4=" + std::to_string(n4) +
                                         ": " + e.what());
            }
            maximum_residual = std::max({maximum_residual, pair.minus_fit.relative_residual,
                                         pair.plus_fit.relative_residual});
            if (reflected) {
                for (auto &t : pair.minus)
                    t.label = -t.label;
                for (auto &t : pair.plus)
                    t.label = -t.label;
            }
            pairs_[n4] = std::move(pair);
        } else if (plus && pairs_.at(n4).plus.empty()) {
            int reflected = n4 < 0;
            Fit fit;
            auto terms = ramond_nonnegative_action(*modules_[reflected], std::abs(n4), 1, fit);
            maximum_residual = std::max(maximum_residual, fit.relative_residual);
            if (reflected)
                for (auto &t : terms)
                    t.label = -t.label;
            pairs_[n4].plus = std::move(terms);
        }
        auto terms = plus ? pairs_.at(n4).plus : pairs_.at(n4).minus;
        if (parity) {
            int source = sign((std::abs(n4) - 1) / 2);
            for (auto &t : terms)
                t.coefficient *= power(S(2), (sign((std::abs(t.label) - 1) / 2) - source) / 2);
        }
        return parities_.emplace(key, std::move(terms)).first->second;
    }
    const std::vector<ActionTerm<S>> &zero(int n4, int parity) {
        std::array<int, 3> key{n4, parity, 2};
        auto old = parities_.find(key);
        if (old != parities_.end())
            return old->second;
        if (!zeros_.count(n4)) {
            int reflected = n4 < 0;
            if (!modules_[reflected])
                modules_[reflected] =
                    std::make_unique<FreeField<S>>(true, b_, reflected ? -p_ : p_);
            Fit fit;
            auto terms = ramond_nonnegative_action(*modules_[reflected], std::abs(n4), 0, fit);
            maximum_residual = std::max(maximum_residual, fit.relative_residual);
            if (reflected)
                for (auto &t : terms)
                    t.label = -t.label;
            zeros_[n4] = std::move(terms);
        }
        auto terms = zeros_.at(n4);
        if (parity) {
            int source = sign((std::abs(n4) - 1) / 2);
            for (auto &t : terms)
                t.coefficient *= power(S(2), (sign((std::abs(t.label) - 1) / 2) - source) / 2);
        }
        return parities_.emplace(key, std::move(terms)).first->second;
    }
};
template <class S> class OuterBranching {
    using Triple = std::array<int, 3>;
    S b_;
    std::array<S, 3> p_;
    int level_, f_, primary_parity_;
    LowAnchors<S> anchors_;
    std::map<int, std::vector<ActionTerm<S>>> ns_actions_;
    std::map<int, std::vector<ActionTerm<S>>> ns_minus_;
    std::array<std::unique_ptr<FreeField<S>>, 2> ns_modules_;
    std::set<Triple> support_;
    std::map<std::tuple<Triple, int, int, Part>, S> vertex_cache_, vertex_bounds_;
    std::map<std::array<int, 3>, S> weights_;
    struct WardRow {
        std::map<Triple, S> coefficients, bounds;
    };
    std::map<std::array<int, 6>, WardRow> row_cache_;
    std::map<std::array<int, 3>, std::map<Triple, S>> tables_;
    static bool boundary(Triple n) {
        return std::abs(n[0]) <= 2 && std::abs(n[1]) <= 3 && std::abs(n[2]) <= 3;
    }
    static int height(Triple n) {
        return std::abs(n[0]) / 4 + (std::abs(n[1]) - 1) / 4 +
               (std::abs(n[2]) - 1) / 4;
    }
    static Triple group_key(Triple n) {
        // Only the explicit boundary actions can preserve this height.
        // NS +/-1/2 are paired by the second identity; R 1/4 and -3/4
        // (and their reflections) are paired by L_{-1}.
        if (std::abs(n[0]) == 2)
            n[0] = 2;
        for (int j = 1; j < 3; j++)
            if (std::abs(n[j]) <= 3)
                n[j] = (n[j] == 1 || n[j] == -3) ? 1 : -1;
        return n;
    }
    static bool zero_pivot(const S &value, const S &bound) {
        if (bound == S(0))
            return value == S(0);
        return small(value / bound, 1e-11, -std::max(15, digits<S>() - 20));
    }
    static std::string label_text(Triple n) {
        return "(4n1,4n2,4n3)=(" + std::to_string(n[0]) + "," +
               std::to_string(n[1]) + "," + std::to_string(n[2]) + ")";
    }
    const std::vector<ActionTerm<S>> &ns_minus(int n4) {
        auto old = ns_minus_.find(n4);
        if (old != ns_minus_.end())
            return old->second;
        int reflected = n4 < 0;
        if (!ns_modules_[reflected])
            ns_modules_[reflected] =
                std::make_unique<FreeField<S>>(false, b_, reflected ? -p_[0] : p_[0]);
        Fit fit;
        auto terms = ns_minus_action(*ns_modules_[reflected], std::abs(n4), fit);
        maximum_action_residual = std::max(maximum_action_residual, fit.relative_residual);
        if (reflected)
            for (auto &t : terms)
                t.label = -t.label;
        return ns_minus_.emplace(n4, std::move(terms)).first->second;
    }
    const S &weight(int copy, int slot, int label) {
        std::array<int, 3> key{copy, slot, label};
        auto it = weights_.find(key);
        if (it == weights_.end())
            it = weights_.emplace(key, branch_weight(copy, label, b_, p_[slot])).first;
        return it->second;
    }
    std::pair<Triple, S> factor(Triple labels, int slot, const ActionTerm<S> &term,
                              S *bound = nullptr) {
        labels[slot] = term.label;
        S v = term.coefficient;
        if (bound)
            *bound = abs_number(v);
        for (int copy = 0; copy < 2; copy++) {
            const Part &word = copy ? term.second : term.first;
            if (word.empty())
                continue;
            auto key = std::make_tuple(labels, slot, copy, word);
            auto found = vertex_cache_.find(key);
            if (found == vertex_cache_.end()) {
                std::array<S, 3> h;
                for (int j = 0; j < 3; j++)
                    h[j] = weight(copy, j, labels[j]);
                found = vertex_cache_.emplace(key, vertex_word(slot, word, h)).first;
            }
            v *= found->second;
            if (bound) {
                auto old = vertex_bounds_.find(key);
                if (old == vertex_bounds_.end()) {
                    std::array<S, 3> h;
                    for (int j = 0; j < 3; j++)
                        h[j] = abs_number(weight(copy, j, labels[j]));
                    S product(1);
                    int level = 0;
                    for (auto it = word.rbegin(); it != word.rend(); ++it) {
                        product *= h[slot] + S(level) + S(*it) * h[slot == 1 ? 2 : 1] +
                                   h[slot == 0 ? 2 : 0];
                        level += *it;
                    }
                    old = vertex_bounds_.emplace(key, product).first;
                }
                *bound *= old->second;
            }
        }
        return {labels, v};
    }
    const WardRow &ward_row(Triple labels, int alpha, int gamma, bool second = false) {
        std::array<int, 6> key{labels[0], labels[1], labels[2], alpha, gamma, int(second)};
        auto old = row_cache_.find(key);
        if (old != row_cache_.end())
            return old->second;
        WardRow row;
        auto add_terms = [&](int slot, const std::vector<ActionTerm<S>> &terms, int scale) {
            for (const auto &t : terms) {
                Triple changed = labels;
                changed[slot] = t.label;
                bool local = group_key(changed) == group_key(labels);
                S bound(0);
                auto term = factor(labels, slot, t, local ? &bound : nullptr);
                row.coefficients[changed] += S(scale) * term.second;
                if (local)
                    row.bounds[changed] += S(std::abs(scale)) * bound;
            }
        };
        if (!second) {
            add_terms(0, ns_actions_.at(labels[0]), 1);
            add_terms(1, actions[0]->get(labels[1], alpha), -1);
            add_terms(2, actions[1]->get(labels[2], gamma), -1);
        } else {
            // Human Notes/SCblock.tex: physical L1 in the second (Ramond) slot.
            add_terms(0, ns_minus(labels[0]), 1);
            add_terms(1, actions[0]->get(labels[1], alpha), -1);
            add_terms(1, actions[0]->zero(labels[1], alpha), -2);
            add_terms(1, actions[0]->get(labels[1], alpha, true), -1);
            add_terms(2, actions[1]->get(labels[2], gamma, true), -1);
            for (const auto &a : actions)
                maximum_action_residual = std::max(maximum_action_residual, a->maximum_residual);
        }
        return row_cache_.emplace(key, std::move(row)).first->second;
    }
    // After substituting cached lower-height values, the bulk case is one
    // scalar division. Only boundary actions couple a group (at most four
    // unanchored coefficients). Elimination stays in S; there is no LAPACK
    // rank decision or large global matrix in this branching recursion.
    bool advance(const std::vector<Triple> &nodes, const std::vector<const WardRow *> &rows,
                 int alpha, int gamma, const std::vector<int> &etas) {
        const int n = nodes.size(), m = rows.size(), rhs_count = etas.size();
        std::map<Triple, int> index;
        for (int j = 0; j < n; j++)
            index[nodes[j]] = j;
        std::vector<std::vector<S>> matrix(m, std::vector<S>(n + rhs_count));
        for (int i = 0; i < m; i++) {
            S scale(0);
            for (const auto &[changed, coefficient] : rows[i]->coefficients) {
                auto column = index.find(changed);
                if (column != index.end()) {
                    const S &bound = rows[i]->bounds.at(changed);
                    if (!zero_pivot(coefficient, bound)) {
                        matrix[i][column->second] = coefficient;
                        scale += abs_number(coefficient);
                    }
                } else {
                    require(height(changed) < height(nodes.front()),
                            "nondecreasing branching dependency: " + label_text(nodes.front()) +
                                " -> " + label_text(changed));
                    for (int r = 0; r < rhs_count; r++) {
                        const auto &cache = tables_.at({alpha, gamma, etas[r]});
                        auto known = cache.find(changed);
                        require(known != cache.end(), "missing lower branching coefficient: " +
                                                         label_text(changed));
                        matrix[i][n + r] -= coefficient * known->second;
                    }
                }
            }
            // The pivot is scaled only against other unknowns in this group,
            // never against large already-known lower-level contributions.
            if (scale != S(0))
                for (auto &v : matrix[i])
                    v /= scale;
        }
        std::vector<int> columns(n);
        for (int j = 0; j < n; j++)
            columns[j] = j;
        for (int k = 0; k < n; k++) {
            int pi = -1, pj = -1;
            double largest = 0;
            for (int i = k; i < m; i++)
                for (int j = k; j < n; j++)
                    if (magnitude(matrix[i][j]) > largest) {
                        largest = magnitude(matrix[i][j]);
                        pi = i;
                        pj = j;
                    }
            if (pi < 0 || zero_pivot(matrix[pi][pj], S(1)))
                return false;
            std::swap(matrix[k], matrix[pi]);
            for (auto &row : matrix)
                std::swap(row[k], row[pj]);
            std::swap(columns[k], columns[pj]);
            S pivot = matrix[k][k];
            for (int j = k; j < n + rhs_count; j++)
                matrix[k][j] /= pivot;
            for (int i = k + 1; i < m; i++) {
                S factor = matrix[i][k];
                for (int j = k; j < n + rhs_count; j++)
                    matrix[i][j] -= factor * matrix[k][j];
            }
        }
        for (int r = 0; r < rhs_count; r++) {
            std::vector<S> values(n);
            for (int i = n - 1; i >= 0; i--) {
                values[i] = matrix[i][n + r];
                for (int j = i + 1; j < n; j++)
                    values[i] -= matrix[i][j] * values[j];
            }
            auto &cache = tables_.at({alpha, gamma, etas[r]});
            for (int j = 0; j < n; j++) {
                require(finite(values[j]), "nonfinite branching coefficient: " +
                                               label_text(nodes[columns[j]]));
                cache[nodes[columns[j]]] = values[j];
                recursive_values++;
            }
        }
        return true;
    }
    double row_residual(const WardRow &row, int alpha, int gamma, int eta) const {
        S residual(0), scale(1);
        for (const auto &[changed, coefficient] : row.coefficients) {
            S term = coefficient * raw(changed[0], changed[1], changed[2], alpha, gamma, eta);
            residual += term;
            scale += abs_number(term);
        }
        return magnitude(residual / scale);
    }

  public:
    std::vector<int> ns, r;
    std::array<std::unique_ptr<RamondActions<S>>, 2> actions;
    double action_seconds = 0, ward_seconds = 0, maximum_action_residual = 0,
           maximum_ward_residual = 0;
    int second_ward_rows = 0, vanishing_first_pivots = 0;
    int recursive_values = 0, recursive_groups = 0, maximum_local_unknowns = 0,
        direct_boundary_values = 0;
    OuterBranching(S b, std::array<S, 3> p, int level, int f, int primary_parity, bool inserted,
                   bool box = false)
        : b_(b), p_(p), level_(level), f_(f), primary_parity_(primary_parity),
          anchors_(b, p, primary_parity) {
        double start = seconds();
        ns = ns_labels(level);
        auto pairs = middle_pairs(level, box);
        int limit = 3;
        if (inserted) {
            for (auto [a, b] : pairs)
                limit = std::max({limit, std::abs(a), std::abs(b)});
        } else {
            while (ramond_level(limit + 2) <= level)
                limit += 2;
        }
        for (int n = -limit; n <= limit; n += 2)
            r.push_back(n);
        for (int n : ns) {
            if (std::abs(n) < 4) {
                ns_actions_[n] = {};
                continue;
            }
            int reflected = n < 0;
            if (!ns_modules_[reflected])
                ns_modules_[reflected] =
                    std::make_unique<FreeField<S>>(false, b_, reflected ? -p_[0] : p_[0]);
            Fit fit;
            std::vector<ActionTerm<S>> terms;
            try {
                terms = ns_action(*ns_modules_[reflected], std::abs(n), fit);
            } catch (const std::exception &e) {
                throw std::runtime_error("NS L1 action n4=" + std::to_string(n) + ": " + e.what());
            }
            maximum_action_residual = std::max(maximum_action_residual, fit.relative_residual);
            if (reflected)
                for (auto &t : terms)
                    t.label = -t.label;
            ns_actions_[n] = std::move(terms);
        }
        for (int edge = 0; edge < 2; edge++) {
            actions[edge] =
                std::make_unique<RamondActions<S>>(b_, p_[edge + 1], inserted && edge == 0);
            for (int n : r)
                for (int parity = 0; parity < 2; parity++)
                    actions[edge]->get(n, parity);
            maximum_action_residual =
                std::max(maximum_action_residual, actions[edge]->maximum_residual);
        }
        if (inserted) {
            for (int n : ns)
                for (auto [incoming, outgoing] : pairs)
                    for (int third : r)
                        if (box ? n * n / 4 <= 2 * level && ramond_level(third) <= level
                                : n * n / 4 + ramond_level(incoming) + ramond_level(outgoing) +
                                2 * ramond_level(third) <=
                            2 * level) {
                            support_.insert({n, incoming, third});
                            support_.insert({n, outgoing, third});
                        }
        } else
            for (int n : ns)
                for (int second : r)
                    for (int third : r)
                        if (box ? n * n / 4 <= 2 * level && ramond_level(second) <= level &&
                                      ramond_level(third) <= level
                                : n * n / 4 + 2 * ramond_level(second) + 2 * ramond_level(third) <=
                            2 * level)
                            support_.insert({n, second, third});
        std::vector<Triple> pending(support_.begin(), support_.end());
        while (!pending.empty()) {
            auto labels = pending.back();
            pending.pop_back();
            for (int slot = 0; slot < 3; slot++)
                for (int parity = 0; parity < (slot ? 2 : 1); parity++) {
                    const auto &terms = slot == 0 ? ns_actions_.at(labels[0])
                                                  : actions[slot - 1]->get(labels[slot], parity);
                    for (const auto &t : terms) {
                        auto changed = labels;
                        changed[slot] = t.label;
                        if (support_.insert(changed).second)
                            pending.push_back(changed);
                    }
                }
        }
        action_seconds = seconds() - start;
    }
    void prepare(int eta, int eta_prime) {
        double start = seconds();
        std::map<Triple, std::vector<Triple>> groups;
        for (auto labels : support_)
            groups[group_key(labels)].push_back(labels);
        std::vector<Triple> order;
        for (const auto &[key, nodes] : groups)
            order.push_back(key);
        std::stable_sort(order.begin(), order.end(), [](Triple a, Triple b) {
            return height(a) < height(b);
        });
        for (int a = 0; a < 2; a++)
            for (int g = 0; g < 2; g++) {
                std::vector<int> etas;
                for (int e : {eta, eta_prime})
                    if (!tables_.count({a, g, e})) {
                        tables_[{a, g, e}] = {};
                        etas.push_back(e);
                        // Exactly the directly evaluated boundary box in SCblock.tex.
                        for (int n : {-2, 0, 2})
                            if ((n / 2 + a + g - f_) % 2 == 0)
                                for (int n2 : {-3, -1, 1, 3})
                                    for (int n3 : {-3, -1, 1, 3}) {
                                        tables_.at({a, g, e})[{n, n2, n3}] =
                                            anchors_.raw({n, n2, n3}, a, g, unphased_vertex ? e : sign(f_) * e);
                                        direct_boundary_values++;
                                    }
                    }
                if (etas.empty())
                    continue;
                for (auto key : order) {
                    const auto &nodes = groups.at(key);
                    if ((nodes.front()[0] / 2 + a + g - f_) % 2 != 0)
                        continue;
                    if (boundary(nodes.front()))
                        continue;
                    require(nodes.size() <= 4, "unexpected nonboundary recursion group size");
                    maximum_local_unknowns = std::max(maximum_local_unknowns, int(nodes.size()));
                    std::vector<const WardRow *> rows;
                    for (auto labels : nodes) {
                        const auto &row = ward_row(labels, a, g);
                        rows.push_back(&row);
                        auto pivot = row.coefficients.find(labels);
                        auto bound = row.bounds.find(labels);
                        if (pivot == row.coefficients.end() || bound == row.bounds.end() ||
                            zero_pivot(pivot->second, bound->second))
                            vanishing_first_pivots++;
                    }
                    if (!advance(nodes, rows, a, g, etas)) {
                        for (auto labels : nodes) {
                            rows.push_back(&ward_row(labels, a, g, true));
                            second_ward_rows++;
                        }
                        if (!advance(nodes, rows, a, g, etas)) {
                            std::string message = "branching recursion cannot advance at alpha=" +
                                std::to_string(a) + ", gamma=" + std::to_string(g);
                            for (auto labels : nodes)
                                message += "; " + label_text(labels);
                            throw std::runtime_error(message);
                        }
                    }
                    recursive_groups++;
                    for (const auto *row : rows)
                        for (int e : etas) {
                            double residual = row_residual(*row, a, g, e);
                            maximum_ward_residual = std::max(maximum_ward_residual, residual);
                            require(std::isfinite(residual) &&
                                        residual <= (std::is_same_v<S, Machine> ? 1e-8 :
                                            std::pow(10., -std::max(15, digits<S>() - 20))),
                                    "branching Ward residual failed at " + label_text(key) +
                                        ", alpha=" + std::to_string(a) +
                                        ", gamma=" + std::to_string(g) +
                                        ", eta=" + std::to_string(e) +
                                        ", residual=" + std::to_string(residual));
                        }
                }
            }
        ward_seconds = seconds() - start;
    }
    const S &raw(int n1, int n2, int n3, int alpha, int gamma, int eta) const {
        return tables_.at({alpha, gamma, eta}).at({n1, n2, n3});
    }
    std::size_t support_size() const {
        return support_.size();
    }
    // Directed diagnostic: evaluate the second identity against solved branching
    // coefficients, retaining the cancellation scale in the denominator.
    double second_ward_residual(Triple labels, int alpha, int gamma, int eta) {
        return row_residual(ward_row(labels, alpha, gamma, true), alpha, gamma, eta);
    }
};
template <class S> class MiddleBranching {
    S b_, p_, q_;
    RamondActions<S> &actions_;
    std::map<std::array<int, 3>, S> values_;
    S factor(int out, int in, const ActionTerm<S> &term, int slot) {
        S v = term.coefficient;
        S b2 = b_ * b_;
        std::array<S, 2> external{-(S(1) + S(2) * b2) / (S(2) * (S(1) - b2)),
                                  (b2 + S(2)) / (S(2) * (S(1) - b2))};
        for (int copy = 0; copy < 2; copy++) {
            std::array<S, 3> h{branch_weight(copy, out, b_, p_), external[copy],
                               branch_weight(copy, in, b_, p_)};
            v *= vertex_word(slot, copy ? term.second : term.first, h);
        }
        return v;
    }

  public:
    MiddleBranching(S b, S p, RamondActions<S> &actions)
        : b_(b), p_(p), q_(b + S(1) / b), actions_(actions) {}
    S raw(int out, int in, int parity) {
        std::array<int, 3> key{out, in, parity};
        auto old = values_.find(key);
        if (old != values_.end())
            return old->second;
        if (std::abs(out - in) != 2)
            return S(0);
        if (std::abs(out) == 1 && std::abs(in) == 1)
            return values_[key] = q_ * root(S(2)) / power(S(2), parity);
        S numerator(0), denominator(0);
        if (std::abs(out) > std::abs(in)) {
            for (const auto &t : actions_.get(in, parity))
                if (t.label == in)
                    denominator += factor(out, in, t, 2);
                else
                    require(std::abs(out - t.label) != 2, "unexpected neighboring middle branch");
            for (const auto &t : actions_.get(out, parity, true))
                numerator += factor(t.label, in, t, 0) * raw(t.label, in, parity);
        } else {
            for (const auto &t : actions_.get(out, parity))
                if (t.label == out)
                    denominator += factor(out, in, t, 0);
                else
                    require(std::abs(t.label - in) != 2, "unexpected neighboring middle branch");
            for (const auto &t : actions_.get(in, parity, true))
                numerator += factor(out, t.label, t, 2) * raw(out, t.label, parity);
        }
        require(!arithmetic_zero(denominator),
                "middle L1 Ward pivot vanished; generic parameters required");
        return values_[key] = numerator / denominator;
    }
};
} // namespace ramond
