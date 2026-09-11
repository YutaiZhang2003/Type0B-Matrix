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
inline std::vector<std::pair<int, int>> middle_pairs(int level) {
    int bound = 1;
    while (ramond_level(bound) <= 2 * level + 2)
        bound += 2;
    std::vector<std::pair<int, int>> out;
    for (int n = -bound; n <= bound; n += 2)
        for (int changed : {n - 2, n + 2})
            if (ramond_level(n) + ramond_level(changed) <= 2 * level)
                out.emplace_back(n, changed);
    return out;
}
template <class S> class RamondActions {
    S b_, p_;
    bool precompute_;
    std::array<std::unique_ptr<FreeField<S>>, 2> modules_;
    std::map<int, ActionPair<S>> pairs_;
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
            auto pair = ramond_actions(*modules_[reflected], std::abs(n4), precompute_ || plus);
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
            auto pair = ramond_actions(*modules_[reflected], std::abs(n4), true);
            if (reflected)
                for (auto &t : pair.plus)
                    t.label = -t.label;
            pairs_[n4].plus = std::move(pair.plus);
        }
        auto terms = plus ? pairs_.at(n4).plus : pairs_.at(n4).minus;
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
    std::set<Triple> support_;
    std::map<std::tuple<Triple, int, int, Part>, S> vertex_cache_;
    struct System {
        std::vector<Triple> unknowns, anchors;
        int ward_rows;
        std::unique_ptr<LinearSystem<S>> matrix;
    };
    std::map<std::pair<int, int>, System> systems_;
    std::map<std::array<int, 3>, std::map<Triple, S>> tables_;
    std::pair<Triple, S> factor(Triple labels, int slot, const ActionTerm<S> &term) {
        labels[slot] = term.label;
        S v = term.coefficient;
        for (int copy = 0; copy < 2; copy++) {
            const Part &word = copy ? term.second : term.first;
            auto key = std::make_tuple(labels, slot, copy, word);
            auto found = vertex_cache_.find(key);
            if (found == vertex_cache_.end()) {
                std::array<S, 3> h;
                for (int j = 0; j < 3; j++)
                    h[j] = branch_weight(copy, labels[j], b_, p_[j]);
                found = vertex_cache_.emplace(key, vertex_word(slot, word, h)).first;
            }
            v *= found->second;
        }
        return {labels, v};
    }
    System make_system(int alpha, int gamma) {
        System sys;
        for (auto labels : support_)
            if ((labels[0] / 2 + alpha + gamma - f_) % 2 == 0)
                sys.unknowns.push_back(labels);
        std::map<Triple, uint32_t> index;
        for (uint32_t j = 0; j < sys.unknowns.size(); j++)
            index[sys.unknowns[j]] = j;
        std::vector<Sparse<S>> rows;
        for (auto labels : sys.unknowns) {
            Sparse<S> row;
            for (int slot = 0; slot < 3; slot++) {
                const auto &terms =
                    slot == 0 ? ns_actions_.at(labels[0])
                              : actions[slot - 1]->get(labels[slot], slot == 1 ? alpha : gamma);
                for (const auto &t : terms) {
                    auto [changed, v] = factor(labels, slot, t);
                    require(index.count(changed), "outer Ward action left its closed support");
                    row[index.at(changed)] += S(slot ? -1 : 1) * v;
                }
            }
            S norm(0);
            for (const auto &[k, v] : row)
                norm += conjugate(v) * v;
            norm = root(real_number(norm));
            if (!arithmetic_zero(norm)) {
                for (auto &[k, v] : row)
                    v /= norm;
                rows.push_back(std::move(row));
            }
        }
        sys.ward_rows = rows.size();
        for (auto labels : sys.unknowns)
            if (std::abs(labels[0]) <= 2 && std::abs(labels[1]) <= 3 && std::abs(labels[2]) <= 3) {
                sys.anchors.push_back(labels);
                rows.push_back({{index.at(labels), S(1)}});
            }
        std::vector<Sparse<S>> columns(sys.unknowns.size());
        for (uint32_t i = 0; i < rows.size(); i++)
            for (const auto &[j, v] : rows[i])
                columns[j][i] = v;
        std::vector<uint32_t> allrows;
        for (uint32_t j = 0; j < rows.size(); j++)
            allrows.push_back(j);
        sys.matrix = std::make_unique<LinearSystem<S>>(std::move(columns), allrows, true);
        return sys;
    }

  public:
    std::vector<int> ns, r;
    std::array<std::unique_ptr<RamondActions<S>>, 2> actions;
    double action_seconds = 0, ward_seconds = 0, maximum_action_residual = 0,
           maximum_ward_residual = 0;
    OuterBranching(S b, std::array<S, 3> p, int level, int f, int primary_parity, bool inserted)
        : b_(b), p_(p), level_(level), f_(f), primary_parity_(primary_parity),
          anchors_(b, p, primary_parity) {
        double start = seconds();
        ns = ns_labels(level);
        auto pairs = middle_pairs(level);
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
        std::array<std::unique_ptr<FreeField<S>>, 2> ns_modules;
        for (int n : ns) {
            if (std::abs(n) < 4) {
                ns_actions_[n] = {};
                continue;
            }
            int reflected = n < 0;
            if (!ns_modules[reflected])
                ns_modules[reflected] =
                    std::make_unique<FreeField<S>>(false, b_, reflected ? -p_[0] : p_[0]);
            Fit fit;
            auto terms = ns_action(*ns_modules[reflected], std::abs(n), fit);
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
                        if (n * n / 4 + ramond_level(incoming) + ramond_level(outgoing) +
                                2 * ramond_level(third) <=
                            2 * level) {
                            support_.insert({n, incoming, third});
                            support_.insert({n, outgoing, third});
                        }
        } else
            for (int n : ns)
                for (int second : r)
                    for (int third : r)
                        if (n * n / 4 + 2 * ramond_level(second) + 2 * ramond_level(third) <=
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
        for (int sign_eta : {eta, eta_prime})
            for (int a = 0; a < 2; a++)
                for (int g = 0; g < 2; g++) {
                    std::array<int, 3> key{a, g, sign_eta};
                    if (tables_.count(key))
                        continue;
                    auto pair = std::make_pair(a, g);
                    if (!systems_.count(pair))
                        systems_[pair] = make_system(a, g);
                    auto &sys = systems_.at(pair);
                    Sparse<S> target;
                    for (size_t j = 0; j < sys.anchors.size(); j++)
                        target[sys.ward_rows + j] =
                            anchors_.raw(sys.anchors[j], a, g, sign(f_) * sign_eta);
                    auto values = sys.matrix->solve(target);
                    maximum_ward_residual =
                        std::max(maximum_ward_residual, sys.matrix->fit.relative_residual);
                    auto &table = tables_[key];
                    for (size_t j = 0; j < values.size(); j++)
                        table[sys.unknowns[j]] = std::move(values[j]);
                }
        ward_seconds = seconds() - start;
    }
    const S &raw(int n1, int n2, int n3, int alpha, int gamma, int eta) const {
        return tables_.at({alpha, gamma, eta}).at({n1, n2, n3});
    }
    std::size_t support_size() const {
        return support_.size();
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
