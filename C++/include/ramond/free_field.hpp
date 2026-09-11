#pragma once
#include "linalg.hpp"
#include <map>
#include <memory>
#include <set>
#include <tuple>
namespace ramond {
/* NS fermions use doubled odd modes; R fermions use integer modes. */
struct State {
    std::array<unsigned char, 64> boson{};
    uint64_t fermion = 0, auxiliary = 0;
    unsigned char physical_ground = 0, auxiliary_ground = 0;
    bool operator==(const State &b) const {
        return boson == b.boson && fermion == b.fermion && auxiliary == b.auxiliary &&
               physical_ground == b.physical_ground && auxiliary_ground == b.auxiliary_ground;
    }
};
struct StateHash {
    size_t operator()(const State &s) const {
        size_t h = mix(s.fermion) ^ mix(s.auxiliary + 1337);
        for (int i = 1; i < 64; i++)
            if (s.boson[i])
                h = mix(h ^ (i * 256 + s.boson[i]));
        return mix(h ^ (s.physical_ground + 4 * s.auxiliary_ground));
    }
};
struct PBW {
    Part l, g;
    int ground = 0;
    bool operator<(const PBW &b) const {
        return std::tie(l, g, ground) < std::tie(b.l, b.g, b.ground);
    }
    bool operator==(const PBW &b) const {
        return l == b.l && g == b.g && ground == b.ground;
    }
};
template <class S> S ell(S x, int n, S b) {
    S q = b + S(1) / b;
    if (n < 0) {
        S v = ell(q - x, -n, b);
        return n % 2 ? v : S(sign((-n) / 2)) * v;
    }
    S v = n % 2 ? root(root(root(S(2)))) : S(1);
    for (int r = 0; r < n; r++)
        for (int s = 0; s < n - r; s++)
            if ((r + s) % 2 == n % 2)
                v *= x + S(r) * b + S(s) / b;
    return v;
}
template <class S> S ns_norm(int n4, S b, S p) {
    if (n4 < 0)
        return ns_norm(-n4, b, -p);
    S q = b + S(1) / b;
    return S(sign(n4 / 2)) * power(S(2), -n4 / 2) * ell(S(2) * p, n4, b) * ell(q + S(2) * p, n4, b);
}
template <class S> S r_norm(int n4, int parity, S b, S p) {
    if (n4 < 0)
        return r_norm(-n4, parity, b, -p);
    int m = (n4 - 1) / 2;
    S pref = parity ? -power(S(2), 2 * ((m + 1) / 2)) : power(S(2), 2 * (m / 2) + 1);
    return pref * ell(S(2) * p, n4, b) / ell(b + S(1) / b + S(2) * p, n4, b);
}
template <class S> S branch_weight(int copy, int n4, S b, S p) {
    S q = b + S(1) / b, t = copy ? S(1) / b : b;
    return (q * q / S(4) - power(p + S(n4) * t / S(2), 2)) / (S(2) * (S(1) - t * t));
}
template <class S> S branch_central(int copy, S b) {
    S q = b + S(1) / b, t = copy ? S(1) / b : b;
    return S(1) + S(3) * q * q / (S(1) - t * t);
}
template <class S> class FreeField {
  public:
    bool ramond;
    S b, p, q;
    int realization;
    std::vector<State> states;

  private:
    std::unordered_map<State, uint32_t, StateHash> ids_;
    std::unordered_map<uint64_t, Sparse<S>> physical_, auxiliary_, embedded_;
    std::map<std::tuple<int, int>, Sparse<S>> primaries_;
    std::map<std::tuple<int, Part, Part>, Sparse<S>> suffixes_;
    std::array<std::array<S, 3>, 2> coefficients_;
    static uint64_t cache_key(int kind, int mode, uint32_t state) {
        return (uint64_t(kind) << 56) | (uint64_t(mode + 32768) << 32) | state;
    }
    static int population(uint64_t x) {
        return __builtin_popcountll(x);
    }
    int mode_of(int bit) const {
        return ramond ? bit + 1 : 2 * bit + 1;
    }
    int mode_sum(uint64_t mask) const {
        int n = 0;
        while (mask) {
            int i = __builtin_ctzll(mask);
            n += mode_of(i);
            mask &= mask - 1;
        }
        return n;
    }
    void add_state(Sparse<S> &out, State s, const S &v) {
        if (v != S(0))
            out[intern(s)] += v;
    }
    void prune(Sparse<S> &s) {
        for (auto it = s.begin(); it != s.end();)
            if (arithmetic_zero(it->second))
                it = s.erase(it);
            else
                ++it;
    }

  public:
    FreeField(bool r, S coupling, S momentum, int realization_ = -1)
        : ramond(r), b(coupling), p(momentum), q(b + S(1) / b), realization(realization_) {
        S inv = S(1) / b, den = inv - b;
        coefficients_[0] = {inv / den, -(inv + S(2) * b) / den, S(1) / den};
        coefficients_[1] = {-b / den, (b + S(2) * inv) / den, -S(1) / den};
        intern(State{});
    }
    uint32_t intern(State s) {
        auto it = ids_.find(s);
        if (it != ids_.end())
            return it->second;
        uint32_t id = states.size();
        require(id != UINT32_MAX, "oscillator state identifier overflow");
        states.push_back(s);
        ids_.emplace(std::move(s), id);
        return id;
    }
    int physical_level(const State &s) const {
        int n = mode_sum(s.fermion);
        for (int i = 1; i < 64; i++)
            n += (ramond ? 1 : 2) * i * s.boson[i];
        return n;
    }
    int aux_level(const State &s) const {
        return mode_sum(s.auxiliary);
    }
    int aux_parity(const State &s) const {
        return (population(s.auxiliary) + (ramond ? s.auxiliary_ground : 0)) % 2;
    }
    S fermion(State &s, int mode, bool auxiliary, int phi_realization = 0) const {
        uint64_t &mask = auxiliary ? s.auxiliary : s.fermion;
        unsigned char &ground = auxiliary ? s.auxiliary_ground : s.physical_ground;
        if (mode == 0) {
            require(ramond, "NS fermion zero mode requested");
            int phase =
                auxiliary ? 1 : ((phi_realization ? phi_realization : realization) == -1 ? 1 : -1);
            ground ^= 1;
            return S(sign(population(mask)) * phase) / root(S(2));
        }
        int bit = ramond ? std::abs(mode) - 1 : std::abs(mode) / 2;
        require(bit >= 0 && bit < 64, "fermion mode exceeds 64-mode storage bound");
        uint64_t flag = UINT64_C(1) << bit;
        bool occupied = mask & flag;
        if ((mode < 0 && occupied) || (mode > 0 && !occupied))
            return S(0);
        int above = bit == 63 ? 0 : population(mask >> (bit + 1));
        mask ^= flag;
        return S(sign(above));
    }
    S boson(State &s, int mode) const {
        require(mode != 0 && std::abs(mode) < 64, "invalid bosonic oscillator mode");
        if (mode < 0) {
            require(s.boson[-mode] < 255, "boson occupancy overflow");
            s.boson[-mode]++;
            return S(1);
        }
        int count = s.boson[mode];
        if (!count)
            return S(0);
        s.boson[mode]--;
        return S(mode * count);
    }
    const Sparse<S> &physical(int kind, int mode, State state) {
        state.auxiliary = 0;
        state.auxiliary_ground = 0;
        uint64_t key = cache_key(kind, mode, intern(state));
        auto found = physical_.find(key);
        if (found != physical_.end())
            return found->second;
        Sparse<S> out;
        std::set<int> indices;
        if (kind == 0) {
            for (int i = 1; i < 64; i++)
                if (state.boson[i]) {
                    indices.insert(i);
                    indices.insert(mode - i);
                }
            if (mode < 0)
                for (int i = mode + 1; i < 0; i++)
                    indices.insert(i);
            for (int j : indices)
                if (j && j != mode) {
                    State s = state;
                    S v = boson(s, j);
                    if (v != S(0)) {
                        v *= boson(s, mode - j);
                        add_state(out, s, v / S(2));
                    }
                }
            indices.clear();
            int m = ramond ? mode : 2 * mode;
            for (int bit = 0; bit < 64; bit++)
                if ((state.fermion >> bit) & 1) {
                    int r = mode_of(bit);
                    indices.insert(r);
                    indices.insert(m - r);
                }
            if (mode < 0)
                for (int r = ramond ? mode : 2 * mode + 1; r <= (ramond ? 0 : -1);
                     r += ramond ? 1 : 2)
                    indices.insert(r);
            for (int r : indices) {
                State s = state;
                S v = fermion(s, r, false);
                if (v != S(0)) {
                    v *= fermion(s, m - r, false);
                    add_state(out, s, S(r) * v / S(ramond ? 2 : 4));
                }
            }
            State s = state;
            S v = boson(s, mode);
            if (v != S(0))
                add_state(out, s, S(Machine(0, .5)) * (q * S(mode) + S(2 * realization) * p) * v);
        } else {
            for (int i = 1; i < 64; i++)
                if (state.boson[i])
                    indices.insert(i);
            for (int bit = 0; bit < 64; bit++)
                if ((state.fermion >> bit) & 1) {
                    int r = mode_of(bit);
                    indices.insert(ramond ? mode - r : (mode - r) / 2);
                }
            if (ramond && mode)
                indices.insert(mode);
            if (mode < 0)
                for (int i = ramond ? mode : mode / 2; i < 0; i++)
                    indices.insert(i);
            for (int j : indices)
                if (j) {
                    State s = state;
                    S v = fermion(s, ramond ? mode - j : mode - 2 * j, false);
                    if (v != S(0)) {
                        v *= boson(s, j);
                        add_state(out, s, v);
                    }
                }
            State s = state;
            S v = fermion(s, mode, false);
            if (v != S(0))
                add_state(out, s,
                          S(Machine(0, 1)) *
                              (q * S(mode) / S(ramond ? 1 : 2) + S(realization) * p) * v);
        }
        prune(out);
        return physical_.emplace(key, std::move(out)).first->second;
    }
    const Sparse<S> &auxiliary(int mode, State state) {
        state.boson = {};
        state.fermion = 0;
        state.physical_ground = 0;
        uint64_t key = cache_key(0, mode, intern(state));
        auto it = auxiliary_.find(key);
        if (it != auxiliary_.end())
            return it->second;
        Sparse<S> out;
        std::set<int> indices;
        int m = ramond ? mode : 2 * mode;
        for (int bit = 0; bit < 64; bit++)
            if ((state.auxiliary >> bit) & 1) {
                int r = mode_of(bit);
                indices.insert(r);
                indices.insert(m - r);
            }
        if (mode < 0)
            for (int r = ramond ? mode : 2 * mode + 1; r <= (ramond ? 0 : -1); r += ramond ? 1 : 2)
                indices.insert(r);
        for (int r : indices) {
            State s = state;
            S v = fermion(s, r, true);
            if (v != S(0)) {
                v *= fermion(s, m - r, true);
                add_state(out, s, S(r) * v / S(ramond ? 2 : 4));
            }
        }
        prune(out);
        return auxiliary_.emplace(key, std::move(out)).first->second;
    }
    Sparse<S> basis(int kind, int mode, uint32_t id) {
        State state = states.at(id);
        Sparse<S> out;
        if (kind == 0 || kind == 1) {
            for (const auto &[k, v] : physical(kind, mode, state)) {
                State s = states[k];
                s.auxiliary = state.auxiliary;
                s.auxiliary_ground = state.auxiliary_ground;
                add_state(out, s, v);
            }
            return out;
        }
        if (kind == 2) {
            for (const auto &[k, v] : auxiliary(mode, state)) {
                State s = state;
                s.auxiliary = states[k].auxiliary;
                s.auxiliary_ground = states[k].auxiliary_ground;
                add_state(out, s, v);
            }
            return out;
        }
        if (kind == 3) {
            int lo = (ramond ? mode : 2 * mode) - aux_level(state), hi = physical_level(state);
            if (!ramond) {
                if (!(lo & 1))
                    lo++;
                if (!(hi & 1))
                    hi--;
            }
            for (int r = lo; r <= hi; r += ramond ? 1 : 2) {
                State aux = state;
                S v =
                    S(sign(aux_parity(state))) * fermion(aux, (ramond ? mode : 2 * mode) - r, true);
                if (v == S(0))
                    continue;
                for (const auto &[k, w] : physical(1, r, state)) {
                    State s = states[k];
                    s.auxiliary = aux.auxiliary;
                    s.auxiliary_ground = aux.auxiliary_ground;
                    add_state(out, s, v * w);
                }
            }
            prune(out);
            return out;
        }
        require(false, "unknown oscillator generator");
        return out;
    }
    const Sparse<S> &embedded(int copy, int mode, uint32_t id) {
        uint64_t key = cache_key(copy, mode, id);
        auto it = embedded_.find(key);
        if (it != embedded_.end())
            return it->second;
        Sparse<S> out;
        for (int j = 0; j < 3; j++) {
            auto terms = basis(j == 0 ? 0 : j + 1, mode, id);
            for (const auto &[k, v] : terms)
                out[k] += coefficients_[copy][j] * v;
        }
        prune(out);
        return embedded_.emplace(key, std::move(out)).first->second;
    }
    Sparse<S> apply(int kind, int mode, const Sparse<S> &in) {
        Sparse<S> out;
        for (const auto &[id, v] : in) {
            if (kind >= 4) {
                for (const auto &[k, w] : embedded(kind - 4, mode, id))
                    out[k] += v * w;
            } else {
                auto image = basis(kind, mode, id);
                for (const auto &[k, w] : image)
                    out[k] += v * w;
            }
        }
        prune(out);
        return out;
    }
    std::vector<std::pair<PBW, Sparse<S>>> pbw_basis(int level) {
        std::vector<std::pair<PBW, Sparse<S>>> out;
        for (int l = 0; l <= (ramond ? level : level / 2); l++)
            for (const auto &ls : partitions(l))
                for (const auto &gs : partitions(level - (ramond ? l : 2 * l), true, !ramond))
                    for (int ground = 0; ground < (ramond ? 2 : 1); ground++) {
                        State state{};
                        state.physical_ground = ground;
                        Sparse<S> expr{{intern(state), S(1)}};
                        for (auto it = gs.rbegin(); it != gs.rend(); ++it)
                            expr = apply(1, -*it, expr);
                        for (auto it = ls.rbegin(); it != ls.rend(); ++it)
                            expr = apply(0, -*it, expr);
                        out.push_back({PBW{ls, gs, ground}, std::move(expr)});
                    }
        return out;
    }
    Sparse<S> raw_ramond(int n4, int parity) {
        int native = n4 > 0 ? -1 : 1, largest = (std::abs(n4) - 1) / 2;
        std::vector<std::pair<int, int>> ops{{0, native}};
        for (int r = 1; r <= largest; r++)
            ops.emplace_back(-r, native);
        if (int(ops.size()) % 2 != parity)
            ops.emplace_back(0, -native);
        Sparse<S> expr{{0, S(1)}};
        for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
            Sparse<S> out;
            for (const auto &[id, v] : expr) {
                State original = states[id], a = original, f = original;
                S av = fermion(a, op->first, true), fv = fermion(f, op->first, false, op->second);
                add_state(out, a, v * av);
                add_state(out, f, v * S(Machine(0, -1)) * S(sign(aux_parity(original))) * fv);
            }
            prune(out);
            expr = std::move(out);
        }
        return expr;
    }
    const Sparse<S> &primary(int n4, int parity = 0) {
        auto key = std::make_tuple(n4, parity);
        auto it = primaries_.find(key);
        if (it != primaries_.end())
            return it->second;
        Sparse<S> expr{{0, S(1)}};
        if (!ramond) {
            require(n4 >= 0 && n4 % 2 == 0, "NS primary requires the positive half-integral chart");
            for (int mode = n4 - 1; mode >= 1; mode -= 2) {
                Sparse<S> out;
                for (const auto &[id, v] : expr) {
                    State original = states[id], a = original, f = original;
                    S av = fermion(a, -mode, true), fv = fermion(f, -mode, false);
                    add_state(out, a, v * av);
                    add_state(out, f, v * S(Machine(0, -1)) * S(sign(aux_parity(original))) * fv);
                }
                expr = std::move(out);
            }
            if (n4) {
                S scale = power(S(2), -n4 / 2) * ell(q + S(2) * p, n4, b);
                for (auto &[id, v] : expr)
                    v *= scale;
            }
        } else {
            expr = raw_ramond(n4, parity);
            int native = n4 > 0 ? -1 : 1;
            if (native != realization) {
                FreeField<S> other(true, b, p, native);
                std::map<std::pair<uint64_t, int>, Sparse<S>> groups;
                for (const auto &[id, v] : expr) {
                    State s = states[id];
                    auto aux = std::make_pair(s.auxiliary, int(s.auxiliary_ground));
                    s.auxiliary = 0;
                    s.auxiliary_ground = 0;
                    groups[aux][other.intern(s)] += v;
                }
                Sparse<S> converted;
                for (const auto &[aux, group] : groups) {
                    int level = other.physical_level(other.states[group.begin()->first]);
                    if (level == 0) {
                        for (const auto &[id, v] : group) {
                            State s = other.states[id];
                            s.auxiliary = aux.first;
                            s.auxiliary_ground = aux.second;
                            add_state(converted, s, v);
                        }
                        continue;
                    }
                    auto native_basis = other.pbw_basis(level), target_basis = pbw_basis(level);
                    std::vector<Sparse<S>> columns;
                    for (const auto &pair : native_basis)
                        columns.push_back(pair.second);
                    Fit fit;
                    auto values = span_solve(group, std::move(columns), fit);
                    for (size_t j = 0; j < values.size(); j++)
                        for (const auto &[id, v] : target_basis[j].second) {
                            State s = states[id];
                            s.auxiliary = aux.first;
                            s.auxiliary_ground = aux.second;
                            add_state(converted, s, v * values[j]);
                        }
                }
                expr = std::move(converted);
            }
        }
        prune(expr);
        return primaries_.emplace(key, std::move(expr)).first->second;
    }
    Sparse<S> descendant(int primary_key, const Sparse<S> &v, const Part &a, const Part &b,
                         bool retain = false) {
        auto key = std::make_tuple(primary_key, a, b);
        auto it = suffixes_.find(key);
        if (it != suffixes_.end())
            return it->second;
        if (a.empty() && b.empty())
            return v;
        Sparse<S> out;
        if (!a.empty() && (b.empty() || a[0] >= b[0]))
            out = apply(4, -a[0], descendant(primary_key, v, tail(a), b, true));
        else
            out = apply(5, -b[0], descendant(primary_key, v, a, tail(b), true));
        if (retain)
            suffixes_.emplace(std::move(key), out);
        return out;
    }
    void clear_action_cache() {
        suffixes_.clear();
        embedded_.clear();
        auxiliary_.clear();
        physical_.clear();
    }
};
template <class S> struct ActionTerm {
    int label;
    Part first, second;
    S coefficient;
};
template <class S> struct ActionPair {
    std::vector<ActionTerm<S>> minus, plus;
    Fit minus_fit, plus_fit;
};
template <class S> ActionPair<S> ramond_actions(FreeField<S> &module, int n4, bool with_plus) {
    const auto &high = module.primary(n4, 0);
    int neighbor = n4 == 1 ? -3 : n4 - 4, degree = n4 == 1 ? 0 : n4 - 1;
    const auto &low = module.primary(neighbor, 0);
    auto pairs = partition_pairs(degree);
    std::vector<Sparse<S>> columns;
    columns.push_back(module.descendant(0, high, {1}, {}));
    columns.push_back(module.descendant(0, high, {}, {1}));
    for (const auto &[a, b] : pairs)
        columns.push_back(module.descendant(1, low, a, b));
    auto target = module.apply(0, -1, high);
    ActionPair<S> answer;
    auto values = span_solve(target, std::move(columns), answer.minus_fit);
    answer.minus.push_back({n4, {1}, {}, values[0]});
    answer.minus.push_back({n4, {}, {1}, values[1]});
    for (size_t j = 0; j < pairs.size(); j++)
        answer.minus.push_back({neighbor, pairs[j].first, pairs[j].second, values[j + 2]});
    if (with_plus && n4 > 1) {
        auto pp = partition_pairs(n4 - 3);
        std::vector<Sparse<S>> cc;
        for (const auto &[a, b] : pp)
            cc.push_back(module.descendant(1, low, a, b));
        auto vv = span_solve(module.apply(0, 1, high), std::move(cc), answer.plus_fit);
        for (size_t j = 0; j < pp.size(); j++)
            answer.plus.push_back({neighbor, pp[j].first, pp[j].second, vv[j]});
    }
    module.clear_action_cache();
    return answer;
}
template <class S> std::vector<ActionTerm<S>> ns_action(FreeField<S> &module, int n4, Fit &fit) {
    const auto &high = module.primary(n4), &low = module.primary(n4 - 4);
    auto pairs = partition_pairs(n4 - 3);
    std::vector<Sparse<S>> columns;
    for (const auto &[a, b] : pairs)
        columns.push_back(module.descendant(0, low, a, b));
    auto values = span_solve(module.apply(0, 1, high), std::move(columns), fit);
    std::vector<ActionTerm<S>> out;
    for (size_t j = 0; j < pairs.size(); j++)
        out.push_back({n4 - 4, pairs[j].first, pairs[j].second, values[j]});
    module.clear_action_cache();
    return out;
}
} // namespace ramond
