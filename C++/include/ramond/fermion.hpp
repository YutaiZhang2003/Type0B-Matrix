#pragma once
#include "storage.hpp"
#include <gmpxx.h>
#include <map>
#include <set>
#include <tuple>
namespace ramond {
using Rational = mpq_class;
template <class S> S from_rational(const Rational &q) {
    if constexpr (std::is_same_v<S, MP>) {
        MP r;
        mpfr_set_q(mpc_realref(r.data()), q.get_mpq_t(), MPFR_RNDN);
        return r;
    } else
        return S(q.get_d());
}
inline Rational half_binomial(int numerator, int n) {
    Rational v(1);
    for (int j = 1; j <= n; j++)
        v *= Rational(numerator - 2 * (j - 1), 2 * j);
    return v;
}
struct AuxState {
    Part modes;
    int ground = 0;
    bool operator==(const AuxState &b) const {
        return modes == b.modes && ground == b.ground;
    }
    bool operator<(const AuxState &b) const {
        return std::tie(modes, ground) < std::tie(b.modes, b.ground);
    }
};
using AuxTriple = std::array<AuxState, 3>;
inline int aux_parity(int slot, const AuxState &s) {
    return (int(s.modes.size()) + (slot ? s.ground : 0)) % 2;
}
inline std::pair<AuxState, Rational> aux_act(int slot, int mode, AuxState s) {
    if (!mode) {
        require(slot != 0, "NS fermion has no zero mode");
        int g = s.ground;
        s.ground ^= 1;
        return {std::move(s), Rational(sign(s.modes.size()), g ? 1 : 2)};
    }
    auto it = std::lower_bound(s.modes.begin(), s.modes.end(), std::abs(mode), std::greater<int>());
    int position = it - s.modes.begin();
    bool exists = it != s.modes.end() && *it == std::abs(mode);
    if ((mode > 0 && !exists) || (mode < 0 && exists))
        return {std::move(s), Rational(0)};
    if (mode > 0)
        s.modes.erase(it);
    else
        s.modes.insert(it, -mode);
    return {std::move(s), Rational(sign(position))};
}
class FermionForm {
    std::map<AuxTriple, Rational> cache_;
    std::set<AuxTriple> active_;

  public:
    Rational value(const AuxTriple &states) {
        auto it = cache_.find(states);
        if (it != cache_.end())
            return it->second;
        std::array<int, 3> parity;
        for (int i = 0; i < 3; i++)
            parity[i] = aux_parity(i, states[i]);
        if ((parity[0] + parity[1] + parity[2]) % 2)
            return 0;
        int target = -1;
        for (int i = 0; i < 3; i++)
            if (!states[i].modes.empty()) {
                target = i;
                break;
            }
        if (target < 0)
            return cache_[states] = Rational(1 << states[1].ground);
        require(!active_.count(states), "cyclic fermion Ward reduction");
        active_.insert(states);
        auto rest = states;
        int first = rest[target].modes.front();
        rest[target].modes.erase(rest[target].modes.begin());
        int rp0 = aux_parity(0, rest[0]), rp1 = aux_parity(1, rest[1]);
        int koszul[3] = {1, sign(rp0), sign(rp0 + rp1)}, largest[3];
        for (int i = 0; i < 3; i++)
            largest[i] = rest[i].modes.empty() ? 0 : rest[i].modes.front();
        int phase_target = target == 0 ? 1 : target == 1 ? 0 : 3;
        Rational coefficient(0), remainder(0);
        auto add = [&](int slot, int mode, const Rational &c, int phase) {
            if (c == 0)
                return;
            auto [final, action] = aux_act(slot, mode, rest[slot]);
            if (action == 0)
                return;
            auto changed = rest;
            changed[slot] = std::move(final);
            int exponent = (phase + aux_parity(1, changed[1]) - parity[1] - phase_target + 8) % 4;
            require(exponent % 2 == 0, "inconsistent rational fermion Ward phase");
            Rational weight = c * action * sign(exponent / 2);
            if (changed == states)
                coefficient += weight;
            else
                remainder += weight * value(changed);
        };
        if (target == 0) {
            int k = (first + 1) / 2,
                cut = std::max({(first + largest[0]) / 2, largest[1], largest[2] - k, 0});
            for (int j = 0; j <= cut; j++) {
                Rational a = sign(j) * half_binomial(-1, j);
                add(0, 2 * j - first, koszul[0] * a, 1);
                add(1, j, koszul[1] * half_binomial(2 * k - 1, j), 0);
                add(2, k + j, koszul[2] * a, 3);
            }
        } else if (target == 1) {
            int cut =
                std::max({(largest[0] - 2 * first - 1) / 2, first + largest[1], largest[2], 0});
            for (int j = 0; j <= cut; j++) {
                Rational d = half_binomial(-1, j), e = sign(j) * half_binomial(-2 * first - 1, j);
                add(0, 2 * (first + j) + 1, koszul[0] * e, 1);
                add(1, -first + j, koszul[1] * d, 0);
                add(2, j, koszul[2] * sign(first) * e, 3);
            }
        } else {
            int cut =
                std::max({(largest[0] - 2 * first - 1) / 2, largest[1], first + largest[2], 0});
            for (int j = 0; j <= cut; j++) {
                Rational a = sign(j) * half_binomial(-1, j);
                add(0, 2 * (first + j) + 1, koszul[0] * a, 1);
                add(1, j, koszul[1] * half_binomial(-2 * first - 1, j), 0);
                add(2, -first + j, koszul[2] * a, 3);
            }
        }
        require(coefficient != 0, "fermion Ward identity missed its target");
        Rational result = -remainder / coefficient;
        active_.erase(states);
        cache_[states] = result;
        return result;
    }
    template <class S> S complex_value(const AuxTriple &states) {
        int phase = aux_parity(1, states[1]);
        return power(S(Machine(0, 1)), phase) * from_rational<S>(value(states)) /
               power(root(S(2)), states[1].ground + states[2].ground);
    }
    size_t cache_size() const {
        return cache_.size();
    }
};
inline int theta_sign(int index) {
    int a = index & 1, b = (index >> 1) & 1, c = (index >> 2) & 1;
    return sign(a * b + a * c + b * c);
}
inline int star_sign(int a, int b) {
    return theta_sign(a) * theta_sign(b) * theta_sign(a ^ b);
}
template <class S> using ParitySeries = std::unordered_map<Index, std::array<S, 8>, Hash>;
inline ParitySeries<Rational> fermion_series(int level, bool inserted, bool full_split = false) {
    int cutoff = 2 * level;
    std::vector<std::vector<Part>> ns(cutoff + 1), r(cutoff + 1);
    for (int i = 0; i <= cutoff; i++) {
        ns[i] = partitions(i, true, true);
        r[i] = partitions(i, true, false);
    }
    FermionForm form;
    ParitySeries<Rational> answer;
    for (int alevel = 0; alevel <= cutoff; alevel++) {
        int a = alevel % 2;
        for (int r3 = 0; r3 <= (cutoff - alevel) / 2; r3++) {
            int budget = cutoff - alevel - 2 * r3;
            for (int l = 0; l <= budget / 2; l++)
                for (const auto &A : ns[alevel])
                    for (const auto &C : r[r3])
                        for (const auto &B : r[l])
                            for (int g = 0; g < 2; g++) {
                                int b = (B.size() + g) % 2, c = a ^ b,
                                    g3 = (c - int(C.size()) + int(C.size()) * 2) % 2;
                                int index = a | (b << 1) | (c << 2),
                                    common = sign(a + b) * theta_sign(index);
                                AuxState third{C, g3}, left{B, g};
                                Rational rho = form.value({AuxState{A, 0}, left, third});
                                if (rho == 0)
                                    continue;
                                Rational value = common * rho * rho / Rational(1 << (g + g3));
                                if (inserted)
                                    value *= sign(g);
                                answer[{alevel, l, l, 2 * r3}][index] += value;
                                if (inserted && full_split)
                                    for (int mode = 1; mode <= budget - 2 * l; mode++) {
                                        if (std::find(B.begin(), B.end(), mode) != B.end())
                                            continue;
                                        Part upper = B;
                                        auto pos = std::lower_bound(upper.begin(), upper.end(),
                                                                    mode, std::greater<int>());
                                        int position = pos - upper.begin();
                                        upper.insert(pos, mode);
                                        Rational other = form.value(
                                            {AuxState{A, 0}, AuxState{upper, 1 - g}, third});
                                        if (other == 0)
                                            continue;
                                        Rational v = common * sign(position + b) * rho * other /
                                                     Rational(1 << g3);
                                        answer[{alevel, l, l + mode, 2 * r3}][index] += v;
                                        answer[{alevel, l + mode, l, 2 * r3}][index] += v;
                                    }
                            }
        }
    }
    return answer;
}
template <class S>
ParitySeries<S> numerical_fermion(int level, bool inserted, S q, bool full_split = false) {
    auto exact = fermion_series(level, inserted, full_split);
    ParitySeries<S> out;
    S factor = inserted ? q / root(S(2)) : S(1);
    for (const auto &[key, v] : exact)
        for (int i = 0; i < 8; i++)
            out[key][i] = factor * from_rational<S>(v[i]);
    return out;
}
} // namespace ramond
