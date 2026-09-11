#pragma once
#include "storage.hpp"
#include <map>
namespace ramond {
// Forward CCY recursion with the universal Schottky vacuum divided out.
// Pole geometry includes dc_{r,s}/dh times A_{r,s}; both it and the incident
// fusion polynomials are cached at fixed weights. No finite-c Gram sums occur here.
inline long binomial(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    long v = 1;
    for (int j = 1; j <= k; j++)
        v = v * (n - j + 1) / j;
    return v;
}
template <class S> S rising(S x, int n) {
    S v(1);
    for (int j = 0; j < n; j++)
        v *= x + S(j);
    return v;
}
inline std::vector<Index> virasoro_indices(int dim, int left, int right) {
    // For a punctured block, keep the downward closure of the diagonal targets.
    // Unequal middle-edge descendant levels are needed inside this closure.
    std::vector<Index> out;
    int m = std::min(left, right);
    for (int a = 0; a <= left; a++)
        for (int b = 0; b <= left - a; b++)
            for (int c = 0; c <= (dim == 3 ? left - a - b : right - a); c++)
                for (int d = 0; d <= (dim == 3 ? 0 : m - a); d++)
                    if (dim == 3 || (a + b + d <= left && a + c + d <= right))
                        out.push_back({a, b, c, d});
    std::sort(out.begin(), out.end(), [](const Index &a, const Index &b) {
        return degree(a) != degree(b) ? degree(a) < degree(b) : a < b;
    });
    return out;
}
inline std::vector<Index> virasoro_box_indices(const Index &limits) {
    std::vector<Index> out;
    for (int a = 0; a <= limits[0]; a++)
        for (int b = 0; b <= limits[1]; b++)
            for (int c = 0; c <= limits[2]; c++)
                for (int d = 0; d <= limits[3]; d++)
                    out.push_back({a, b, c, d});
    return out;
}
template <class S> class CCY {
    struct Pole {
        S central, residue, x;
        int id;
    };
    int dim_;
    std::array<S, 5> h_;
    std::vector<S> centers_;
    std::unordered_map<std::array<int, 4>, Pole, Hash> geometry_;
    std::unordered_map<std::array<int, 7>, S, Hash> fusion_;
    std::unordered_map<std::array<int, 6>, S, Hash> rho_;
    std::unordered_map<std::array<int, 3>, S, Hash> norm_;
    std::array<std::unordered_map<std::array<int, 6>, S, Hash>, 3> vertices_;
    std::unordered_map<std::array<int, 2>, S, Hash> inverse_denominators_;
    const S &inverse_denominator(int incoming, int outgoing) {
        const std::array<int, 2> key{incoming, outgoing};
        auto found = inverse_denominators_.find(key);
        if (found != inverse_denominators_.end())
            return found->second;
        const S &a = centers_[incoming], &b = centers_[outgoing];
        S difference = a - b;
        double scale = std::max({1., magnitude(a), magnitude(b)});
        require(!small(difference / S(scale), 64 * 0x1p-52, 10 - digits<S>()),
                "coincident or unresolved CCY poles; generic limit or more precision required");
        return inverse_denominators_.emplace(key, S(1) / difference).first->second;
    }
    S weight(int edge, int shift) const {
        return h_[edge] + S(shift);
    }
    S norm(int edge, int shift, int n) {
        std::array<int, 3> key{edge, shift, n};
        auto it = norm_.find(key);
        if (it != norm_.end())
            return it->second;
        S v = rising(S(2) * weight(edge, shift), n);
        for (int j = 2; j <= n; j++)
            v *= S(j);
        require(v != S(0), "degenerate global Gram norm; generic-weight limit required");
        norm_[key] = v;
        return v;
    }
    S rho(int vertex, int i, int j, int k, const std::array<int, 5> &shifts, bool normalized) {
        std::array<int, 3> e = dim_ == 3    ? std::array<int, 3>{0, 1, 2}
                               : vertex < 2 ? std::array<int, 3>{0, 1 + vertex, 3}
                                            : std::array<int, 3>{2, 4, 1};
        std::array<int, 3> s{shifts[e[0]], e[1] == 4 ? 0 : shifts[e[1]], shifts[e[2]]};
        std::array<int, 6> key{vertex, i, k, s[0], s[1], s[2]};
        S h1 = weight(e[0], s[0]), h2 = weight(e[1], s[1]), h3 = weight(e[2], s[2]), core;
        auto old = rho_.find(key);
        if (old != rho_.end())
            core = old->second;
        else {
            std::vector<S> rise(k + 1, S(1)), suffix(i + 1, S(1));
            for (int t = 0; t < k; t++)
                rise[t + 1] = rise[t] * (h3 + h2 - h1 + S(t));
            for (int t = i - 1; t >= 0; t--)
                suffix[t] = suffix[t + 1] * (h1 + h2 - h3 + S(-k + t));
            S v(0), fallh(1), fallk(1);
            for (int p = 0; p <= std::min(i, k); p++) {
                v += S(binomial(i, p)) * fallh * fallk * rise[k - p] * suffix[p];
                fallh *= S(2) * h3 + S(k - 1 - p);
                fallk *= S(k - p);
            }
            rho_[key] = v;
            core = std::move(v);
        }
        S value = core * rising(h1 - h2 - h3 + S(i - j + 1 - k), j);
        if (normalized)
            value /= norm(e[0], s[0], i) * norm(e[2], s[2], k);
        return value;
    }
    const S &vertex_factor(int vertex, int i, int j, int k,
                           const std::array<int, 5> &shifts) {
        const std::array<int, 3> edges = vertex < 2 ? std::array<int, 3>{0, 1 + vertex, 3}
                                                    : std::array<int, 3>{2, 4, 1};
        const std::array<int, 6> key{shifts[edges[0]], shifts[edges[1]], shifts[edges[2]], i, j, k};
        auto &cache = vertices_[vertex];
        auto found = cache.find(key);
        if (found != cache.end()) {
            vertex_factor_hits++;
            return found->second;
        }
        vertex_factor_evaluations++;
        return cache.emplace(key, rho(vertex, i, j, k, shifts, vertex != 1)).first->second;
    }
    S global(const Index &shift, const Index &n) {
        std::array<int, 5> s{shift[0], shift[1], shift[2], shift[3], 0};
        if (dim_ == 3) {
            S r = rho(0, n[0], n[1], n[2], s, false);
            return r * r / (norm(0, s[0], n[0]) * norm(1, s[1], n[1]) * norm(2, s[2], n[2]));
        }
        return vertex_factor(0, n[0], n[1], n[3], s) * vertex_factor(1, n[0], n[2], n[3], s) *
               vertex_factor(2, n[2], 0, n[1], s);
    }
    void accumulate_global(S &value, S &scratch, const S &amplitude,
                           const Index &shift, const Index &n) {
        if (dim_ == 3) {
            value += amplitude * global(shift, n);
            return;
        }
        const std::array<int, 5> s{shift[0], shift[1], shift[2], shift[3], 0};
        const S &a = vertex_factor(0, n[0], n[1], n[3], s),
                &b = vertex_factor(1, n[0], n[2], n[3], s),
                &c = vertex_factor(2, n[2], 0, n[1], s);
        if constexpr (std::is_same_v<S, MP>) {
            // Reuse the allocated MPC mantissas throughout the assembly sum.
            mpc_mul(scratch.data(), a.data(), b.data(), MPC_RNDNN);
            mpc_mul(scratch.data(), scratch.data(), c.data(), MPC_RNDNN);
            mpc_mul(scratch.data(), amplitude.data(), scratch.data(), MPC_RNDNN);
            mpc_add(value.data(), value.data(), scratch.data(), MPC_RNDNN);
        } else
            value += amplitude * (a * b * c);
    }
    const Pole &pole(int edge, int shift, int r, int s) {
        std::array<int, 4> key{edge, shift, r, s};
        auto old = geometry_.find(key);
        if (old != geometry_.end())
            return old->second;
        S h = weight(edge, shift),
          rad = S((r - s) * (r - s)) + S(4 * (r * s - 1)) * h + S(4) * h * h,
          base = S(r * s - 1) + S(2) * h;
        S x1 = (base + root(rad)) / S(1 - r * r), x2 = (base - root(rad)) / S(1 - r * r),
          x = magnitude(x1) >= magnitude(x2) ? x1 : x2;
        require(x != S(0), "CCY pole at infinity");
        S c = S(13) + S(6) * (x + S(1) / x), num = -S(12) * power(x, 2 * r * s - 1),
          den = S(1 - r * r) * x * x - S(1 - s * s);
        std::vector<std::pair<int, int>> factors;
        for (int a = 1 - r; a <= r; a++)
            for (int b = 1 - s; b <= s; b++)
                if ((a || b) && !(a == r && b == s))
                    factors.emplace_back(a, b);
        for (int target : {-1, 1}) {
            auto it = std::find_if(factors.begin(), factors.end(),
                                   [&](auto p) { return p.first && p.second == p.first * target; });
            if (it == factors.end())
                num *= x + S(target);
            else {
                den *= S(it->first);
                factors.erase(it);
            }
        }
        for (auto [a, b] : factors)
            den *= S(a) * x + S(b);
        require(den != S(0), "confluent Kac residue requires a generic-weight limit");
        int id = 0;
        while (id < int(centers_.size()) && centers_[id] != c)
            id++;
        if (id == int(centers_.size()))
            centers_.push_back(c);
        return geometry_.emplace(key, Pole{c, num / den, x, id}).first->second;
    }
    S fusion(const Pole &p, int r, int s, int top, int ts, int bottom, int bs) {
        std::array<int, 7> key{p.id, r, s, top, ts, bottom, bs};
        auto old = fusion_.find(key);
        if (old != fusion_.end())
            return old->second;
        S a = weight(top, ts), b = weight(bottom, bs), x = p.x,
          lam = x + S(2) + S(1) / x - S(4) * a, diff = S(4) * (b - a), v(1);
        for (int u = 1 - r; u < r; u += 2)
            for (int t = 1 - s; t < s; t += 2) {
                if (!u && !t)
                    v *= b - a;
                else if (u > 0 || (!u && t > 0)) {
                    S shift = S(u * u) * x + S(2 * u * t) + S(t * t) / x;
                    v *= (power(diff + shift, 2) - S(4) * lam * shift) / S(16);
                }
            }
        fusion_[key] = v;
        return v;
    }
    S residue(const Pole &p, const Index &shift, int edge, int r, int s) {
        std::array<int, 5> q{shift[0], shift[1], shift[2], shift[3], 0};
        if (dim_ == 3) {
            int a = edge == 0 ? 2 : 0, b = edge == 1 ? 2 : 1;
            S f = fusion(p, r, s, a, q[a], b, q[b]);
            return p.residue * f * f;
        }
        static const int pairs[4][4] = {{3, 1, 3, 2}, {3, 0, 2, 4}, {3, 0, 1, 4}, {0, 1, 0, 2}};
        S v = p.residue;
        for (int j = 0; j < 2; j++) {
            int a = pairs[edge][2 * j], b = pairs[edge][2 * j + 1];
            v *= fusion(p, r, s, a, q[a], b, q[b]);
        }
        return v;
    }

  public:
    std::size_t transitions = 0, seed_terms = 0,
                vertex_factor_evaluations = 0, vertex_factor_hits = 0;
    CCY(int dim, S central, const std::vector<S> &weights, S external = S(0))
        : dim_(dim), centers_{central} {
        require((dim == 3 || dim == 4) && int(weights.size()) == dim, "invalid CCY graph weights");
        for (int j = 0; j < dim; j++)
            h_[j] = weights[j];
        h_[4] = external;
    }
    Series<S> reduced(std::vector<Index> indices) {
        std::sort(indices.begin(), indices.end(), [](const Index &a, const Index &b) {
            return degree(a) != degree(b) ? degree(a) < degree(b) : a < b;
        });
        std::unordered_map<Index, bool, Hash> allowed;
        Index maximum{};
        for (auto n : indices) {
            for (int j = 0; j < 4; j++) {
                require(n[j] >= 0, "negative Virasoro level");
                maximum[j] = std::max(maximum[j], n[j]);
            }
            allowed[n] = true;
        }
        for (auto n : indices)
            for (int j = 0; j < dim_; j++)
                if (n[j]) {
                    auto lower = n;
                    lower[j]--;
                    require(allowed.count(lower), "CCY index set must be downward closed");
                }
        std::unordered_map<Index, std::map<int, S>, Hash> amplitudes;
        amplitudes[Index{}][0] = S(1);
        Series<S> totals, answer;
        for (auto shift : indices) {
            auto found = amplitudes.find(shift);
            if (found == amplitudes.end())
                continue;
            auto incoming = std::move(found->second);
            amplitudes.erase(found);
            S total(0);
            for (const auto &[id, v] : incoming)
                total += v;
            totals[shift] = total;
            for (int edge = 0; edge < dim_; edge++)
                for (int r = 2; r <= maximum[edge] - shift[edge]; r++)
                    for (int s = 1; r * s <= maximum[edge] - shift[edge]; s++) {
                        auto changed = shift;
                        changed[edge] += r * s;
                        if (!allowed.count(changed))
                            continue;
                        const auto &p = pole(edge, shift[edge], r, s);
                        S res = residue(p, shift, edge, r, s);
                        if (res == S(0))
                            continue;
                        S value(0);
                        for (const auto &[id, amplitude] : incoming) {
                            value += amplitude * inverse_denominator(id, p.id);
                            transitions++;
                        }
                        amplitudes[changed][p.id] += res * value;
                    }
        }
        std::vector<Index> shifts;
        for (const auto &[shift, v] : totals)
            shifts.push_back(shift);
        std::sort(shifts.begin(), shifts.end());
        S scratch;
        for (auto n : indices) {
            S v(0);
            for (auto shift : shifts)
                if (below(shift, n)) {
                    accumulate_global(v, scratch, totals.at(shift), shift, n - shift);
                    seed_terms++;
                }
            answer[n] = std::move(v);
        }
        return answer;
    }
};
template <class S>
Series<S> scalar_product(const Series<S> &a, const Series<S> &b, SeriesDomain cutoff) {
    Series<S> out;
    for (const auto &[ka, x] : a)
        if (x != S(0))
            for (const auto &[kb, y] : b)
                if (y != S(0) && cutoff.contains(ka + kb))
                    out[ka + kb] += x * y;
    return out;
}
template <class S>
Series<S> diagonal_product(const Series<S> &a, const Series<S> &b, int left, int right,
                           int first = -1, int third = -1) {
    Series<S> out;
    for (int t = 0; t <= std::min(left, right); t++) {
        int l = left - t, r = right - t;
        for (int x = 0; x <= (first < 0 ? t : first); x++)
            for (int d = 0; d <= (third < 0 ? t - x : third); d++) {
                S v(0);
                for (int xx = 0; xx <= x; xx++)
                    for (int ll = 0; ll <= l; ll++)
                        for (int rr = 0; rr <= r; rr++)
                            for (int dd = 0; dd <= d; dd++)
                                v += get(a, Index{xx, ll, rr, dd}) *
                                     get(b, Index{x - xx, l - ll, r - rr, d - dd});
                out[{2 * x, l, r, 2 * d}] = std::move(v);
            }
    }
    return out;
}
} // namespace ramond
