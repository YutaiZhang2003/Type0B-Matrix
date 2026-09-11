#pragma once
#include "number.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>
namespace ramond {
inline std::size_t mix(std::size_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
struct Hash {
    template <class T, std::size_t N> std::size_t operator()(const std::array<T, N> &a) const {
        std::size_t h = 0;
        for (const auto &x : a)
            h = mix(h ^ std::hash<T>{}(x));
        return h;
    }
};
using Index = std::array<int, 4>;
inline int degree(const Index &a) {
    return a[0] + a[1] + a[2] + a[3];
}
inline Index operator+(Index a, const Index &b) {
    for (int j = 0; j < 4; j++)
        a[j] += b[j];
    return a;
}
inline Index operator-(Index a, const Index &b) {
    for (int j = 0; j < 4; j++)
        a[j] -= b[j];
    return a;
}
inline bool below(const Index &a, const Index &b) {
    for (int j = 0; j < 4; j++)
        if (a[j] > b[j])
            return false;
    return true;
}
// A downward-closed domain for ordinary (integer-exponent) series.
struct SeriesDomain {
    int total;
    bool box = false;
    Index limits{};
    SeriesDomain(int cutoff) : total(cutoff) {}
    SeriesDomain(Index maxima) : total(degree(maxima)), box(true), limits(maxima) {}
    bool contains(const Index &k) const {
        return box ? below(k, limits) : degree(k) <= total;
    }
};
inline bool physical_contains(const Index &k, int level, bool box) {
    return k[1] == k[2] && (box ? k[0] <= 2 * level && k[1] <= level && k[3] <= 2 * level
                               : degree(k) <= 2 * level);
}
template <class S> using Series = std::unordered_map<Index, S, Hash>;
template <class K, class S, class H = std::hash<K>>
S get(const std::unordered_map<K, S, H> &m, const K &k) {
    auto p = m.find(k);
    return p == m.end() ? S(0) : p->second;
}
template <class K, class S, class H>
void add(std::unordered_map<K, S, H> &m, const K &k, const S &v) {
    if (v != S(0))
        m[k] += v;
}
using Part = std::vector<int>;
inline void partitions_rec(int left, int top, bool strict, bool odd, Part &p,
                           std::vector<Part> &out) {
    if (!left) {
        out.push_back(p);
        return;
    }
    for (int n = std::min(left, top); n >= 1; n--)
        if (!odd || (n & 1)) {
            p.push_back(n);
            partitions_rec(left - n, n - int(strict), strict, odd, p, out);
            p.pop_back();
        }
}
inline std::vector<Part> partitions(int n, bool strict = false, bool odd = false) {
    std::vector<Part> out;
    Part p;
    if (n >= 0)
        partitions_rec(n, n, strict, odd, p, out);
    return out;
}
inline std::vector<std::pair<Part, Part>> partition_pairs(int n) {
    std::vector<std::pair<Part, Part>> out;
    for (int i = 0; i <= n; i++)
        for (const auto &a : partitions(i))
            for (const auto &b : partitions(n - i))
                out.emplace_back(a, b);
    return out;
}
inline int sum(const Part &p) {
    int s = 0;
    for (int n : p)
        s += n;
    return s;
}
inline Part tail(const Part &p) {
    return Part(p.begin() + 1, p.end());
}
inline double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline int sign(int n) {
    return (n & 1) ? -1 : 1;
}
} // namespace ramond
