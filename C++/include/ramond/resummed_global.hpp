#pragma once
#include "ccy.hpp"

namespace ramond {
// Accuracy controls for the complete SL(2) descendant family. Neither field
// is a Virasoro residue order or a double-Virasoro primary cutoff.
struct GlobalControls {
    double tolerance = 1e-13;
    int maximum_shell = 64;
    void validate() const {
        require(std::isfinite(tolerance) && tolerance > 0 && tolerance < 1,
                "global tolerance must be between zero and one");
        require(maximum_shell >= 4 && maximum_shell <= 256,
                "global safety ceiling must be between 4 and 256");
    }
};
struct GlobalDiagnostics {
    size_t seeds = 0, terms = 0;
    int largest_shell = 0;
    double largest_last_shell_scaled = 0;
};
template <class S> using Laurent = std::map<int, S>;
template <class S> double laurent_norm(const Laurent<S> &p) {
    double v = 0;
    for (const auto &[k, x] : p) v += magnitude(x);
    return v;
}

// Coherent descendant-norm roots are essential for complex DV weights.
// Taking a new principal sqrt of a ratio can introduce a relative sign.
template <class S> std::vector<S> norm_steps(S h, int maximum) {
    std::vector<S> steps(maximum + 1);
    for (int n = 1; n <= maximum; ++n) {
        steps[n] = root(S(n) * (S(2) * h + S(n - 1)));
        require(steps[n] != S(0), "degenerate global norm requires a combined limit");
    }
    return steps;
}
template <class S> class NormalizedRho {
    std::vector<S> s1, s3;
    S ew, eu, euw;
    std::vector<std::vector<S>> a;
    int reached = 0;
  public:
    NormalizedRho(S h1, S h2, S h3, int maximum)
        : s1(norm_steps(h1, maximum)), s3(norm_steps(h3, maximum)),
          ew(h1-h2-h3), eu(h3-h1-h2), euw(h2-h1-h3), a(maximum+1) {
        a[0].push_back(S(1));
    }
    void ensure(int n) {
        while (reached < n) {
            int total = ++reached;
            a[0].push_back((S(total-1)-ew)*a[0][total-1]/s3[total]);
            for (int i = 0; i < total; ++i) {
                int k = total-1-i;
                S rhs = (S(i)-eu)*a[i][k];
                if (k) {
                    S ratio = S(k)/s3[k];
                    rhs += (S(i)-euw)*ratio*a[i][k-1];
                    if (i)
                        rhs += (eu+euw+S(1-i))*(S(i)/s1[i])*ratio*a[i-1][k-1];
                }
                a[i+1].push_back(rhs/s1[i+1]);
                require(finite(a[i+1][k]), "nonfinite normalized global coefficient");
            }
        }
    }
    const std::vector<S> &operator[](int i) const { return a[i]; }
};
template <class S>
S hypergeometric_2f1(S a, S b, S c, S z, double tolerance) {
    require(magnitude(z) < 1, "resummed global seed requires |q| < 1");
    S sum(1), term(1);
    int small_shells = 0;
    for (int n = 1; n <= 10000; ++n) {
        term *= (a+S(n-1))*(b+S(n-1))*z/(S(n)*(c+S(n-1)));
        require(finite(term), "nonfinite hypergeometric term");
        sum += term;
        small_shells = magnitude(term) <= tolerance*std::max(1., magnitude(sum))
                           ? small_shells + 1 : 0;
        if (small_shells >= 3) return sum;
    }
    throw std::runtime_error("global 2F1 failed to converge at its safety ceiling");
}
template <class S> std::vector<S> powers(S q, int maximum) {
    std::vector<S> out(maximum + 1, S(1));
    for (int n = 1; n <= maximum; ++n) out[n] = out[n-1]*q;
    return out;
}
inline void record_seed(GlobalDiagnostics &diagnostic, int shell, double norm,
                        double scale) {
    diagnostic.seeds++;
    diagnostic.largest_shell = std::max(diagnostic.largest_shell, shell);
    diagnostic.largest_last_shell_scaled = std::max(
        diagnostic.largest_last_shell_scaled, norm/std::max(1., scale));
}
template <class S>
S theta_global_resummed(const std::array<S, 3> &h, const std::array<S, 3> &q,
                        const GlobalControls &controls, GlobalDiagnostics &diagnostic) {
    controls.validate();
    for (auto z : q) require(magnitude(z) < 1, "global seed requires |q| < 1");
    int maximum = controls.maximum_shell;
    NormalizedRho<S> rho(h[0], h[1], h[2], maximum);
    auto q1 = powers(q[0], maximum), q3 = powers(q[2], maximum);
    std::map<int, S> hyper;
    S total(0);
    int small_shells = 0;
    for (int n = 0; n <= maximum; ++n) {
        rho.ensure(n);
        double shell_norm = 0;
        for (int i = 0; i <= n; ++i) {
            int k = n-i, difference = i-k;
            if (!hyper.count(difference)) {
                S a = h[1]+h[2]-h[0]-S(difference);
                hyper[difference] = hypergeometric_2f1(
                    a, a, S(2)*h[1], q[1], controls.tolerance*.01);
            }
            S term = q1[i]*q3[k]*rho[i][k]*rho[i][k]*hyper.at(difference);
            total += term;
            shell_norm += magnitude(term);
            diagnostic.terms++;
        }
        require(finite(total) && std::isfinite(shell_norm), "nonfinite global sum");
        small_shells = shell_norm <= controls.tolerance*std::max(1., magnitude(total))
                           ? small_shells + 1 : 0;
        if (n >= 4 && small_shells >= 3) {
            record_seed(diagnostic, n, shell_norm, magnitude(total));
            return total;
        }
    }
    throw std::runtime_error("theta global endpoint sum failed to converge at its safety ceiling");
}

// A punctured block must retain both middle-edge levels until the two
// Virasoro copies are multiplied and the diagonal is extracted. Put
// x=sqrt(q_middle)*z, y=sqrt(q_middle)/z and resum into a Laurent series in z.
// This sums the full four-edge global family using absolute homogeneous
// shell norms. It is independent of the null-recursion order. No Fourier
// aliasing or identification of unequal middle levels is introduced.
template <class S>
Laurent<S> punctured_global_resummed(const std::array<S, 4> &h, S external,
                                    const std::array<S, 3> &q,
                                    const GlobalControls &controls,
                                    GlobalDiagnostics &diagnostic) {
    controls.validate();
    for (auto z : q) require(magnitude(z) < 1, "global seed requires |q| < 1");
    int maximum = controls.maximum_shell;
    NormalizedRho<S> left(h[0], h[1], h[3], maximum),
         right(h[0], h[2], h[3], maximum), middle(h[2], external, h[1], maximum);
    auto sleft = norm_steps(h[1], maximum), sright = norm_steps(h[2], maximum);
    auto q1 = powers(q[0], maximum), q3 = powers(q[2], maximum);
    S minus_root = -root(q[1]);
    struct MiddleEntry {
        std::vector<S> a{S(1)}, b{S(1)};
        std::map<int, std::vector<S>> shells;
    };
    std::map<int, MiddleEntry> middle_cache;
    std::vector<std::vector<S>> outer_cache(maximum+1);
    Laurent<S> total;
    int small_shells = 0;
    for (int n = 0; n <= maximum; ++n) {
        left.ensure(n); right.ensure(n); middle.ensure(n);
        for (int i = 0; i <= n; ++i) {
            int k = n-i;
            outer_cache[i].push_back(q1[i]*q3[k]*left[i][k]*right[i][k]);
        }
        double shell_norm = 0;
        for (int endpoint = 0; endpoint <= n; ++endpoint)
            for (int i = 0; i <= endpoint; ++i) {
                int k = endpoint-i, mid = n-endpoint;
                const S &outer = outer_cache[i][k];
                int difference = i-k;
                auto &entry = middle_cache[difference];
                auto &a = entry.a, &b = entry.b;
                S aa = h[1]+h[3]-h[0]-S(difference),
                  bb = h[2]+h[3]-h[0]-S(difference);
                while (int(a.size()) <= mid) {
                    int m = a.size();
                    a.push_back(a.back()*(aa+S(m-1))*minus_root/sleft[m]);
                    b.push_back(b.back()*(bb+S(m-1))*minus_root/sright[m]);
                }
                if (!entry.shells.count(mid)) {
                    auto &values = entry.shells[mid];
                    for (int j = 0; j <= mid; ++j)
                        values.push_back(a[j]*b[mid-j]*middle[mid-j][j]);
                }
                const auto &values = entry.shells.at(mid);
                for (int j = 0; j <= mid; ++j) {
                    int l = mid-j;
                    S term = outer*values[j];
                    total[j-l] += term;
                    shell_norm += magnitude(term);
                    diagnostic.terms++;
                }
            }
        double scale = laurent_norm(total);
        require(std::isfinite(scale) && std::isfinite(shell_norm), "nonfinite punctured global sum");
        small_shells = shell_norm <= controls.tolerance*std::max(1., scale)
                           ? small_shells + 1 : 0;
        if (n >= 4 && small_shells >= 3) {
            record_seed(diagnostic, n, shell_norm, scale);
            return total;
        }
    }
    throw std::runtime_error("punctured global sum failed to converge at its safety ceiling");
}

template <class S>
Laurent<S> resummed_ccy(int dimension, S central, const std::vector<S> &weights,
                        S external, const std::array<S, 3> &q, int recursion_order,
                        const GlobalControls &controls, GlobalDiagnostics &diagnostic) {
    require(recursion_order >= 0, "negative pole-recursion order");
    // Physical total null level per copy; global descendants spend no budget.
    // For a puncture the two middle segments belong to the same tube. Its
    // downward-closed domain is a+d+max(left,right)<=R, NOT a+left+right+d<=R.
    // In particular the (left,right)=(2,2) residue is needed already at R=2.
    auto allowed = virasoro_indices(dimension, recursion_order, recursion_order);
    CCY<S> engine(dimension, central, weights, external);
    auto amplitudes = engine.pole_amplitudes(std::move(allowed));
    Laurent<S> total;
    S middle_root = root(q[1]);
    for (const auto &[shift, amplitude] : amplitudes) {
        if (amplitude == S(0)) continue;
        if (dimension == 3) {
            std::array<S, 3> h;
            for (int e = 0; e < 3; ++e) h[e] = weights[e]+S(shift[e]);
            S factor = amplitude;
            for (int e = 0; e < 3; ++e) factor *= power(q[e], shift[e]);
            total[0] += factor*theta_global_resummed(h, q, controls, diagnostic);
        } else {
            std::array<S, 4> h;
            for (int e = 0; e < 4; ++e) h[e] = weights[e]+S(shift[e]);
            S factor = amplitude*power(q[0], shift[0])*power(q[2], shift[3])*
                       power(middle_root, shift[1]+shift[2]);
            auto seed = punctured_global_resummed(h, external, q, controls, diagnostic);
            for (const auto &[difference, value] : seed)
                total[difference+shift[1]-shift[2]] += factor*value;
        }
    }
    return total;
}
template <class S>
S diagonal_laurent_product(const Laurent<S> &a, const Laurent<S> &b, int difference) {
    S total(0);
    for (const auto &[k, x] : a) {
        auto found = b.find(difference-k);
        if (found != b.end()) total += x*found->second;
    }
    return total;
}
} // namespace ramond
