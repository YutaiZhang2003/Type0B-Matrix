#pragma once
#include "pipeline.hpp"
#include "resummed_global.hpp"
namespace ramond {
struct ResummedSettings {
    Settings branching;
    int recursion_order = 3;
    GlobalControls global;
    std::array<std::string, 3> plumbing;
    double auxiliary_tolerance = 1e-14, vacuum_tolerance = 1e-14;
    int auxiliary_max_level = 24, vacuum_max_arc_length = 16, vacuum_max_mode = 50;
};
template <class S> using ParityValue = std::array<S, 8>;
template <class S> ParityValue<S> spectrum(const ParityValue<S> &a) {
    ParityValue<S> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            out[i] += S(theta_sign(j)*sign(__builtin_popcount(unsigned(i & j))))*a[j];
    return out;
}
template <class S> ParityValue<S> inverse_spectrum(const ParityValue<S> &a) {
    ParityValue<S> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            out[i] += S(theta_sign(i)*sign(__builtin_popcount(unsigned(i & j))))*a[j]/S(8);
    return out;
}
template <class S>
ParityValue<S> evaluate_parity(const ParitySeries<S> &series, const std::array<S, 3> &q) {
    ParityValue<S> out{};
    for (const auto &[key, row] : series) {
        require(key[1] == key[2], "physical evaluation requires the middle diagonal");
        S factor = power(root(q[0]), key[0])*power(q[1], key[1])*power(root(q[2]), key[3]);
        for (int j = 0; j < 8; ++j) out[j] += factor*row[j];
    }
    return out;
}
template <class S>
ParityValue<S> auxiliary_value(const ResummedSettings &s, S background_charge,
                               const std::array<S, 3> &q, int &level_reached,
                               double &last_change) {
    require(s.auxiliary_tolerance > 0 && s.auxiliary_tolerance < 1 && s.auxiliary_max_level >= 8,
            "invalid auxiliary convergence controls");
    ParityValue<S> previous{};
    int small_steps = 0;
    for (int level = 4; level <= s.auxiliary_max_level; level += 2) {
        auto current = evaluate_parity(numerical_fermion(level, s.branching.inserted,
                                                        background_charge), q);
        double change = 0, scale = 1;
        for (int j = 0; j < 8; ++j) {
            change += magnitude(current[j]-previous[j]);
            scale += magnitude(current[j]);
        }
        last_change = change/scale;
        std::cerr << "auxiliary level " << level << ": relative change " << last_change << '\n';
        small_steps = last_change <= s.auxiliary_tolerance ? small_steps+1 : 0;
        if (small_steps >= 2) { level_reached = level; return current; }
        previous = std::move(current);
    }
    throw std::runtime_error("auxiliary fermion failed its independent convergence check; last change="+
                             std::to_string(last_change));
}
template <class S> using NumericMatrix = std::array<S, 4>;
template <class S> NumericMatrix<S> matrix_product(const NumericMatrix<S> &a,
                                                  const NumericMatrix<S> &b) {
    return {a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3]};
}
template <class S>
S vacuum_value(const ResummedSettings &s, const std::array<S, 3> &q,
                int &arc_reached, double &last_shell) {
    require(s.vacuum_tolerance > 0 && s.vacuum_tolerance < 1 &&
            s.vacuum_max_arc_length >= 8 && s.vacuum_max_mode >= 3,
            "invalid vacuum convergence controls");
    std::array<NumericMatrix<S>, 3> maps{
        NumericMatrix<S>{S(0),S(1),q[0],S(0)},
        NumericMatrix<S>{S(1),q[1]-S(1),S(1),S(-1)},
        NumericMatrix<S>{S(0),q[2],S(1),S(0)}};
    std::set<Part> seen;
    S total(1);
    int small_shells = 0;
    for (int length = 2; length <= s.vacuum_max_arc_length; length += 2) {
        std::set<Part> classes;
        Part word;
        cycles_visit(0, 0, word, length, classes);
        cycles_visit(1, 1, word, length, classes);
        double shell_norm = 0;
        for (const auto &w : classes) {
            if (!seen.insert(w).second) continue;
            NumericMatrix<S> matrix{S(1),S(0),S(0),S(1)};
            S determinant(1);
            for (int arc : w) {
                matrix = matrix_product(maps[arc/2], matrix);
                determinant *= -q[arc/2];
            }
            S trace = matrix[0]+matrix[3], disc = root(trace*trace-S(4)*determinant),
              plus = (trace+disc)/S(2), minus = (trace-disc)/S(2),
              large = magnitude(plus) >= magnitude(minus) ? plus : minus;
            require(large != S(0), "singular Schottky word");
            S multiplier = determinant/(large*large);
            double radius = magnitude(multiplier);
            require(radius < 1, "Schottky multiplier is not contracting");
            S factor(1), mode = multiplier;
            for (int m = 2; m <= s.vacuum_max_mode; ++m) {
                mode *= multiplier;
                factor /= S(1)-mode;
            }
            double tail = std::pow(radius, s.vacuum_max_mode+1)/
                          ((1-radius)*(1-std::pow(radius, s.vacuum_max_mode+1)));
            require(tail <= s.vacuum_tolerance*.01,
                    "vacuum oscillator ceiling does not meet its tolerance");
            shell_norm += magnitude(factor-S(1));
            total *= factor;
        }
        last_shell = shell_norm;
        small_shells = shell_norm <= s.vacuum_tolerance ? small_shells+1 : 0;
        if (small_shells >= 3) { arc_reached = length; return total; }
    }
    throw std::runtime_error("Schottky word shells failed their independent convergence check");
}
template <class S> struct ResummedResult {
    ParityValue<S> numerator{}, auxiliary{}, physical{};
    GlobalDiagnostics global;
    S vacuum;
    size_t branch_cases = 0;
    int auxiliary_level = 0, vacuum_arc_length = 0;
    double sector_residual = 0, auxiliary_change = 0, vacuum_shell = 0,
           ward_residual = 0, action_residual = 0, seconds = 0;
};
template <class S> ResummedResult<S> resummed_pipeline(const ResummedSettings &s) {
    double started = seconds(), last = started;
    const auto &settings = s.branching;
    s.global.validate();
    require(s.recursion_order >= 0, "negative recursion order");
    require(settings.level >= 0 && (settings.p == 0 || settings.p == 1) &&
            (settings.f == 0 || settings.f == 1) && std::abs(settings.eta) == 1,
            "invalid branching parameters");
    ResummedResult<S> result;
    S b = parse<S>(settings.b), b2 = b*b, charge = b+S(1)/b;
    require(b != S(0) && b2 != S(1) && b == conjugate(b), "invalid b");
    std::array<S, 3> p, q;
    for (int e = 0; e < 3; ++e) {
        p[e] = parse<S>(settings.momenta[e]); q[e] = parse<S>(s.plumbing[e]);
        require(magnitude(q[e]) < 1, "resummation requires |q| < 1");
    }
    const int level = settings.level, cutoff = 2*level;
    const int eta = settings.eta, eta_prime = settings.inserted ? -eta : eta;
    OuterBranching<S> outer(b, p, level, settings.f, settings.p, settings.inserted, settings.box);
    outer.prepare(eta, eta_prime);
    result.ward_residual = outer.maximum_ward_residual;
    result.action_residual = outer.maximum_action_residual;
    std::unique_ptr<MiddleBranching<S>> middle;
    auto pairs = middle_pairs(level, settings.box);
    if (settings.inserted) middle = std::make_unique<MiddleBranching<S>>(b, p[1], *outer.actions[0]);
    else { pairs.clear(); for (int n : outer.r) pairs.emplace_back(n,n); }
    std::array<S, 2> central{branch_central(0,b), branch_central(1,b)},
        external{-(S(1)+S(2)*b2)/(S(2)*(S(1)-b2)), (b2+S(2))/(S(2)*(S(1)-b2))};
    std::map<std::array<int,4>,S> transposed_products;
    for (int n1 : outer.ns)
        for (auto [incoming, outgoing] : pairs)
            for (int n3 : outer.r) {
                Index shift{n1*n1/4, ramond_level(incoming), ramond_level(outgoing),
                            2*ramond_level(n3)};
                if (settings.box ? shift[0] > cutoff || shift[1] > level ||
                                   shift[2] > level || shift[3] > cutoff
                                 : degree(shift) > cutoff) continue;
                std::array<std::pair<int,S>, 2> factors;
                for (int alpha = 0; alpha < 2; ++alpha) {
                    int gamma = ((settings.f-n1/2-alpha)%2+2)%2,
                        index = ((n1/2+settings.p)%2+2)%2 | (alpha<<1) | (gamma<<2);
                    S denominator = ns_norm(n1,b,p[0])*r_norm(incoming,alpha,b,p[1])*
                                    r_norm(n3,gamma,b,p[2]);
                    S factor = outer.raw(n1,incoming,n3,alpha,gamma,eta)*
                               outer.raw(n1,outgoing,n3,alpha,gamma,eta_prime);
                    if (settings.inserted) {
                        denominator *= r_norm(outgoing,alpha,b,p[1]);
                        factor *= middle->raw(outgoing,incoming,alpha);
                    }
                    factors[alpha] = {index, factor*S(sign(n1/2)*theta_sign(index))/denominator};
                }
                if (factors[0].second == S(0) && factors[1].second == S(0)) continue;
                std::array<int,4> key{n1,std::min(incoming,outgoing),std::max(incoming,outgoing),n3};
                S value;
                auto stored = transposed_products.find(key);
                if (settings.inserted && stored != transposed_products.end()) {
                    value = stored->second;
                    transposed_products.erase(stored);
                } else {
                std::array<Laurent<S>, 2> blocks;
                for (int copy = 0; copy < 2; ++copy) {
                    std::vector<S> weights{branch_weight(copy,n1,b,p[0]),
                                          branch_weight(copy,incoming,b,p[1])};
                    if (settings.inserted) weights.push_back(branch_weight(copy,outgoing,b,p[1]));
                    weights.push_back(branch_weight(copy,n3,b,p[2]));
                    blocks[copy] = resummed_ccy(settings.inserted ? 4 : 3, central[copy],
                        weights, external[copy], q, s.recursion_order, s.global, result.global);
                }
                value = diagonal_laurent_product(blocks[0], blocks[1], shift[2]-shift[1])*
                          power(root(q[0]),shift[0])*power(root(q[1]),shift[1]+shift[2])*
                          power(root(q[2]),shift[3]);
                if (settings.inserted) transposed_products[key] = value;
                }
                for (const auto &[index,factor] : factors) result.numerator[index] += factor*value;
                result.branch_cases++;
                if (seconds()-last > 15) {
                    std::cerr << "resummed numerator: " << result.branch_cases << " branches; "
                              << result.global.seeds << " converged global seeds\n";
                    last = seconds();
                }
            }
    result.auxiliary = auxiliary_value(s, charge, q, result.auxiliary_level, result.auxiliary_change);
    auto numerator = spectrum(result.numerator), auxiliary = spectrum(result.auxiliary);
    ParityValue<S> quotient{};
    double scale = 1;
    for (const auto &v : numerator) scale = std::max(scale, magnitude(v));
    for (int character = 0; character < 8; ++character) {
        int eigenvalue = -sign(__builtin_popcount(unsigned(character & 6)));
        int sigma = settings.inserted ? -1 : 1;
        if (eigenvalue == sigma) {
            require(magnitude(auxiliary[character]) > 1e-12, "singular supported auxiliary character");
            quotient[character] = numerator[character]/auxiliary[character];
        } else result.sector_residual = std::max(result.sector_residual, magnitude(numerator[character])/scale);
    }
    require(result.sector_residual <= 1e-8,
            "resummed numerator leaves the recoverable sector; increase primary/residue orders or precision");
    result.physical = inverse_spectrum(quotient);
    result.vacuum = vacuum_value(s, q, result.vacuum_arc_length, result.vacuum_shell);
    for (auto &v : result.physical) v *= result.vacuum*result.vacuum;
    result.seconds = seconds()-started;
    return result;
}
template <class S> void encode_resummed(std::ostream &out, const ResummedSettings &s,
                                       const ResummedResult<S> &r) {
    const auto &b = s.branching;
    out << std::setprecision(17) << "{\"status\":\"computed\",\"method\":\"double_virasoro_global_resummed\","
        << "\"mode\":\"" << (b.inserted ? "inserted" : "ordinary") << "\",\"branch_level\":" << b.level
        << ",\"branch_truncation\":\"" << (b.box ? "per-edge" : "total") << "\",\"recursion_order\":" << s.recursion_order
        << ",\"recursion_domain\":\"physical total level per copy; punctured diagonal downward closure\",\"global_tolerance\":" << s.global.tolerance
        << ",\"global_maximum_shell\":" << s.global.maximum_shell << ",\"dps\":" << b.dps
        << ",\"b\":\"" << b.b << "\",\"p\":" << b.p << ",\"f\":" << b.f
        << ",\"etas\":[" << b.eta << ',' << (b.inserted ? -b.eta : b.eta) << "],\"momenta\":[";
    for (int e = 0; e < 3; ++e) { if(e) out << ','; out << '"' << b.momenta[e] << '"'; }
    out << "],\"q_values\":[";
    for (int e = 0; e < 3; ++e) { if(e) out << ','; out << '"' << s.plumbing[e] << '"'; }
    out << "],\"diagnostics\":{\"global_seeds\":" << r.global.seeds << ",\"global_terms\":" << r.global.terms
        << ",\"largest_global_shell\":" << r.global.largest_shell << ",\"last_global_shell_scaled\":" << r.global.largest_last_shell_scaled
        << ",\"sector_residual\":" << r.sector_residual << ",\"ward_residual\":" << r.ward_residual
        << ",\"action_residual\":" << r.action_residual << ",\"branches\":" << r.branch_cases
        << ",\"auxiliary_level\":" << r.auxiliary_level << ",\"auxiliary_last_change\":" << r.auxiliary_change
        << ",\"vacuum_arc_length\":" << r.vacuum_arc_length << ",\"vacuum_last_shell\":" << r.vacuum_shell
        << "},\"seconds\":" << r.seconds << ",\"vacuum\":" << json_number(r.vacuum);
    for (const auto &[name, values] : std::vector<std::pair<std::string,ParityValue<S>>>{
            {"reduced_numerator", r.numerator}, {"auxiliary", r.auxiliary}, {"physical_values", r.physical}}) {
        out << ",\"" << name << "\":[";
        for (int j=0;j<8;++j) { if(j) out << ','; out << json_number(values[j]); }
        out << ']';
    }
    out << "}\n";
}
} // namespace ramond
