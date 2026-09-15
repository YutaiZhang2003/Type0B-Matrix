#pragma once
// Small physical PBW systems fix the low branching anchors only. Production
// Virasoro blocks are computed by CCY, not by physical or Virasoro Gram sewing.
#include "ccy.hpp"
#include "direct_pbw.hpp"
#include "fermion.hpp"
#include "free_field.hpp"
namespace ramond {
template <class S> class PBWModule {
    FreeField<S> &free_;
    std::map<int, std::vector<std::pair<PBW, Sparse<S>>>> bases_;
    std::map<PBW, Sparse<S>> fock_;
    std::map<std::tuple<int, int, PBW>, std::map<PBW, S>> actions_;

  public:
    S h;
    PBWModule(FreeField<S> &f)
        : free_(f),
          h(f.q * f.q / S(8) - f.p * f.p / S(2) + (f.ramond ? rational<S>(1, 16) : S(0))) {}
    int level(const PBW &s) const {
        return (free_.ramond ? 1 : 2) * sum(s.l) + sum(s.g);
    }
    int parity(const PBW &s) const {
        return (s.g.size() + (free_.ramond ? s.ground : 0)) % 2;
    }
    S l0(const PBW &s) const {
        return h + S(level(s)) / S(free_.ramond ? 1 : 2);
    }
    const std::vector<std::pair<PBW, Sparse<S>>> &basis(int level) {
        auto it = bases_.find(level);
        if (it != bases_.end())
            return it->second;
        return bases_.emplace(level, free_.pbw_basis(level)).first->second;
    }
    std::map<PBW, S> from_fock(const Sparse<S> &v) {
        if (v.empty())
            return {};
        int lev = free_.physical_level(free_.states.at(v.begin()->first));
        const auto &b = basis(lev);
        std::vector<Sparse<S>> cols;
        for (const auto &p : b)
            cols.push_back(p.second);
        Fit fit;
        auto coeff = span_solve(v, std::move(cols), fit);
        std::map<PBW, S> out;
        for (size_t j = 0; j < coeff.size(); j++)
            if (!arithmetic_zero(coeff[j]))
                out[b[j].first] = coeff[j];
        return out;
    }
    const Sparse<S> &fock(const PBW &s) {
        auto old = fock_.find(s);
        if (old != fock_.end())
            return old->second;
        State base{};
        base.physical_ground = s.ground;
        Sparse<S> v{{free_.intern(base), S(1)}};
        for (auto it = s.g.rbegin(); it != s.g.rend(); ++it)
            v = free_.apply(1, -*it, v);
        for (auto it = s.l.rbegin(); it != s.l.rend(); ++it)
            v = free_.apply(0, -*it, v);
        return fock_.emplace(s, std::move(v)).first->second;
    }
    const std::map<PBW, S> &act(int kind, int mode, const PBW &s) {
        auto key = std::make_tuple(kind, mode, s);
        auto old = actions_.find(key);
        if (old != actions_.end())
            return old->second;
        std::map<PBW, S> out;
        int newlevel = level(s) - mode * (kind == 0 && !free_.ramond ? 2 : 1);
        if (newlevel >= 0) {
            if (kind == 0 && mode == 0)
                out[s] = l0(s);
            else
                out = from_fock(free_.apply(kind, mode, fock(s)));
        }
        return actions_.emplace(std::move(key), std::move(out)).first->second;
    }
};
template <class S> class PhysicalForm {
    std::array<PBWModule<S> *, 3> modules_;
    int f_, eta_, p_;
    using Triple = std::array<PBW, 3>;
    std::map<Triple, S> cache_;
    std::set<Triple> active_;
    S sum_action(Triple states, int slot, int kind, int mode) {
        S v(0);
        for (const auto &[final, c] : modules_[slot]->act(kind, mode, states[slot])) {
            auto changed = states;
            changed[slot] = final;
            v += c * value(changed);
        }
        return v;
    }
    S gward(const Triple &states, int target, const Triple &rest) {
        int p0 = modules_[0]->parity(rest[0]), p1 = modules_[1]->parity(rest[1]);
        int koszul[3] = {1, sign(p0), sign(p0 + p1)};
        if (target < 2)
            koszul[2] *= sign(p_);
        else {
            koszul[0] *= sign(p_);
            koszul[1] *= sign(p_);
        }
        std::map<Triple, S> equation;
        auto add_action = [&](int slot, int mode, const S &c) {
            if (c == S(0))
                return;
            for (const auto &[final, v] : modules_[slot]->act(1, mode, rest[slot])) {
                auto changed = rest;
                changed[slot] = final;
                equation[changed] += c * v;
            }
        };
        const S imag(Machine(0, 1));
        int bound = 16; // Only the certified low anchors reach this evaluator.
        if (target == 0) {
            int mode = states[0].g.front(), m = (mode - 1) / 2;
            for (int j = 0; j <= bound; j++) {
                S a = from_rational<S>(sign(j) * half_binomial(1, j));
                add_action(0, 2 * j - mode, S(koszul[0]) * (-imag) * a);
                add_action(1, j, S(koszul[1]) * from_rational<S>(half_binomial(2 * m + 1, j)));
                add_action(2, m + j, S(koszul[2]) * imag * a);
            }
        } else if (target == 1) {
            int n = states[1].g.front();
            for (int j = 0; j <= bound; j++) {
                S c = from_rational<S>(sign(j) * half_binomial(1 - 2 * n, j));
                add_action(0, 2 * (n + j) - 1, S(koszul[0]) * (-imag) * c);
                add_action(1, -n + j, S(koszul[1]) * from_rational<S>(half_binomial(1, j)));
                add_action(2, j, S(koszul[2] * sign(n)) * imag * c);
            }
        } else {
            int n = states[2].g.front();
            for (int j = 0; j <= bound; j++) {
                S a = from_rational<S>(sign(j) * half_binomial(1, j));
                add_action(0, 2 * (n + j) - 1, S(koszul[0]) * (-imag) * a);
                add_action(1, j, S(koszul[1]) * from_rational<S>(half_binomial(1 - 2 * n, j)));
                add_action(2, -n + j, S(koszul[2]) * imag * a);
            }
        }
        S target_coefficient = equation[states];
        equation.erase(states);
        require(magnitude(target_coefficient) > 1e-10,
                "supercurrent Ward identity missed its target");
        S answer(0);
        for (const auto &[changed, c] : equation)
            if (!arithmetic_zero(c))
                answer -= c * value(changed);
        return answer / target_coefficient;
    }

  public:
    PhysicalForm(std::array<PBWModule<S> *, 3> modules, int f, int eta, int p)
        : modules_(modules), f_(f), eta_(eta), p_(p) {}
    S value(const Triple &states) {
        auto found = cache_.find(states);
        if (found != cache_.end())
            return found->second;
        int parity = 0;
        for (int j = 0; j < 3; j++)
            parity += modules_[j]->parity(states[j]);
        if (parity % 2 != f_)
            return S(0);
        require(!active_.count(states), "cyclic low-anchor SCA Ward reduction");
        active_.insert(states);
        S answer(0);
        if (!states[1].l.empty()) {
            int n = states[1].l.front();
            auto rest = states;
            rest[1].l = tail(rest[1].l);
            if (n == 1)
                answer = (modules_[0]->l0(states[0]) - modules_[1]->l0(rest[1]) -
                          modules_[2]->l0(states[2])) *
                         value(rest);
            else {
                int maximum = std::max(modules_[0]->level(states[0]) + n + 2,
                                       modules_[2]->level(states[2]) + 3);
                for (int j = 0; j <= maximum; j++)
                    answer += S(binomial(n - 2 + j, n - 2)) *
                              (sum_action(rest, 0, 0, n + j) +
                               S(sign(n)) * sum_action(rest, 2, 0, j - 1));
            }
        } else if (!states[0].l.empty()) {
            int n = states[0].l.front();
            auto rest = states;
            rest[0].l = tail(rest[0].l);
            answer = sum_action(rest, 2, 0, n);
            for (int j = -1; j <= n; j++)
                answer += S(binomial(n + 1, j + 1)) * sum_action(rest, 1, 0, j);
        } else if (!states[2].l.empty()) {
            int n = states[2].l.front();
            auto rest = states;
            rest[2].l = tail(rest[2].l);
            answer = sum_action(rest, 0, 0, n);
            for (int j = 0; j <= modules_[1]->level(states[1]) + 3; j++)
                answer -=
                    from_rational<S>(half_binomial(2 * (1 - n), j)) * sum_action(rest, 1, 0, j - 1);
        } else {
            int target = -1;
            for (int j = 0; j < 3; j++)
                if (!states[j].g.empty()) {
                    target = j;
                    break;
                }
            if (target >= 0) {
                auto rest = states;
                rest[target].g = tail(rest[target].g);
                answer = gward(states, target, rest);
            } else {
                int g2 = states[1].ground, g3 = states[2].ground;
                S phase = power((S(-1) + S(Machine(0, 1))) / root(S(2)), g2 + g3);
                if (!f_) {
                    if (!g2 && !g3)
                        answer = S(1);
                    else if (g2 && g3)
                        answer = phase * S(eta_);
                } else {
                    if (!g2 && g3)
                        answer = phase;
                    else if (g2 && !g3)
                        answer = phase * S(Machine(0, eta_));
                }
            }
        }
        active_.erase(states);
        cache_[states] = answer;
        return answer;
    }
};

// Experimental convention: literal human SCA form times the fermion form,
// with only the Koszul sign. Production headers are not modified.
inline bool unphased_vertex = true;
inline bool apply_vertex_transport = true;
inline bool keep_ns_branch_sign = true;
// Multiplies the literal graded-product vertex. A and B are physical
// descendant parities, alpha is the second-slot physical ground parity,
// and auxiliary_second is mathsf{B}+mathsf{b}. The last term converts
// the literal Koszul order when the physical NS primary is odd.
template<class S> S vertex_transport(int A,int B,int alpha,int f,int p1,int auxiliary_second) {
    return power(S(Machine(0,-1)), A%2) * S(sign(f*(A+B+alpha)+p1*auxiliary_second));
}
template<class S> class UnphasedPhysicalForm {
    ScaWords<S> words_;
    std::array<std::unique_ptr<ScaModule<S>>,3> modules_;
    std::map<std::pair<int,int>,std::unique_ptr<ScaWard<S>>> forms_;
    int p_;
    S odd_ground_phase_;
public:
    UnphasedPhysicalForm(S b, const std::array<S,3> &momenta, int p):p_(p) {
        S q=b+S(1)/b, central=rational<S>(3,2)+S(3)*q*q;
        odd_ground_phase_=(S(-1)+S(Machine(0,1)))/root(S(2));
        for(int j=0;j<3;j++) {
            S h=q*q/S(8)-momenta[j]*momenta[j]/S(2)+(j?rational<S>(1,16):S(0));
            modules_[j]=std::make_unique<ScaModule<S>>(words_,h,central,momenta[j]/root(S(2)),j!=0,false);
        }
    }
    S value(const std::array<PBW,3> &states,int f,int eta) {
        auto key=std::make_pair(f,eta);
        if(!forms_.count(key)) forms_[key]=std::make_unique<ScaWard<S>>(words_,
            std::array<ScaModule<S>*,3>{modules_[0].get(),modules_[1].get(),modules_[2].get()},p_,f,eta);
        std::array<int,3> ids;
        for(int j=0;j<3;j++) {
            ScaWord w;
            for(int n:states[j].l) w.push_back({0,-2*n});
            for(int r:states[j].g) w.push_back({1,-(j?2:1)*r});
            ids[j]=2*words_.intern(w)+states[j].ground;
        }
        // FreeField uses (w+, exp(3 pi i/4) w-); only this ground-basis
        // conversion remains. There is no NS descendant contour phase.
        return power(odd_ground_phase_,states[1].ground+states[2].ground)*forms_.at(key)->value(ids);
    }
};
template <class S> class LowAnchors {
    struct Term {
        AuxState auxiliary;
        PBW physical;
        S coefficient;
    };
    std::array<std::unique_ptr<FreeField<S>>, 3> free_;
    std::array<std::unique_ptr<PBWModule<S>>, 3> pbw_;
    std::unique_ptr<FreeField<S>> reflected_;
    std::unique_ptr<PBWModule<S>> reflected_pbw_;
    std::map<std::array<int, 3>, std::vector<Term>> branches_;
    std::map<std::pair<int, int>, std::unique_ptr<PhysicalForm<S>>> forms_;
    FermionForm auxiliary_;
    int p_;
    UnphasedPhysicalForm<S> unphased_;
    const std::vector<Term> &branch(int slot, int label, int parity) {
        std::array<int, 3> key{slot, label, parity};
        auto found = branches_.find(key);
        if (found != branches_.end())
            return found->second;
        require((slot == 0 && std::abs(label) <= 2) || (slot > 0 && std::abs(label) <= 3),
                "direct anchors are restricted to low primaries");
        auto *module = free_[slot].get();
        auto *pbw = pbw_[slot].get();
        int n = label;
        if (!slot && label < 0) {
            module = reflected_.get();
            pbw = reflected_pbw_.get();
            n = -label;
        }
        auto expr = module->primary(n, parity);
        std::map<AuxState, Sparse<S>> groups;
        for (const auto &[id, v] : expr) {
            State state = module->states[id];
            AuxState a;
            for (int bit = 63; bit >= 0; bit--)
                if ((state.auxiliary >> bit) & 1)
                    a.modes.push_back(slot ? bit + 1 : 2 * bit + 1);
            a.ground = state.auxiliary_ground;
            state.auxiliary = 0;
            state.auxiliary_ground = 0;
            groups[a][module->intern(state)] += v;
        }
        std::vector<Term> terms;
        for (const auto &[a, v] : groups)
            for (const auto &[state, c] : pbw->from_fock(v))
                terms.push_back({a, state, c});
        return branches_.emplace(key, std::move(terms)).first->second;
    }

  public:
    LowAnchors(S b, const std::array<S, 3> &p, int primary_parity) : p_(primary_parity), unphased_(b,p,primary_parity) {
        for (int j = 0; j < 3; j++) {
            free_[j] = std::make_unique<FreeField<S>>(j != 0, b, p[j]);
            pbw_[j] = std::make_unique<PBWModule<S>>(*free_[j]);
        }
        reflected_ = std::make_unique<FreeField<S>>(false, b, -p[0]);
        reflected_pbw_ = std::make_unique<PBWModule<S>>(*reflected_);
    }
    S raw(std::array<int, 3> labels, int alpha, int gamma, int eta) {
        int f = ((labels[0] / 2 + alpha + gamma) % 2 + 2) % 2;
        auto key = std::make_pair(f, eta);
        if (!forms_.count(key))
            forms_[key] = std::make_unique<PhysicalForm<S>>(
                std::array<PBWModule<S> *, 3>{pbw_[0].get(), pbw_[1].get(), pbw_[2].get()}, f, eta,
                p_);
        auto &form = *forms_.at(key);
        S answer(0);
        for (const auto &a : branch(0, labels[0], 0))
            for (const auto &b : branch(1, labels[1], alpha))
                for (const auto &c : branch(2, labels[2], gamma)) {
                    AuxTriple states{a.auxiliary, b.auxiliary, c.auxiliary};
                    S aux = auxiliary_.complex_value<S>(states);
                    if (aux == S(0))
                        continue;
                    int phase = pbw_[0]->parity(a.physical) * aux_parity(0, a.auxiliary) +
                                (pbw_[1]->parity(b.physical) + p_) * aux_parity(2, c.auxiliary);
                    S physical;
                    if (unphased_vertex) {
                        phase = (pbw_[0]->parity(a.physical)+p_) *
                                    (aux_parity(1,b.auxiliary)+aux_parity(2,c.auxiliary)) +
                                pbw_[1]->parity(b.physical)*aux_parity(2,c.auxiliary);
                        physical = unphased_.value({a.physical,b.physical,c.physical}, f, eta);
                        if (apply_vertex_transport)
                            physical *= vertex_transport<S>(a.physical.g.size(), b.physical.g.size(),
                                b.physical.ground, f, p_, aux_parity(1,b.auxiliary));
                    } else physical = form.value({a.physical,b.physical,c.physical});
                    answer += a.coefficient*b.coefficient*c.coefficient*S(sign(phase))*aux*physical;
                }
        return answer;
    }
};
/* One descendant leg and two primary legs: the Virasoro Ward identity is a product. */
template <class S> S vertex_word(int slot, const Part &word, const std::array<S, 3> &h) {
    S value(1);
    int level = 0;
    for (auto it = word.rbegin(); it != word.rend(); ++it) {
        int n = *it;
        if (slot == 0)
            value *= h[0] + S(level) + S(n) * h[1] - h[2];
        else if (slot == 1)
            value *= S(sign(n)) * (h[1] + S(level) + S(n) * h[2] - h[0]);
        else
            value *= h[2] + S(level) + S(n) * h[1] - h[0];
        level += n;
    }
    return value;
}
} // namespace ramond
