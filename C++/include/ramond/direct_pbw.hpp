#pragma once
// Direct physical SCA PBW sewing, independent of branching and CCY.
// All mode indices below are TWICE the physical mode index.
#include "fermion.hpp"
#include <deque>
#include <memory>

namespace ramond {
struct ScaMode {
    int kind, twice;
    bool operator<(const ScaMode &b) const { return std::tie(kind, twice) < std::tie(b.kind, b.twice); }
    bool operator==(const ScaMode &b) const { return kind == b.kind && twice == b.twice; }
};
using ScaWord = std::vector<ScaMode>;
template<class S> using ScaExpr = std::vector<std::pair<int,S>>;
template<class S> inline void pbw_add(std::map<int,S> &out, int key, const S &value) {
    if (value == S(0)) return;
    auto [it, inserted] = out.try_emplace(key, value);
    if (!inserted) { it->second += value; if (it->second == S(0)) out.erase(it); }
}
template<class S> inline void pbw_madd(S &out, const S &a, const S &b, S &scratch) {
    if constexpr (std::is_same_v<S,MP>) {
        mpc_mul(scratch.data(), a.data(), b.data(), MPC_RNDNN);
        mpc_add(out.data(), out.data(), scratch.data(), MPC_RNDNN);
    } else out += a*b;
}
template<class S> inline void pbw_msub(S &out, const S &a, const S &b, S &scratch) {
    if constexpr (std::is_same_v<S,MP>) {
        mpc_mul(scratch.data(), a.data(), b.data(), MPC_RNDNN);
        mpc_sub(out.data(), out.data(), scratch.data(), MPC_RNDNN);
    } else out -= a*b;
}

template<class S> class ScaWords {
    std::map<ScaWord,int> ids_;
    std::unordered_map<int,ScaExpr<S>> canonical_;
    std::map<std::pair<int,int>,S> binomials_;
  public:
    struct Info { ScaWord word; int level2, parity, tail; };
    std::deque<Info> words;
    ScaWords() { words.push_back({{},0,0,0}); ids_[{}]=0; }
    int intern(const ScaWord &word) {
        auto it=ids_.find(word); if(it!=ids_.end()) return it->second;
        require(!word.empty(),"missing empty PBW word");
        int rest=intern(ScaWord(word.begin()+1,word.end()));
        int id=static_cast<int>(words.size());
        words.push_back({word,words[rest].level2-word.front().twice,
                         (words[rest].parity+word.front().kind)%2,rest});
        ids_.emplace(word,id); return id;
    }
    int prepend(ScaMode mode,int rest) {
        ScaWord word{mode}; const auto &tail=words[rest].word;
        word.insert(word.end(),tail.begin(),tail.end()); return intern(word);
    }
    static S bracket_coefficient(ScaMode a,ScaMode b) {
        if(!a.kind && !b.kind) return rational<S>(a.twice-b.twice,2);
        if(!a.kind && b.kind) return rational<S>(a.twice-2*b.twice,4);
        if(a.kind && !b.kind) return rational<S>(2*a.twice-b.twice,4);
        return S(2);
    }
    const ScaExpr<S> &canonical(int id) {
        auto found=canonical_.find(id); if(found!=canonical_.end()) return found->second;
        const auto &word=words[id].word;
        for(size_t j=0;j+1<word.size();j++) {
            const auto a=word[j],b=word[j+1];
            require(a.twice<0 && b.twice<0,"canonicalization needs negative modes");
            if(a==b && a.kind) {
                ScaWord reduced=word;
                reduced[j]={0,2*a.twice}; reduced.erase(reduced.begin()+j+1);
                ScaExpr<S> value=canonical(intern(reduced));
                return canonical_.emplace(id,std::move(value)).first->second;
            }
            if(!(b<a)) continue;
            std::map<int,S> result;
            ScaWord exchanged=word; std::swap(exchanged[j],exchanged[j+1]);
            int sign_swap=a.kind && b.kind ? -1 : 1;
            for(const auto &[k,c]:canonical(intern(exchanged))) pbw_add(result,k,S(sign_swap)*c);
            ScaWord reduced=word;
            reduced[j]={a.kind^b.kind,a.twice+b.twice}; reduced.erase(reduced.begin()+j+1);
            S factor=bracket_coefficient(a,b);
            for(const auto &[k,c]:canonical(intern(reduced))) pbw_add(result,k,factor*c);
            return canonical_.emplace(id,ScaExpr<S>(result.begin(),result.end())).first->second;
        }
        return canonical_.emplace(id,ScaExpr<S>{{id,S(1)}}).first->second;
    }
    const S &binomial_half(int numerator,int k) {
        auto key=std::make_pair(numerator,k); auto it=binomials_.find(key);
        if(it!=binomials_.end()) return it->second;
        return binomials_.emplace(key,from_rational<S>(half_binomial(numerator,k))).first->second;
    }
};

template<class S> class ScaModule {
    ScaWords<S> &words_;
    std::unordered_map<std::array<int,3>,ScaExpr<S>,Hash> actions_;
    ScaExpr<S> empty_;
    bool ramond_,native_;
    S g0_[2];
  public:
    S h,c,beta;
    ScaModule(ScaWords<S> &words,S weight,S central,S bet,bool is_r,bool native)
        :words_(words),ramond_(is_r),native_(native),h(weight),c(central),beta(bet) {
        S imag(Machine(0,1));
        if(native_) g0_[0]=g0_[1]=-imag*beta;
        else { g0_[0]=beta*(S(1)+imag)/root(S(2)); g0_[1]=imag*g0_[0]; }
    }
    const ScaExpr<S> &act(int kind,int mode2,int state) {
        int word=state/2,ground=state%2;
        if(mode2>words_.words[word].level2) return empty_;
        std::array<int,3> key{kind,mode2,state};
        auto found=actions_.find(key); if(found!=actions_.end()) return found->second;
        std::map<int,S> out;
        if(kind==0 && mode2==0) out[state]=h+rational<S>(words_.words[word].level2,2);
        else if(mode2<0) {
            for(const auto &[id,v]:words_.canonical(words_.prepend({kind,mode2},word))) out[2*id+ground]=v;
        } else if(!word) {
            if(kind && !mode2) {
                require(ramond_,"NS G0 is not defined"); out[1-ground]=g0_[ground];
            }
        } else {
            auto first=words_.words[word].word.front(); int rest=words_.words[word].tail;
            S exchange(kind && first.kind ? -1 : 1);
            for(const auto &[next,v]:act(kind,mode2,2*rest+ground))
                for(const auto &[id,factor]:words_.canonical(words_.prepend(first,next/2)))
                    pbw_add(out,2*id+next%2,exchange*v*factor);
            S factor=ScaWords<S>::bracket_coefficient({kind,mode2},first);
            for(const auto &[next,v]:act(kind^first.kind,mode2+first.twice,2*rest+ground))
                pbw_add(out,next,factor*v);
            if(mode2+first.twice==0 && kind==first.kind) {
                S central=kind ? c*rational<S>(mode2*mode2-1,12)
                               : c*rational<S>(mode2*mode2*mode2-4*mode2,96);
                pbw_add(out,2*rest+ground,central);
            }
        }
        return actions_.emplace(key,ScaExpr<S>(out.begin(),out.end())).first->second;
    }
    S inner(int left,int right) {
        std::map<int,S> current{{right,S(1)}};
        for(auto mode:words_.words[left/2].word) {
            std::map<int,S> next;
            for(const auto &[state,outer]:current)
                for(const auto &[target,v]:act(mode.kind,-mode.twice,state)) pbw_add(next,target,outer*v);
            current=std::move(next);
        }
        auto found=current.find(left%2);
        return found==current.end() ? S(0) : found->second;
    }
    size_t cache_size() const { return actions_.size(); }
};

template<class S> class ScaWard {
    ScaWords<S> &words_;
    std::array<ScaModule<S>*,3> modules_;
    int p_,f_,eta_;
    struct Entry { S value; bool active=true; };
    std::unordered_map<std::array<int,3>,Entry,Hash> cache_;
    S epsilon(int first,int third) {
        return S(Machine(0,-sign(p_+words_.words[first/2].parity+
                                  words_.words[third/2].parity+third%2+1)));
    }
    int level(int state) const { return words_.words[state/2].level2; }
    int tail(int state) const { return 2*words_.words[state/2].tail+state%2; }
    void add_action(S &answer,std::array<int,3> states,int slot,int kind,int mode2,const S &factor) {
        if(factor==S(0)) return;
        const auto &action=modules_[slot]->act(kind,mode2,states[slot]);
        if(action.empty()) return;
        S weighted,scratch;
        for(const auto &[next,c]:action) {
            states[slot]=next;
            if constexpr(std::is_same_v<S,MP>) mpc_mul(weighted.data(),factor.data(),c.data(),MPC_RNDNN);
            else weighted=factor*c;
            pbw_madd(answer,weighted,value(states),scratch);
            action_terms++;
        }
    }
  public:
    size_t hits=0,misses=0,action_terms=0;
    ScaWard(ScaWords<S> &words,std::array<ScaModule<S>*,3> modules,int p,int f,int eta)
        :words_(words),modules_(modules),p_(p),f_(f),eta_(eta) {}
    const S &value(const std::array<int,3> &s) {
        auto found=cache_.find(s);
        if(found!=cache_.end()) { require(!found->second.active,"cyclic direct SCA Ward recursion"); hits++; return found->second.value; }
        misses++;
        auto &entry=cache_.try_emplace(s).first->second;
        S answer(0); auto rest=s;
        const auto &w1=words_.words[s[0]/2].word;
        const auto &w2=words_.words[s[1]/2].word;
        const auto &w3=words_.words[s[2]/2].word;
        auto bin=[&](int a,int k)->const S& { return words_.binomial_half(a,k); };
        if(!w1.empty() && w1.front().kind) {
            int r2=-w1.front().twice; rest[0]=tail(s[0]); S eps=epsilon(rest[0],s[2]);
            int maximum=std::max({0,r2+level(rest[0]),level(s[1]),level(s[2])})/2+4;
            for(int j=0;j<=maximum;j++) {
                add_action(answer,rest,1,1,2*j,bin(r2,j));
                if(j) add_action(answer,rest,0,1,2*j-r2,S(-sign(j))*bin(1,j));
                add_action(answer,rest,2,1,r2-1+2*j,-eps*S(sign(j))*bin(1,j));
            }
        } else if(!w2.empty()) {
            auto mode=w2.front(); int n=-mode.twice/2; rest[1]=tail(s[1]);
            if(!mode.kind && n==1) {
                S exponent=modules_[0]->h-modules_[1]->h-modules_[2]->h+
                    rational<S>(level(s[0])-level(rest[1])-level(s[2]),2);
                answer=exponent*value(rest);
            } else if(!mode.kind) {
                int maximum=std::max({0,level(s[0])-2*n,level(s[2])+2})/2+4;
                for(int j=0;j<=maximum;j++) {
                    const S &b=bin(2*(n-2+j),n-2);
                    add_action(answer,rest,0,0,2*(n+j),b);
                    add_action(answer,rest,2,0,2*(j-1),S(sign(n))*b);
                }
            } else {
                S eps=epsilon(s[0],s[2]);
                int maximum=std::max({0,2*n+level(rest[1]),level(s[0])-2*n+1,level(s[2])})/2+4;
                for(int j=0;j<=maximum;j++) {
                    const S &b=bin(1-2*n,j);
                    add_action(answer,rest,0,1,2*j+2*n-1,S(sign(j))*b);
                    add_action(answer,rest,2,1,2*j,eps*S(sign(n+j))*b);
                    if(j) add_action(answer,rest,1,1,2*(j-n),-bin(1,j));
                }
            }
        } else if(!w1.empty()) {
            require(!w1.front().kind,"unreduced outer G mode");
            int n=-w1.front().twice/2; rest[0]=tail(s[0]);
            add_action(answer,rest,2,0,2*n,S(1));
            for(int j=-1;j<=n;j++) add_action(answer,rest,1,0,2*j,bin(2*(n+1),j+1));
        } else if(!w3.empty()) {
            auto mode=w3.front(); int n=-mode.twice/2; rest[2]=tail(s[2]);
            if(!mode.kind) {
                S exponent=modules_[2]->h+S(n)*modules_[1]->h-modules_[0]->h+rational<S>(level(rest[2]),2);
                answer=exponent*value(rest);
            } else {
                S eps=epsilon(s[0],rest[2]);
                int maximum=std::max(0,2*n+level(rest[2]))/2+4;
                for(int j=0;j<=maximum;j++) {
                    add_action(answer,rest,1,1,2*j,bin(1-2*n,j)/eps);
                    add_action(answer,rest,0,1,2*n-1+2*j,S(-sign(j))*bin(1,j)/eps);
                    if(j) add_action(answer,rest,2,1,2*(j-n),S(-sign(j))*bin(1,j));
                }
            }
        } else {
            int g2=s[1]%2,g3=s[2]%2;
            if((g2+g3)%2==f_) {
                if(!g2) answer=S(1);
                else answer=g3 ? S(eta_) : S(Machine(0,eta_));
            }
        }
        require(finite(answer),"nonfinite direct SCA Ward coefficient");
        entry.value=std::move(answer); entry.active=false; return entry.value;
    }
    size_t cache_size() const { return cache_.size(); }
    size_t cache_buckets() const { return cache_.bucket_count(); }
    static size_t entry_pair_bytes() { return sizeof(std::pair<const std::array<int,3>,Entry>); }
};

template<class S> std::vector<S> pbw_inverse(std::vector<S> a,int n) {
    std::vector<int> permutation(n); for(int i=0;i<n;i++) permutation[i]=i;
    S scratch;
    for(int k=0;k<n;k++) {
        int pivot=k;
        for(int i=k+1;i<n;i++) if(magnitude(a[i*n+k])>magnitude(a[pivot*n+k])) pivot=i;
        require(a[pivot*n+k]!=S(0),"singular physical Gram matrix");
        if(pivot!=k) { for(int j=0;j<n;j++) std::swap(a[k*n+j],a[pivot*n+j]); std::swap(permutation[k],permutation[pivot]); }
        for(int i=k+1;i<n;i++) {
            a[i*n+k]/=a[k*n+k];
            for(int j=k+1;j<n;j++) pbw_msub(a[i*n+j],a[i*n+k],a[k*n+j],scratch);
        }
    }
    std::vector<S> inverse(size_t(n)*n),x(n);
    for(int col=0;col<n;col++) {
        for(int i=0;i<n;i++) {
            x[i]=S(permutation[i]==col ? 1 : 0);
            for(int j=0;j<i;j++) pbw_msub(x[i],a[i*n+j],x[j],scratch);
        }
        for(int i=n-1;i>=0;i--) {
            for(int j=i+1;j<n;j++) pbw_msub(x[i],a[i*n+j],x[j],scratch);
            x[i]/=a[i*n+i]; inverse[i*n+col]=x[i];
        }
    }
    return inverse;
}

template<class S> class DirectPBW {
    ScaWords<S> words_;
    std::array<std::unique_ptr<ScaModule<S>>,3> gram_modules_,ward_modules_;
    std::unique_ptr<ScaWard<S>> forms_[2];
    struct Edge { std::vector<int> states; std::vector<S> inverse; };
    std::map<std::array<int,3>,Edge> edges_;
    std::array<S,3> phases_;
    int p_,f_;
    bool opposite_;
    const Edge &edge(int slot,int level2,int parity) {
        std::array<int,3> key{slot,level2,parity};
        auto found=edges_.find(key); if(found!=edges_.end()) return found->second;
        double start=seconds(); Edge result;
        int level=slot ? level2/2 : level2;
        for(int l=0;l<=(slot?level:level/2);l++)
            for(const auto &ls:partitions(l))
                for(const auto &gs:partitions(level-(slot?l:2*l),true,!slot)) {
                    ScaWord word;
                    for(int n:ls) word.push_back({0,-2*n});
                    for(int n:gs) word.push_back({1,slot ? -2*n : -n});
                    int id=words_.intern(word),g=slot ? (parity+gs.size())%2 : 0;
                    if((words_.words[id].parity+g)%2==parity) result.states.push_back(2*id+g);
                }
        times.metadata+=seconds()-start; start=seconds();
        int n=static_cast<int>(result.states.size()); std::vector<S> gram(size_t(n)*n);
        for(int i=0;i<n;i++) for(int j=0;j<=i;j++) {
            S v=gram_modules_[slot]->inner(result.states[i],result.states[j]);
            gram[i*n+j]=v; gram[j*n+i]=v; counts.gram_entries++;
        }
        times.gram_entries+=seconds()-start; start=seconds();
        result.inverse=pbw_inverse(std::move(gram),n);
        times.gram_inverse+=seconds()-start; counts.gram_matrices++;
        return edges_.emplace(key,std::move(result)).first->second;
    }
    S contract(const std::vector<S> &left,const std::array<const Edge*,3> &edges,const std::vector<S> &right) {
        std::array<size_t,3> d{edges[0]->states.size(),edges[1]->states.size(),edges[2]->states.size()};
        size_t volume=d[0]*d[1]*d[2];
        std::vector<S> current=right,next(volume); S scratch;
        // Each axis contraction writes directly in (a,b,c) order. No cached permutations.
        for(int axis=0;axis<3;axis++) {
            size_t stride=axis==0 ? d[1]*d[2] : axis==1 ? d[2] : 1;
            size_t n=d[axis]; const auto &inverse=edges[axis]->inverse;
            for(size_t base=0;base<volume;base+=n*stride)
                for(size_t offset=0;offset<stride;offset++)
                    for(size_t i=0;i<n;i++) {
                        S &v=next[base+i*stride+offset]; v=S(0);
                        for(size_t j=0;j<n;j++) pbw_madd(v,inverse[i*n+j],current[base+j*stride+offset],scratch);
                    }
            current.swap(next);
        }
        S answer(0);
        for(size_t i=0;i<volume;i++) pbw_madd(answer,left[i],current[i],scratch);
        counts.contraction_products+=volume*(d[0]+d[1]+d[2]+1);
        counts.contractions++; return answer;
    }
  public:
    struct Timings { double metadata=0,gram_entries=0,gram_inverse=0,vertices=0,contractions=0; } times;
    struct Counts { size_t gram_entries=0,gram_matrices=0,vertex_entries=0,reused_vertex_tensors=0,
                          contractions=0,contraction_products=0; } counts;
    DirectPBW(S b,const std::array<S,3> &p,int primary_parity,int form_parity,int eta,bool opposite)
        :p_(primary_parity),f_(form_parity),opposite_(opposite) {
        S q=b+S(1)/b,c=rational<S>(3,2)+S(3)*q*q;
        for(int j=0;j<3;j++) {
            S h=q*q/S(8)-p[j]*p[j]/S(2)+(j?rational<S>(1,16):S(0));
            gram_modules_[j]=std::make_unique<ScaModule<S>>(words_,h,c,p[j]/root(S(2)),j!=0,true);
            ward_modules_[j]=std::make_unique<ScaModule<S>>(words_,h,c,p[j]/root(S(2)),j!=0,false);
        }
        std::array<ScaModule<S>*,3> modules{ward_modules_[0].get(),ward_modules_[1].get(),ward_modules_[2].get()};
        forms_[0]=std::make_unique<ScaWard<S>>(words_,modules,p_,f_,eta);
        if(opposite_) forms_[1]=std::make_unique<ScaWard<S>>(words_,modules,p_,f_,-eta);
        for(int k=0;k<3;k++) phases_[k]=power((S(-1)+S(Machine(0,1)))/root(S(2)),k);
    }
    std::array<S,8> coefficient(std::array<int,3> levels2) {
        std::array<S,8> result{};
        for(int parity=0;parity<2;parity++) {
            std::array<int,3> pr{levels2[0]%2,parity,(f_+levels2[0]+parity)%2};
            std::array<const Edge*,3> edges{&edge(0,levels2[0],pr[0]),&edge(1,levels2[1],pr[1]),&edge(2,levels2[2],pr[2])};
            size_t volume=edges[0]->states.size()*edges[1]->states.size()*edges[2]->states.size();
            if(!volume) continue;
            std::vector<S> tensors[2]; double tick=seconds();
            for(int v=0;v<(opposite_?2:1);v++) {
                tensors[v].reserve(volume);
                for(int a:edges[0]->states) for(int b:edges[1]->states) for(int c:edges[2]->states)
                    tensors[v].push_back(phases_[b%2+c%2]*forms_[v]->value({a,b,c}));
                counts.vertex_entries+=volume;
            }
            if(!opposite_) counts.reused_vertex_tensors++;
            times.vertices+=seconds()-tick; tick=seconds();
            int index=((pr[0]+p_)%2)|(pr[1]<<1)|(pr[2]<<2);
            result[index]=S(theta_sign(index))*contract(tensors[0],edges,tensors[opposite_?1:0]);
            times.contractions+=seconds()-tick;
        }
        return result;
    }
    size_t ward_entries() const { return forms_[0]->cache_size()+(opposite_?forms_[1]->cache_size():0); }
    size_t ward_hits() const { return forms_[0]->hits+(opposite_?forms_[1]->hits:0); }
    size_t ward_terms() const { return forms_[0]->action_terms+(opposite_?forms_[1]->action_terms:0); }
    size_t ward_buckets() const { return forms_[0]->cache_buckets()+(opposite_?forms_[1]->cache_buckets():0); }
    size_t word_count() const { return words_.words.size(); }
    size_t action_entries() const {
        size_t n=0; for(int j=0;j<3;j++) n+=gram_modules_[j]->cache_size()+ward_modules_[j]->cache_size(); return n;
    }
};
} // namespace ramond
