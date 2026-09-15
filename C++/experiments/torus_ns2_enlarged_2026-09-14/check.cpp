// Direct enlarged NS--NS torus two-point sewing in the draft's pants frame.
// NS punctures are at infinity; glue the two R punctures at 1 pairwise,
// and the two at 0 pairwise. No convolution or vanishing projection is used.
#include "ramond/direct_pbw.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace ramond;
using S = MP;

struct TensorState {
    int physical, physical_level, physical_parity;
    AuxState auxiliary;
};
struct Edge {
    std::vector<TensorState> states;
    std::vector<S> inverse;
};

class TorusCheck {
    ScaWords<S> words;
    std::array<std::unique_ptr<ScaModule<S>>,2> native, human, external;
    std::array<std::array<std::unique_ptr<ScaWard<S>>,2>,2> forms;
    FermionForm auxiliary_form;
    std::array<S,3> phase;
    std::map<std::array<int,3>,Edge> edge_cache;
    std::map<std::pair<int,int>,std::vector<int>> physical_bases;
    int f;

    const std::vector<int>& basis(int level,int parity) {
        auto key=std::make_pair(level,parity);
        auto it=physical_bases.find(key); if(it!=physical_bases.end()) return it->second;
        std::vector<int> result;
        for(int l=0;l<=level;l++) for(const auto &ls:partitions(l))
            for(const auto &gs:partitions(level-l,true)) {
                ScaWord word;
                for(int n:ls) word.push_back({0,-2*n});
                for(int n:gs) word.push_back({1,-2*n});
                int ground=(parity+gs.size())%2;
                result.push_back(2*words.intern(word)+ground);
            }
        return physical_bases.emplace(key,std::move(result)).first->second;
    }
    const Edge& edge(int slot,int level,int total_parity) {
        std::array<int,3> key{slot,level,total_parity};
        auto it=edge_cache.find(key); if(it!=edge_cache.end()) return it->second;
        Edge result;
        for(int l=0;l<=level;l++) for(int p=0;p<2;p++)
            for(const auto &modes:partitions(level-l,true)) {
                int ground=(total_parity+p+modes.size())%2;
                for(int physical:basis(l,p)) result.states.push_back({physical,l,p,{modes,ground}});
            }
        int n=result.states.size(); std::vector<S> gram(size_t(n)*n);
        for(int i=0;i<n;i++) for(int j=0;j<=i;j++) {
            const auto &a=result.states[i],&b=result.states[j];
            if(a.physical_level!=b.physical_level || !(a.auxiliary==b.auxiliary)) continue;
            // Auxiliary BPZ norm is (-1)^(number of oscillators + ground).
            S value=S(sign(a.auxiliary.modes.size()+a.auxiliary.ground))*
                    native[slot]->inner(a.physical,b.physical);
            gram[size_t(i)*n+j]=gram[size_t(j)*n+i]=value;
        }
        result.inverse=pbw_inverse(std::move(gram),n);
        return edge_cache.emplace(key,std::move(result)).first->second;
    }
    S vertex(int which,int sign_index,const TensorState &a,const TensorState &b) {
        S free=auxiliary_form.complex_value<S>({AuxState{{},0},a.auxiliary,b.auxiliary});
        if(free==S(0)) return S(0);
        S physical=phase[a.physical%2+b.physical%2]*
                   forms[which][sign_index]->value({0,a.physical,b.physical});
        int koszul=a.physical_parity*aux_parity(2,b.auxiliary);
        return S(sign(koszul))*free*physical;
    }
    S contract(const std::vector<S> &left,const Edge &a,const Edge &b,const std::vector<S> &right) {
        size_t n=a.states.size(),m=b.states.size();
        std::vector<S> first(n*m),second(n*m); S scratch,answer(0);
        for(size_t i=0;i<n;i++) for(size_t k=0;k<m;k++)
            for(size_t j=0;j<n;j++) pbw_madd(first[i*m+k],a.inverse[i*n+j],right[j*m+k],scratch);
        for(size_t i=0;i<n;i++) for(size_t k=0;k<m;k++)
            for(size_t l=0;l<m;l++) pbw_madd(second[i*m+k],b.inverse[k*m+l],first[i*m+l],scratch);
        for(size_t i=0;i<n*m;i++) pbw_madd(answer,left[i],second[i],scratch);
        return answer;
    }
  public:
    TorusCheck(int form_parity):f(form_parity) {
        S b=parse<S>("7/5"),q=b+S(1)/b,c=rational<S>(3,2)+S(3)*q*q;
        std::array<S,2> p{parse<S>("13/29"),parse<S>("17/31")};
        std::array<S,2> pext{parse<S>("11/23"),parse<S>("19/37")};
        for(int i=0;i<2;i++) {
            S h=q*q/S(8)-p[i]*p[i]/S(2)+rational<S>(1,16);
            native[i]=std::make_unique<ScaModule<S>>(words,h,c,p[i]/root(S(2)),true,true);
            human[i]=std::make_unique<ScaModule<S>>(words,h,c,p[i]/root(S(2)),true,false);
            external[i]=std::make_unique<ScaModule<S>>(words,q*q/S(8)-pext[i]*pext[i]/S(2),c,S(0),false,false);
        }
        for(int i=0;i<2;i++) for(int j=0;j<2;j++)
            forms[i][j]=std::make_unique<ScaWard<S>>(words,
                std::array<ScaModule<S>*,3>{external[i].get(),human[0].get(),human[1].get()},0,f,j?-1:1);
        for(int i=0;i<3;i++) phase[i]=power((S(-1)+S(Machine(0,1)))/root(S(2)),i);
    }
    std::array<std::array<S,4>,2> coefficient(int level2,int level3) {
        std::array<std::array<S,4>,2> result{};
        for(int e2=0;e2<2;e2++) {
            int e3=(f+e2)%2; const Edge &a=edge(0,level2,e2),&b=edge(1,level3,e3);
            std::vector<S> left,right[2];
            for(const auto &s:a.states) for(const auto &t:b.states) {
                left.push_back(vertex(0,0,s,t));
                for(int eta=0;eta<2;eta++) right[eta].push_back(vertex(1,eta,s,t));
            }
            for(int eta=0;eta<2;eta++)
                result[eta][e2+2*e3]=S(sign(e2*e3))*contract(left,a,b,right[eta]);
        }
        return result;
    }
};

int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: check OUTPUT_JSON"); MP::precision(40);
        double start=seconds(); constexpr int cutoff=3;
        std::ofstream out(argv[1]); require(bool(out),"cannot open output");
        out<<"{\"method\":\"direct tensor-product Gram and vertex sewing; no convolution\","
           <<"\"precision_bits\":136,\"dps\":40,\"q2_q3_cutoffs\":[3,3],"
           <<"\"b\":\"7/5\",\"external_momenta\":[\"11/23\",\"19/37\"],"
           <<"\"internal_momenta\":[\"13/29\",\"17/31\"],\"p_external\":[0,0],\"runs\":[";
        double negative_max=0;
        for(int f=0;f<2;f++) {
            if(f) out<<','; TorusCheck engine(f);
            out<<"{\"f\":"<<f<<",\"coefficients\":["; bool first=true;
            for(int a=0;a<=cutoff;a++) for(int b=0;b<=cutoff;b++) {
                auto value=engine.coefficient(a,b);
                if(!first) out<<','; first=false;
                out<<"{\"levels\":["<<a<<','<<b<<"],\"eta_eta_prime_plus\":[";
                for(int i=0;i<4;i++) { if(i) out<<','; out<<json_number(value[0][i]); }
                out<<"],\"eta_eta_prime_minus\":[";
                for(int i=0;i<4;i++) {
                    if(i) out<<','; out<<json_number(value[1][i]);
                    negative_max=std::max(negative_max,magnitude(value[1][i]));
                }
                out<<"]}";
            }
            out<<"]}";
        }
        out<<"],\"maximum_absolute_negative_coefficient\":"<<std::setprecision(17)<<negative_max
           <<",\"runtime_seconds\":"<<seconds()-start<<"}\n";
        require(bool(out),"failed output write");
        std::cout<<"direct enlarged: "<<seconds()-start<<" s, max negative coefficient "<<negative_max<<'\n';
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
