#include "ramond/direct_pbw.hpp"
#include "ramond/fermion.hpp"
#include "../experiments/total_level6_2026-09-16/ns_local.hpp"
#include <iostream>
using namespace ramond;

// Residues in the BPZ chart w=1/z, theta_w=-i theta_z/z.  The first-slot
// phase belongs to the local three-point form, not to the completed block.
int main() {
    try {
        MP::precision(40);
        const MP I(Machine(0,1)), b=parse<MP>("7/5"), Q=b+MP(1)/b;
        const MP c=rational<MP>(3,2)+MP(3)*Q*Q;
        const std::array<MP,3> P{parse<MP>("11/23"),parse<MP>("13/29"),parse<MP>("17/31")};
        ScaWords<MP> words;
        std::array<std::unique_ptr<ScaModule<MP>>,3> modules;
        for(int s=0;s<3;s++) {
            MP h=Q*Q/MP(8)-P[s]*P[s]/MP(2)+(s?rational<MP>(1,16):MP(0));
            modules[s]=std::make_unique<ScaModule<MP>>(words,h,c,P[s]/root(MP(2)),s!=0,false);
        }
        std::array<ScaModule<MP>*,3> pointers{modules[0].get(),modules[1].get(),modules[2].get()};
        std::vector<int> ns{0},r{0,1};
        for(ScaWord w:{ScaWord{{1,-1}},ScaWord{{0,-2}},ScaWord{{1,-3}}}) {
            int id=words.intern(w);
            ns.push_back(2*id);
        }
        for(ScaWord w:{ScaWord{{0,-2}},ScaWord{{1,-2}}}) {
            int id=words.intern(w);r.push_back(2*id);r.push_back(2*id+1);
        }
        double maximum=0,physical_maximum=0,auxiliary_maximum=0;
        size_t physical_cases=0,auxiliary_cases=0,ns_physical_cases=0,ns_auxiliary_cases=0;
        auto audit=[&](MP residual,double scale,size_t &count) {
            maximum=std::max(maximum,magnitude(residual)/std::max(1.,scale));count++;
        };
        for(int primary=0;primary<2;primary++)
        for(int f=0;f<2;f++)for(int eta:{-1,1}) {
            ScaWard<MP> form(words,pointers,primary,f,eta);
            auto parity=[&](int state) {return (words.words[state/2].parity+state%2)%2;};
            auto first_parity=[&](int state) {return (primary+parity(state))%2;};
            auto value=[&](std::array<int,3> s) {
                return (first_parity(s[0])?I:MP(1))*form.value(s);
            };
            for(int x:ns)for(int y:r)for(int z:r) {
                std::array<int,3> s{x,y,z};
                if(words.words[x/2].level2+words.words[y/2].level2+words.words[z/2].level2>4)continue;
                for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++) {
                    MP residual=0;double scale=0;
                    auto add=[&](int slot,int mode2,MP factor) {
                        for(auto [state,co]:modules[slot]->act(1,mode2,s[slot])) {
                            auto t=s;t[slot]=state;MP term=factor*co*value(t);
                            residual+=term;scale+=magnitude(term);
                        }
                    };
                    int bound=words.words[x/2].level2+words.words[y/2].level2+
                              words.words[z/2].level2+std::abs(m)+std::abs(n)+8;
                    for(int j=0;j<=bound;j++) {
                        MP a=from_rational<MP>(half_binomial(2*n+1,j));
                        add(0,2*j-2*m-2*n-1,MP(sign(j))*a);
                        add(1,2*(n+j),-I*MP(sign(first_parity(x)))*
                                              from_rational<MP>(half_binomial(2*m+1,j)));
                        add(2,2*(m+j),MP(sign(parity(x)+f+parity(y)+n+j))*a);
                    }
                    double err=magnitude(residual)/std::max(1.,scale);
                    audit(residual,scale,physical_cases);
                    physical_maximum=std::max(physical_maximum,err);
                }
            }
        }
        FermionForm free;
        std::vector<AuxState> ans{{{},0},{{1},0}},ar{{{},0}};
        ar.push_back({{},1});ar.push_back({{1},0});ar.push_back({{1},1});
        for(auto a:ans)for(auto bb:ar)for(auto cc:ar)for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++) {
            AuxTriple s{a,bb,cc};MP residual=0;double scale=0;
            auto value=[&](AuxTriple t) {
                return (aux_parity(0,t[0])?I:MP(1))*free.complex_value<MP>(t);
            };
            auto add=[&](int slot,int mode,MP factor) {
                auto [next,co]=aux_act(slot,mode,s[slot]);if(co==0)return;
                MP action=mode?from_rational<MP>(co):
                    MP(sign(s[slot].modes.size()))/root(MP(2));
                auto t=s;t[slot]=next;MP term=factor*action*value(t);
                residual+=term;scale+=magnitude(term);
            };
            int bound=std::abs(m)+std::abs(n)+8;
            for(int j=0;j<=bound;j++) {
                MP d=from_rational<MP>(half_binomial(2*n-1,j));
                add(0,2*j-2*m-2*n+1,MP(sign(j))*d);
                add(1,n+j,I*MP(sign(aux_parity(0,a)))*
                              from_rational<MP>(half_binomial(2*m-1,j)));
                add(2,m+j,MP(sign(aux_parity(0,a)+aux_parity(1,bb)+n+j))*d);
            }
            double err=magnitude(residual)/std::max(1.,scale);
            audit(residual,scale,auxiliary_cases);
            auxiliary_maximum=std::max(auxiliary_maximum,err);
        }
        std::array<std::unique_ptr<ScaModule<MP>>,3> ns_modules;
        for(int s=0;s<3;s++)
            ns_modules[s]=std::make_unique<ScaModule<MP>>(
                words,Q*Q/MP(8)-P[s]*P[s]/MP(2),c,MP(0),false,false);
        level6::NSWard<MP> ns_form(words,
             {ns_modules[0].get(),ns_modules[1].get(),ns_modules[2].get()});
        for(int x:ns)for(int y:ns)for(int z:ns) {
            if(words.words[x/2].level2+words.words[y/2].level2+words.words[z/2].level2>4)continue;
            std::array<int,3> s{x,y,z};
            auto parity=[&](int state){return words.words[state/2].parity;};
            auto value=[&](std::array<int,3> t){
                return (parity(t[0])?I:MP(1))*ns_form.value(t);
            };
            for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++) {
                MP residual=0;double scale=0;
                auto add=[&](int slot,int mode2,MP factor) {
                    for(auto [state,co]:ns_modules[slot]->act(1,mode2,s[slot])) {
                        auto t=s;t[slot]=state;MP term=factor*co*value(t);
                        residual+=term;scale+=magnitude(term);
                    }
                };
                int bound=words.words[x/2].level2+words.words[y/2].level2+
                          words.words[z/2].level2+std::abs(m)+std::abs(n)+8;
                for(int j=0;j<=bound;j++) {
                    MP a=from_rational<MP>(half_binomial(2*n,j));
                    add(0,2*j-2*m-2*n+1,MP(sign(j))*a);
                    add(1,2*n+2*j-1,-I*MP(sign(parity(x)))*
                        from_rational<MP>(half_binomial(2*m,j)));
                    add(2,2*m+2*j-1,-I*MP(sign(parity(x)+parity(y)+n+j))*a);
                }
                audit(residual,scale,ns_physical_cases);
            }
        }
        std::vector<AuxState> ns_aux{{{},0},{{1},0},{{3},0},{{3,1},0}};
        for(auto a:ns_aux)for(auto bb:ns_aux)for(auto cc:ns_aux) {
            if(sum(a.modes)+sum(bb.modes)+sum(cc.modes)>4)continue;
            AuxTriple s{a,bb,cc};
            auto value=[&](AuxTriple t) {
                return (aux_parity(0,t[0])?I:MP(1))*MP(level6::ns_fermion(t));
            };
            for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++) {
                MP residual=0;double scale=0;
                auto add=[&](int slot,int mode,MP factor) {
                    auto [next,co]=aux_act(0,mode,s[slot]);if(co==0)return;
                    auto t=s;t[slot]=next;MP term=factor*from_rational<MP>(co)*value(t);
                    residual+=term;scale+=magnitude(term);
                };
                for(int j=0;j<=12;j++) {
                    MP d=from_rational<MP>(half_binomial(2*n,j));
                    add(0,2*j-2*m-2*n-1,MP(sign(j))*d);
                    add(1,2*n+2*j+1,I*MP(sign(aux_parity(0,a)))*
                        from_rational<MP>(half_binomial(2*m,j)));
                    add(2,2*m+2*j+1,I*MP(sign(aux_parity(0,a)+aux_parity(1,bb)+n+j))*d);
                }
                audit(residual,scale,ns_auxiliary_cases);
            }
        }
        std::cerr<<"physical_maximum="<<physical_maximum<<" auxiliary_maximum="<<auxiliary_maximum<<'\n';
        require(maximum<1e-30,"one-over-z graded BPZ Ward identity failed");
        std::cout<<"{\"physical_residues\":"<<physical_cases
                 <<",\"auxiliary_residues\":"<<auxiliary_cases
                 <<",\"all_NS_physical_residues\":"<<ns_physical_cases
                 <<",\"all_NS_auxiliary_residues\":"<<ns_auxiliary_cases
                 <<",\"maximum_scaled_error\":"<<maximum<<"}\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
