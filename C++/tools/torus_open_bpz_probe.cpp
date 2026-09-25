#include "ramond/direct_pbw.hpp"
#include <iomanip>
#include <iostream>
using namespace ramond;

// Cut the third edge of the NS--R--R theta channel.  The two remaining
// inverse Grams and both vertices use the same BPZ inversion z -> 1/z.
int main() {
    try {
        using S=Machine;
        const S I(0,1), b(1), Q=b+S(1)/b;
        const S c=S(1.5)+S(3)*Q*Q;
        const std::array<S,3> p{I*S(0.22),I*S(0.27),I*S(0.31)};
        ScaWords<S> words;
        std::array<std::unique_ptr<ScaModule<S>>,3> ward;
        for(int slot=0;slot<3;slot++) {
            S h=Q*Q/S(8)-p[slot]*p[slot]/S(2)
                +(slot?S(1)/S(16):S(0));
            ward[slot]=std::make_unique<ScaModule<S>>(
                words,h,c,p[slot]/root(S(2)),slot!=0,false);
        }
        auto inner=[&](int slot,int left,int right) {
            const auto &word=words.words[left/2].word;
            int odd=0;
            std::map<int,S> current{{right,S(1)}};
            for(auto mode:word) {
                odd+=mode.kind;
                std::map<int,S> next;
                for(const auto &[state,outer]:current)
                    for(const auto &[target,value]:
                        ward[slot]->act(mode.kind,-mode.twice,state))
                        pbw_add(next,target,outer*value);
                current=std::move(next);
            }
            auto found=current.find(left%2);
            if(found==current.end())return S(0);
            int ground_parity=slot?left%2:0;
            return found->second*S(slot&&left%2?-1:1)
                *S(sign(odd*ground_parity+odd*(odd-1)/2))*power(I,odd);
        };
        struct Basis {std::vector<int> states;std::vector<S> inverse;};
        auto basis=[&](int slot,int twice_level,int parity) {
            Basis out;
            int level=slot?twice_level/2:twice_level;
            for(int l=0;l<=(slot?level:level/2);l++)
                for(const auto &ls:partitions(l))
                    for(const auto &gs:partitions(level-(slot?l:2*l),true,!slot)) {
                        ScaWord word;
                        for(int mode:ls)word.push_back({0,-2*mode});
                        for(int mode:gs)word.push_back({1,slot?-2*mode:-mode});
                        int id=words.intern(word);
                        int ground=slot?(parity+int(gs.size()))%2:0;
                        if((words.words[id].parity+ground)%2==parity)
                            out.states.push_back(2*id+ground);
                    }
            int count=int(out.states.size());
            std::vector<S> matrix(size_t(count)*count);
            for(int i=0;i<count;i++)for(int j=0;j<count;j++)
                matrix[size_t(i)*count+j]=
                    inner(slot,out.states[i],out.states[j]);
            out.inverse=pbw_inverse(std::move(matrix),count);
            return out;
        };
        std::array<ScaModule<S>*,3> modules{ward[0].get(),ward[1].get(),ward[2].get()};
        std::cout<<std::setprecision(17);
        for(int n=0;n<=2;n++)for(int r=0;r<=1;r++)
        for(int parity=0;parity<2;parity++) {
            auto ns=basis(0,n,n%2),ram=basis(1,2*r,parity);
            int nn=int(ns.states.size()),nr=int(ram.states.size());
            for(int f=0;f<2;f++)for(int eta:{1,-1}) {
                ScaWard<S> form(words,modules,0,f,eta);
                for(int left_ground=0;left_ground<2;left_ground++)
                for(int right_ground=0;right_ground<2;right_ground++)
                for(int external_mode=0;external_mode<3;external_mode++) {
                    int left=2*words.intern(external_mode==1?ScaWord{{1,-2}}:ScaWord{})
                             +left_ground;
                    int right=2*words.intern(external_mode==2?ScaWord{{1,-2}}:ScaWord{})
                              +right_ground;
                    S value=0;
                    for(int i=0;i<nn;i++)for(int j=0;j<nn;j++)
                    for(int a=0;a<nr;a++)for(int d=0;d<nr;d++)
                        value+=form.value({ns.states[i],ram.states[a],left})
                              *ns.inverse[size_t(i)*nn+j]
                              *ram.inverse[size_t(a)*nr+d]
                              *form.value({ns.states[j],ram.states[d],right});
                    // Two sewn odd edges cross once; both vertices have
                    // a first-slot phase i when the NS state is odd.
                    value*=S(sign((n%2)*parity))*power(I,2*(n%2));
                    std::cout<<n<<' '<<r<<' '<<parity<<' '<<f<<' '<<eta<<' '
                             <<left_ground<<' '<<right_ground<<' '
                             <<external_mode<<' '<<value.real()<<' '
                             <<value.imag()<<'\n';
                }
            }
        }
    } catch(const std::exception &e) {
        std::cerr<<e.what()<<'\n';return 1;
    }
}
