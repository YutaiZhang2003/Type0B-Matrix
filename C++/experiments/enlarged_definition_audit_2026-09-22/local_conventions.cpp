// Directed check of the conventions entering machine-notes Eq. (2.10).
#include "ramond/anchors.hpp"
#include "ramond/direct_pbw.hpp"
#include <iostream>
using namespace ramond;
int main() {
    MP::precision(40);
    MP b=parse<MP>("7/5"),q=b+MP(1)/b,central=rational<MP>(3,2)+MP(3)*q*q;
    MP I(Machine(0,1)),t=(MP(-1)+I)/root(MP(2));
    std::array<MP,3> p{parse<MP>("11/23"),parse<MP>("13/29"),parse<MP>("17/31")};
    ScaWords<MP> words;
    std::array<std::unique_ptr<FreeField<MP>>,3> free;
    std::array<std::unique_ptr<PBWModule<MP>>,3> pbw;
    std::array<std::unique_ptr<ScaModule<MP>>,3> human,native;
    std::array<std::vector<PBW>,3> states;
    for(int j=0;j<3;j++) {
        free[j]=std::make_unique<FreeField<MP>>(j!=0,b,p[j]);
        pbw[j]=std::make_unique<PBWModule<MP>>(*free[j]);
        MP h=q*q/MP(8)-p[j]*p[j]/MP(2)+(j?rational<MP>(1,16):MP(0));
        human[j]=std::make_unique<ScaModule<MP>>(words,h,central,p[j]/root(MP(2)),j!=0,false);
        native[j]=std::make_unique<ScaModule<MP>>(words,h,central,p[j]/root(MP(2)),j!=0,true);
        for(int ground=0;ground<(j?2:1);ground++) {
            PBW s{};s.ground=ground;states[j].push_back(s);
            s.g={1};states[j].push_back(s);
            s.g.clear();s.l={1};states[j].push_back(s);
        }
    }
    auto id=[&](int slot,const PBW&s) {
        ScaWord w;for(int n:s.l)w.push_back({0,-2*n});
        for(int n:s.g)w.push_back({1,-(slot?2:1)*n});
        return 2*words.intern(w)+s.ground;
    };
    double max_vertex=0,max_gram=0;int vertices=0,grams=0,failed=0;
    auto check=[&](MP a,MP z,double&maximum) {
        double e=magnitude(a-z)/std::max({1.,magnitude(a),magnitude(z)});
        maximum=std::max(maximum,e);if(e>1e-28)failed++;
    };
    for(int primary=0;primary<2;primary++)for(int f=0;f<2;f++)for(int eta:{-1,1}) {
        PhysicalForm<MP> production({pbw[0].get(),pbw[1].get(),pbw[2].get()},f,sign(f)*eta,primary);
        ScaWard<MP> rho(words,{human[0].get(),human[1].get(),human[2].get()},primary,f,eta);
        for(auto&a:states[0])for(auto&bb:states[1])for(auto&cc:states[2]) {
            int A=a.g.size()%2,B=bb.g.size()%2;
            MP expected=power(t,bb.ground+cc.ground)*power(-I,A)
                *MP(sign(f*(A+B+bb.ground)))
                *rho.value({id(0,a),id(1,bb),id(2,cc)});
            check(production.value({a,bb,cc}),expected,max_vertex);vertices++;
        }
    }
    for(int j=0;j<3;j++)for(auto&a:states[j])for(auto&bb:states[j]) {
        int ai=id(j,a),bi=id(j,bb);
        if(words.words[ai/2].level2!=words.words[bi/2].level2)continue;
        MP human_gram=power(I,a.ground)*human[j]->inner(ai,bi);
        check(native[j]->inner(ai,bi),power(t,a.ground+bb.ground)*human_gram,max_gram);grams++;
    }
    std::cout<<std::setprecision(17)<<"{\"dps\":40,\"local_vertex_cases\":"<<vertices
        <<",\"max_vertex_scaled_error\":"<<max_vertex<<",\"Gram_cases\":"<<grams
        <<",\"max_Gram_scaled_error\":"<<max_gram<<",\"failed\":"<<failed<<"}\n";
    return failed?1:0;
}
