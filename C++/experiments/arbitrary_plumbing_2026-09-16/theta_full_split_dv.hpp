#pragma once
#include "ramond/pipeline.hpp"
namespace ramond {
// Full independent q_L,q_R expansion, used only at low level for validation.
// Unlike production diagonal_product, no equal-level projection is applied.
template<class S> ParitySeries<S> theta_full_split_dv(const Settings &settings) {
    require(settings.inserted,"full split DV requires inserted mode");
    int level=settings.level,cutoff=2*level;
    S b=parse<S>(settings.b),b2=b*b;
    std::array<S,3> p;
    for(int i=0;i<3;i++)p[i]=parse<S>(settings.momenta[i]);
    OuterBranching<S> outer(b,p,level,settings.f,settings.p,true,false);
    outer.prepare(settings.eta,-settings.eta);
    MiddleBranching<S> middle(b,p[1],*outer.actions[0]);
    auto pairs=middle_pairs(level,false);
    std::array<S,2> external{-(S(1)+S(2)*b2)/(S(2)*(S(1)-b2)),(b2+S(2))/(S(2)*(S(1)-b2))};
    ParitySeries<S> answer;
    for(int n1:outer.ns)for(auto [incoming,outgoing]:pairs)for(int n3:outer.r) {
        Index shift{n1*n1/4,ramond_level(incoming),ramond_level(outgoing),2*ramond_level(n3)};
        int remaining=cutoff-degree(shift);
        if(remaining<0)continue;
        std::array<std::pair<int,S>,2> factors;
        for(int alpha=0;alpha<2;alpha++) {
            int gamma=((settings.f-n1/2-alpha)%2+2)%2;
            int index=(((n1/2+settings.p)%2+2)%2)|(alpha<<1)|(gamma<<2);
            S denominator=ns_norm(n1,b,p[0])*r_norm(incoming,alpha,b,p[1])
                *r_norm(outgoing,alpha,b,p[1])*r_norm(n3,gamma,b,p[2]);
            S factor=outer.raw(n1,incoming,n3,alpha,gamma,settings.eta)
                *outer.raw(n1,outgoing,n3,alpha,gamma,-settings.eta)
                *middle.raw(outgoing,incoming,alpha)*S(sign(n1/2)*theta_sign(index))/denominator;
            factors[alpha]={index,factor};
        }
        if(factors[0].second==S(0)&&factors[1].second==S(0))continue;
        std::vector<Index> indices;
        for(int a=0;2*a<=remaining;a++)for(int l=0;2*a+l<=remaining;l++)
        for(int r=0;2*a+l+r<=remaining;r++)for(int d=0;2*a+l+r+2*d<=remaining;d++)
            indices.push_back({a,l,r,d});
        std::array<Series<S>,2> blocks;
        for(int copy=0;copy<2;copy++) {
            std::vector<S> weights{branch_weight(copy,n1,b,p[0]),branch_weight(copy,incoming,b,p[1]),
                branch_weight(copy,outgoing,b,p[1]),branch_weight(copy,n3,b,p[2])};
            CCY<S> engine(4,branch_central(copy,b),weights,external[copy]);
            blocks[copy]=engine.reduced(indices);
        }
        for(const auto &[kl,vl]:blocks[0])for(const auto &[kr,vr]:blocks[1]) {
            Index k=kl+kr;Index physical{2*k[0],k[1],k[2],2*k[3]};
            if(degree(physical)>remaining)continue;
            auto &row=answer[shift+physical];
            for(const auto &[index,factor]:factors)row[index]+=factor*vl*vr;
        }
    }
    return answer;
}
}
