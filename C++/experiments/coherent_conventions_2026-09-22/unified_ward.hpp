#pragma once
// Independent NS-R-R contour solver. No call to either existing Ward evaluator
// and no parity-dependent conversion of its answers.
#include "ramond/direct_pbw.hpp"
#include "ramond/anchors.hpp"

namespace coherent {
using namespace ramond;
template<class S> class Physical {
    ScaWords<S>& w;
    std::array<ScaModule<S>*,3> mod;
    int f,eta;
    using Triple=std::array<int,3>;
    std::map<Triple,S> cache;
    std::set<Triple> active;
    int tail(int x)const{return 2*w.words[x/2].tail+x%2;}
    int lev(int x)const{return w.words[x/2].level2;}
    void add(std::map<Triple,S>&eq,Triple s,int slot,int kind,int mode2,S c){
        if(c==S(0))return;
        for(auto [next,v]:mod[slot]->act(kind,mode2,s[slot])){
            auto t=s;t[slot]=next;eq[t]+=c*v;
        }
    }
    S action(Triple s,int slot,int kind,int mode2){
        S ans=0;
        for(auto [next,v]:mod[slot]->act(kind,mode2,s[slot])){
            auto t=s;t[slot]=next;ans+=v*value(t);
        }return ans;
    }
public:
    Physical(ScaWords<S>&words,std::array<ScaModule<S>*,3> modules,int ff,int ee)
        :w(words),mod(modules),f(ff),eta(ee){}
    int parity(int x)const{return (w.words[x/2].parity+x%2)%2;}
    // Residues of z^m(z-1)^n sqrt(z(z-1)) times the G correlator.
    // sqrt(z(z-1)) ~ z at infinity, +sqrt(z-1) at 1, +i sqrt(z) at 0.
    std::map<Triple,S> equation(Triple s,int m,int n){
        std::map<Triple,S> eq;
        int bound=std::max({lev(s[0]),lev(s[1]),lev(s[2])})/2+std::abs(m)+std::abs(n)+5;
        S I(Machine(0,1));
        for(int j=0;j<=bound;j++){
            S a=from_rational<S>(half_binomial(2*n+1,j));
            add(eq,s,0,1,2*j-2*m-2*n-1,-S(sign(j))*a);
            add(eq,s,1,1,2*(n+j),from_rational<S>(half_binomial(2*m+1,j)));
            add(eq,s,2,1,2*(m+j),I*S(sign(parity(s[1])+n+j))*a);
        }return eq;
    }
    S value(Triple s){
        if((parity(s[0])+parity(s[1])+parity(s[2]))%2!=f)return S(0);
        auto old=cache.find(s);if(old!=cache.end())return old->second;
        require(!active.count(s),"cycle in independent physical contour solver");active.insert(s);
        const auto first=[&](int slot){return w.words[s[slot]/2].word.front();};
        S ans=0;auto rest=s;int gslot=-1;
        if(s[0]/2 && first(0).kind)gslot=0;
        else if(s[1]/2){
            auto a=first(1);rest[1]=tail(s[1]);int n=-a.twice/2;
            if(a.kind)gslot=1;
            else if(n==1)ans=(mod[0]->h-mod[1]->h-mod[2]->h+rational<S>(lev(s[0])-lev(rest[1])-lev(s[2]),2))*value(rest);
            else for(int j=0;j<=std::max(lev(s[0]),lev(s[2]))/2+3;j++)
                ans+=S(binomial(n-2+j,n-2))*(action(rest,0,0,2*(n+j))+S(sign(n))*action(rest,2,0,2*(j-1)));
        }else if(s[0]/2){
            int n=-first(0).twice/2;rest[0]=tail(s[0]);
            ans=action(rest,2,0,2*n);
            for(int j=-1;j<=n;j++)ans+=S(binomial(n+1,j+1))*action(rest,1,0,2*j);
        }else if(s[2]/2){
            auto a=first(2);rest[2]=tail(s[2]);int n=-a.twice/2;
            if(a.kind)gslot=2;
            else ans=(mod[2]->h+S(n)*mod[1]->h-mod[0]->h+rational<S>(lev(rest[2]),2))*value(rest);
        }else{
            if(!f)ans=s[1]%2?S(eta):S(1);
            else ans=s[1]%2?S(Machine(0,-eta)):S(1);
        }
        if(gslot>=0){
            int mode=-first(gslot).twice;rest=s;rest[gslot]=tail(s[gslot]);
            int m=gslot==0?(mode-1)/2:gslot==2?-mode/2:0;
            int n=gslot==1?-mode/2:0;
            auto eq=equation(rest,m,n);S c=eq[s];eq.erase(s);
            require(c!=S(0),"physical contour missed target");ans=0;
            for(auto [t,v]:eq)if(v!=S(0))ans-=v*value(t);
            ans/=c;
        }
        active.erase(s);cache[s]=ans;return ans;
    }
    size_t size()const{return cache.size();}
};

template<class S> class Auxiliary {
    std::map<AuxTriple,S> cache;
    std::set<AuxTriple> active;
    static std::pair<AuxState,S> act(int slot,int mode,AuxState s){
        if(!mode){require(slot!=0,"NS zero mode");s.ground^=1;return {s,S(sign(s.modes.size()))/root(S(2))};}
        auto [t,c]=aux_act(slot,mode,s);return {t,from_rational<S>(c)};
    }
    static void add(std::map<AuxTriple,S>&eq,AuxTriple s,int slot,int mode,S c){
        if(c==S(0))return;auto [next,v]=act(slot,mode,s[slot]);s[slot]=next;eq[s]+=c*v;
    }
public:
    // Residues of z^m(z-1)^n / sqrt(z(z-1)) times the psi correlator.
    // The plus sign of the infinity term is from psi_r^dagger=-psi_-r.
    std::map<AuxTriple,S> equation(AuxTriple s,int m,int n){
        std::map<AuxTriple,S> eq;int bound=std::abs(m)+std::abs(n)+5;
        for(auto &x:s)if(!x.modes.empty())bound+=x.modes.front();
        S I(Machine(0,1));
        for(int j=0;j<=bound;j++){
            S a=from_rational<S>(half_binomial(2*n-1,j));
            add(eq,s,0,2*j-2*m-2*n+1,S(sign(j))*a);
            add(eq,s,1,n+j,from_rational<S>(half_binomial(2*m-1,j)));
            add(eq,s,2,m+j,-I*S(sign(aux_parity(1,s[1])+n+j))*a);
        }return eq;
    }
    S value(AuxTriple s){
        if((aux_parity(0,s[0])+aux_parity(1,s[1])+aux_parity(2,s[2]))%2)return S(0);
        auto old=cache.find(s);if(old!=cache.end())return old->second;
        int slot=-1;for(int j=0;j<3;j++)if(!s[j].modes.empty()){slot=j;break;}
        if(slot<0)return s[1].ground?S(Machine(0,1)):S(1);
        require(!active.count(s),"cycle in independent auxiliary contour solver");active.insert(s);
        auto rest=s;int k=rest[slot].modes.front();rest[slot].modes.erase(rest[slot].modes.begin());
        int m=slot==0?(k+1)/2:slot==2?-k:0,n=slot==1?-k:0;
        auto eq=equation(rest,m,n);S c=eq[s];eq.erase(s);require(c!=S(0),"auxiliary contour missed target");
        S ans=0;for(auto [t,v]:eq)if(v!=S(0))ans-=v*value(t);ans/=c;
        active.erase(s);cache[s]=ans;return ans;
    }
    size_t size()const{return cache.size();}
};
}
