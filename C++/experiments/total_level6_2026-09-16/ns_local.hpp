#pragma once
#include "ramond/direct_pbw.hpp"
#include "ramond/anchors.hpp"
namespace level6 {
using namespace ramond;
// The existing human-convention NS Ward recurrence, with scalar type preserved.
template<class S> class NSWard {
    ScaWords<S>&w; std::array<ScaModule<S>*,3> m;
    std::map<std::array<int,4>,S> cache;
    int lev(int x)const{return w.words[x/2].level2;}
    int par(int x)const{return w.words[x/2].parity;}
    int tail(int x)const{return 2*w.words[x/2].tail;}
    int phase(std::array<int,3>s)const{return sign((par(s[0])+par(s[1])+par(s[2]))*par(s[2]));}
    S action(std::array<int,3> target,std::array<int,3>s,int slot,int kind,int mode,int rev){
        S ans=0;int initial=phase(target);
        for(auto [t,c]:m[rev?2-slot:slot]->act(kind,mode,s[slot])){
            auto changed=s;changed[slot]=t;
            ans+=S(initial*phase(changed))*c*value(changed,rev);
        }return ans;
    }
public:
    NSWard(ScaWords<S>&words,std::array<ScaModule<S>*,3> modules):w(words),m(modules){}
    S value(std::array<int,3>s,int rev=0){
        std::array<int,4>key{s[0],s[1],s[2],rev};auto it=cache.find(key);if(it!=cache.end())return it->second;
        std::array<S,3>h{m[rev?2:0]->h,m[1]->h,m[rev?0:2]->h};
        auto rest=s;S ans=0;bool boundary=true;
        for(int x:s)if(x && (w.words[x/2].word.size()!=1 || !(w.words[x/2].word[0]==ScaMode{1,-1})))boundary=false;
        if(boundary){
            int bits=par(s[0])*4+par(s[1])*2+par(s[2]);
            if(bits==3)ans=h[0]-h[1]-h[2];
            else if(bits==5)ans=h[0]-h[1]+h[2];
            else if(bits==6)ans=h[0]+h[1]-h[2];
            else if(bits==7)ans=h[0]+h[1]+h[2]-rational<S>(1,2);
            else ans=S(1);
            ans*=S(phase(s));
        }else if(s[1]){
            auto mode=w.words[s[1]/2].word.front();rest[1]=tail(s[1]);
            if(mode.kind){
                int k=-mode.twice,upper=std::max(0,(lev(s[0])-k)/2);
                for(int j=0;j<=upper;j++)ans+=w.binomial_half(k-3+2*j,j)*action(s,rest,0,1,k+2*j,rev);
                int sg=-sign(par(s[0])+par(s[2])+(k+1)/2);
                for(int j=0;j<=lev(s[2])/2+1;j++)ans+=S(sg)*w.binomial_half(k-3+2*j,j)*action(s,rest,2,1,2*j-1,rev);
            }else{
                int n=-mode.twice/2;
                if(n==1)ans=(h[0]-h[1]-h[2]+rational<S>(lev(s[0])-lev(rest[1])-lev(s[2]),2))*value(rest,rev);
                else{
                    for(int j=0;j<=std::max(0,lev(s[0])/2-n);j++)ans+=S(binomial(n-2+j,n-2))*action(s,rest,0,0,2*(n+j),rev);
                    for(int j=0;j<=lev(s[2])/2+1;j++)ans+=S(sign(n)*binomial(n-2+j,n-2))*action(s,rest,2,0,2*(j-1),rev);
                }
            }
        }else if(s[0]){
            auto word=w.words[s[0]/2].word;auto mode=word.front();rest[0]=tail(s[0]);
            if(mode==ScaMode{1,-1} && word.size()>1){
                auto next=word[1];ScaWord rem(word.begin()+2,word.end()),reordered{next,mode};reordered.insert(reordered.end(),rem.begin(),rem.end());
                auto changed=s;changed[0]=2*w.intern(reordered);ans=S(next.kind?-1:1)*value(changed,rev);
                rem.insert(rem.begin(),{next.kind?0:1,next.twice-1});changed[0]=2*w.intern(rem);
                ans+=(next.kind?S(2):rational<S>(-next.twice-2,4))*value(changed,rev);
            }else if(mode.kind){
                int k=-mode.twice;
                if(k==1){auto reflected=s;std::swap(reflected[0],reflected[2]);ans=S(phase(s)*phase(reflected))*value(reflected,1-rev);}
                else{
                    ans=S(sign(par(rest[0])+par(s[2])+1))*action(s,rest,2,1,k,rev);
                    for(int j=-1;j<=(k-1)/2;j++)ans+=S(binomial((k+1)/2,j+1))*action(s,rest,1,1,2*j+1,rev);
                }
            }else{
                int n=-mode.twice/2;ans=action(s,rest,2,0,2*n,rev);
                for(int j=-1;j<=n;j++)ans+=S(binomial(n+1,j+1))*action(s,rest,1,0,2*j,rev);
            }
        }else if(s[2]){auto reflected=s;std::swap(reflected[0],reflected[2]);ans=S(phase(s)*phase(reflected))*value(reflected,1-rev);}
        else ans=S(1);
        return cache.emplace(key,ans).first->second;
    }
};
inline long ns_pfaffian(const std::vector<std::pair<int,int>>&f){
    if(f.empty())return 1;if(f.size()%2)return 0;long ans=0;
    auto [a,i]=f[0];
    for(int k=1;k<int(f.size());k++){
        auto [b,j]=f[k];long v=0;
        if(a==0&&b==1)v=binomial(i-1,j-1);
        if(a==0&&b==2)v=i==j;
        if(a==1&&b==2)v=sign(i-1)*binomial(i+j-2,i-1);
        if(v){std::vector<std::pair<int,int>>rest;for(int l=1;l<int(f.size());l++)if(l!=k)rest.push_back(f[l]);ans+=sign(k+1)*v*ns_pfaffian(rest);}
    }return ans;
}
inline long ns_fermion(const std::array<AuxState,3>&states){
    std::vector<std::pair<int,int>>f;
    for(auto it=states[0].modes.rbegin();it!=states[0].modes.rend();++it)f.emplace_back(0,(*it+1)/2);
    for(int slot=1;slot<3;slot++)for(int n:states[slot].modes)f.emplace_back(slot,(n+1)/2);
    return sign(states[0].modes.size())*ns_pfaffian(f);
}
template<class S> class NSBranches {
    struct Term{AuxState aux;int physical;S coefficient;};
    ScaWords<S>&words;S b;std::array<S,3> p;
    std::array<std::array<std::unique_ptr<FreeField<S>>,2>,3> free;
    std::array<std::array<std::unique_ptr<PBWModule<S>>,2>,3> modules;
    std::map<std::pair<int,int>,std::vector<Term>> states;
    std::map<std::array<int,3>,S> cache;
    NSWard<S> &ward;
    const std::vector<Term>&branch(int slot,int n){
        auto key=std::make_pair(slot,n);auto old=states.find(key);if(old!=states.end())return old->second;
        int r=n<0;if(!free[slot][r]){free[slot][r]=std::make_unique<FreeField<S>>(false,b,r?-p[slot]:p[slot]);modules[slot][r]=std::make_unique<PBWModule<S>>(*free[slot][r]);}
        auto &ff=*free[slot][r];auto expr=ff.primary(std::abs(n));std::map<AuxState,Sparse<S>> groups;
        for(auto [id,c]:expr){auto state=ff.states[id];AuxState aux;for(int k=63;k>=0;k--)if((state.auxiliary>>k)&1)aux.modes.push_back(2*k+1);state.auxiliary=0;groups[aux][ff.intern(state)]+=c;}
        std::vector<Term>out;
        for(auto &[aux,vec]:groups)for(auto &[pbw,c]:modules[slot][r]->from_fock(vec)){
            ScaWord w;for(int n:pbw.l)w.push_back({0,-2*n});for(int n:pbw.g)w.push_back({1,-n});out.push_back({aux,2*words.intern(w),c});
        }
        return states.emplace(key,std::move(out)).first->second;
    }
public:
    NSBranches(ScaWords<S>&w,S coupling,std::array<S,3>momenta,NSWard<S>&form):words(w),b(coupling),p(momenta),ward(form){}
    S raw(std::array<int,3> n){auto old=cache.find(n);if(old!=cache.end())return old->second;S ans=0;
        for(auto&a:branch(0,n[0]))for(auto&b:branch(1,n[1]))for(auto&c:branch(2,n[2])){
            long ff=ns_fermion({a.aux,b.aux,c.aux});if(!ff)continue;
            int phase=sign(words.words[b.physical/2].parity*c.aux.modes.size());
            ans+=S(phase*ff)*a.coefficient*b.coefficient*c.coefficient*ward.value({a.physical,b.physical,c.physical});
        }return cache.emplace(n,ans).first->second;
    }
};
}
