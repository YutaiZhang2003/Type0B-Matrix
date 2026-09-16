#pragma once
#include "ns_schottky.hpp"
// All-NS fixed-weight c-recursion in the human-note trinion convention.
class NSRecursion {
    struct Pole {S central,prefactor,x;int id;};
    Graph graph;std::vector<S> weights,centers;
    std::map<std::array<int,4>,Pole>poles;
    std::map<std::array<int,8>,S>fusions;
    std::map<std::array<int,7>,S>vertices;
    std::map<std::array<int,3>,S>norms;
    std::map<std::array<int,17>,S>cache;
    QPoly vacuum;
    S h(int e,int shift)const{return weights[e]+S(shift)/S(2);}
    int mask(const Key&k)const {int m=0;for(int e=0;e<graph.edges;e++)m|=(k[e]%2)<<e;return m;}
    const Pole&pole(int e,int shift,int r,int s){
        std::array<int,4>key{e,shift,r,s};auto old=poles.find(key);if(old!=poles.end())return old->second;
        S weight=h(e,shift),affine=S(4)*weight+S(r*s-1),rad=root(S(16)*weight*weight+S(8*(r*s-1))*weight+S((r-s)*(r-s)));
        if(magnitude(affine-rad)>magnitude(affine+rad))rad=-rad;
        S x=(affine+rad)/S(1-r*r),b=root(x),c=rational<S>(15,2)+S(3)*(x+S(1)/x);
        S jac=-S(3)*(S(1)-S(1)/(x*x))*(S(4)+(S(16)*weight+S(4*(r*s-1)))/rad)/S(1-r*r);
        S A=rational<S>(1,2);for(int a=1-r;a<=r;a++)for(int t=1-s;t<=s;t++)if((a+t)%2==0&&(a||t)&&!(a==r&&t==s))A*=root(S(2))/(S(a)*b+S(t)/b);
        int id=centers.size();centers.push_back(c);return poles.emplace(key,Pole{c,jac*A,x,id}).first->second;
    }
    S fusion(const Pole&p,int r,int s,int a,int da,int b,int db,int f){
        std::array<int,8>key{p.id,r,s,a,da,b,db,f};auto old=fusions.find(key);if(old!=fusions.end())return old->second;
        // Pair opposite momentum shifts before multiplication: no arbitrary square roots.
        S x=p.x,lam=x+S(2)+S(1)/x-S(8)*h(a,da),diff=S(8)*(h(b,db)-h(a,da)),value=1;
        for(int u=1-r;u<r;u+=2)for(int v=1-s;v<s;v+=2){int congr=((u+v-r-s)%4+4)%4;if(congr!=(f?0:2))continue;
            if(!u&&!v)value*=h(b,db)-h(a,da);
            else if(u>0||(!u&&v>0)){S d=S(u*u)*x+S(2*u*v)+S(v*v)/x;value*=((diff+d)*(diff+d)-S(4)*lam*d)/S(64);}
        }return fusions.emplace(key,value).first->second;
    }
    S vertex(int v,const Key&shift,const Key&level){
        auto es=graph.slots[v];std::array<int,7>key{v,shift[es[0]],shift[es[1]],shift[es[2]],level[es[0]],level[es[1]],level[es[2]]};
        auto old=vertices.find(key);if(old!=vertices.end())return old->second;
        S h0=h(es[0],shift[es[0]]),h1=h(es[1],shift[es[1]]),h2=h(es[2],shift[es[2]]);
        int a=level[es[0]]%2,b=level[es[1]]%2,c=level[es[2]]%2,bits=4*a+2*b+c;
        S bottom=1;switch(bits){case 6:bottom=h0+h1-h2;break;case 5:bottom=h0-h1+h2;break;case 3:bottom=h0-h1-h2;break;case 1:bottom=-S(1);break;case 7:bottom=-(h0+h1+h2-rational<S>(1,2));break;}
        h0+=S(a)/S(2);h1+=S(b)/S(2);h2+=S(c)/S(2);
        int i=level[es[0]]/2,j=level[es[1]]/2,k=level[es[2]]/2;S sum=0;
        for(int p=0;p<=std::min(i,k);p++){S term=S(binomial(i,p));for(int t=0;t<p;t++)term*=S(k-t)*(S(2)*h2+S(k-1-t));term*=rising(h1+h2-h0,k-p)*rising(h0+h1-h2+S(p-k),i-p);sum+=term;}
        sum*=bottom*rising(h0-h1-h2+S(i-j+1-k),j);return vertices.emplace(key,sum).first->second;
    }
    S global(const Key&shift,const Key&level){S ans=S(graph.sign_of(mask(level)));for(int v=0;v<int(graph.slots.size());v++)ans*=vertex(v,shift,level);
        for(int e=0;e<graph.edges;e++){std::array<int,3>key{e,shift[e],level[e]};auto old=norms.find(key);if(old==norms.end()){int n=level[e]/2;S norm=rising(S(2)*h(e,shift[e]),n+level[e]%2);for(int j=2;j<=n;j++)norm*=S(j);old=norms.emplace(key,norm).first;}ans/=old->second;}return ans;}
    // The local crossing is global-middle parity times vacuum-ket parity.
    // K polarization alone drops this sign. It cancels between the two
    // theta vertices, but does not cancel on Mercedes.
    S regular(const Key&shift,const Key&level){S ans=0;for(auto&[v,co]:vacuum)if(leq(v,level)){auto n=minus(level,v);int vm=mask(v),nm=mask(n);ans+=from_rational<S>(co)*S(graph.kernel(nm,vm))*global(shift,n);}return ans;}
    S recurse(int incoming,const Key&shift,const Key&level){std::array<int,17>key{};key[0]=incoming;for(int e=0;e<8;e++){key[1+e]=shift[e];key[9+e]=level[e];}auto old=cache.find(key);if(old!=cache.end())return old->second;
        S ans=regular(shift,level);int originalmask=mask(level);
        for(int e=0;e<graph.edges;e++)for(int r=2;r<=level[e];r++)for(int s=1;r*s<=level[e];s++)if((r+s)%2==0){auto p=pole(e,shift[e],r,s);S residue=p.prefactor;int exponent=0;
            for(int half:graph.ends[e]){auto es=graph.slots[half/3];int slot=half%3;int a=es[slot==2?0:2],b=es[slot==0?1:(slot==1?0:1)];int f=0;for(int ee:es)f^=level[ee]%2;residue*=fusion(p,r,s,a,shift[a],b,shift[b],f);
                // An odd null changes the shifted primary's intrinsic parity.
                // Reset it to the even-primary convention used by global():
                // rho(p1,p2,p3)=(-1)^(p1*A+p2*C) rho(0,0,0).
                // Combined with the human-note null-factorization formulas,
                // the three incidence signs are 1, (-1)^C, and -1.
                if(r*s%2){if(slot==1)exponent+=level[es[2]]%2;if(slot==2)exponent++;}
            }
            if(residue==S(0))continue;auto childlevel=level,childshift=shift;childlevel[e]-=r*s;childshift[e]+=r*s;
            S denominator=centers[incoming]-p.central;require(magnitude(denominator)>1e-25,"coincident NS poles require a confluent limit");
            residue*=S(sign(exponent)*graph.sign_of(originalmask)*graph.sign_of(mask(childlevel)));
            ans+=residue/denominator*recurse(p.id,childshift,childlevel);
        }return cache.emplace(key,ans).first->second;
    }
public:
    NSRecursion(Graph g,S c,std::vector<S>h,int level):graph(g),weights(h),centers{c}{vacuum=ns_mercedes_schottky(g,level);}
    // shift+remaining level equals the requested multidegree at every node.
    // Therefore these full-block states cannot be reused by a different target.
    S coefficient(const Key&k){cache.clear();return recurse(0,Key{},k);}
};
