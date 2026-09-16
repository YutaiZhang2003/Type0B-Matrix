#pragma once
#include "ramond/ccy.hpp"
#include "ramond/fermion.hpp"
#include <functional>
namespace level6 {
using namespace ramond;
using Key=std::array<int,8>;
template<class S>using Poly=std::map<Key,S>;
inline int total(const Key&k){int n=0;for(int v:k)n+=v;return n;}
inline Key plus(Key a,const Key&b){for(int e=0;e<8;e++)a[e]+=b[e];return a;}
inline Key minus(Key a,const Key&b){for(int e=0;e<8;e++)a[e]-=b[e];return a;}
inline bool leq(const Key&a,const Key&b){for(int e=0;e<8;e++)if(a[e]>b[e])return false;return true;}
inline void degree_sort(std::vector<Key>&v){std::sort(v.begin(),v.end(),[](auto&a,auto&b){return total(a)==total(b)?a<b:total(a)<total(b);});}
struct Graph {
    int edges;std::vector<std::array<int,3>> slots;
    std::vector<std::array<int,2>> ends;
    Graph(int e,std::vector<std::array<int,3>> s,std::vector<std::array<int,2>> pair):edges(e),slots(s),ends(pair){}
    Graph split(const std::vector<int>&marked)const{
        Graph g=*this;
        for(int e:marked){int right=g.edges++,v=g.slots.size(),leftend=g.ends[e][0],rightend=g.ends[e][1];
            g.slots[rightend/3][rightend%3]=right;
            g.ends[e]={leftend,3*v+2};g.ends.push_back({3*v,rightend});
            g.slots.push_back({right,-1,e});
        }return g;
    }
    int sign_of(int mask)const{
        std::vector<int>order;for(auto p:ends){order.push_back(p[0]);order.push_back(p[1]);}
        int exponent=0;for(int i=0;i<int(order.size());i++)for(int j=i+1;j<int(order.size());j++)if(order[i]>order[j]){
            int a=slots[order[i]/3][order[i]%3],b=slots[order[j]/3][order[j]%3];exponent+=((mask>>a)&1)*((mask>>b)&1);
        }return sign(exponent);
    }
    int kernel(int a,int b)const{int k=0;for(auto v:slots)k+=((a>>v[1])&1)*((b>>v[2])&1);return sign_of(a^b)*sign_of(a)*sign_of(b)*sign(k);}
};
template<class S>class GraphCCY {
    struct Pole{S central,residue,x;int id;};
    Graph graph;std::vector<S>h_;std::vector<S>centers_;
    std::map<std::array<int,4>,Pole>geometry_;
    std::map<std::array<int,7>,S>fusion_;
    std::map<std::array<int,2>,S>denominators;
    std::map<std::array<int,3>,S>norms;
    std::map<std::array<int,7>,S>vertices;
    S weight(int e,int shift)const{return h_[e]+S(shift);}
    #include "pole_fragment.inc"
    S denominator(int incoming,int outgoing){auto k=std::array<int,2>{incoming,outgoing};auto it=denominators.find(k);if(it!=denominators.end())return it->second;
        S d=centers_[incoming]-centers_[outgoing];require(!small(d/S(std::max({1.,magnitude(centers_[incoming]),magnitude(centers_[outgoing])})),64*0x1p-52,10-digits<S>()),"unresolved generic graph CCY poles");return denominators.emplace(k,S(1)/d).first->second;}
    S norm(int e,int shift,int n){auto k=std::array<int,3>{e,shift,n};auto it=norms.find(k);if(it!=norms.end())return it->second;S ans=rising(S(2)*weight(e,shift),n);for(int i=2;i<=n;i++)ans*=S(i);return norms.emplace(k,ans).first->second;}
    S vertex(int v,const Key&shift,const Key&level){
        auto e=graph.slots[v];std::array<int,3>s{},n{};for(int j=0;j<3;j++)if(e[j]>=0){s[j]=shift[e[j]];n[j]=level[e[j]];}else e[j]=graph.edges;
        std::array<int,7>key{v,s[0],s[1],s[2],n[0],n[1],n[2]};auto it=vertices.find(key);if(it!=vertices.end())return it->second;
        S h1=weight(e[0],s[0]),h2=weight(e[1],s[1]),h3=weight(e[2],s[2]);int i=n[0],j=n[1],k=n[2];S ans=0;
        for(int p=0;p<=std::min(i,k);p++){
            S x=S(binomial(i,p));for(int u=0;u<p;u++)x*=S(k-u)*(S(2)*h3+S(k-1-u));
            x*=rising(h2+h3-h1,k-p)*rising(h1+h2-h3+S(p-k),i-p);ans+=x;
        }
        ans*=rising(h1-h2-h3+S(i-j+1-k),j);
        return vertices.emplace(key,ans).first->second;
    }
    S global(const Key&s,const Key&n){S ans=1;for(int v=0;v<int(graph.slots.size());v++)ans*=vertex(v,s,n);for(int e=0;e<graph.edges;e++)ans/=norm(e,s[e],n[e]);return ans;}
    S residue(const Pole&p,const Key&shift,int edge,int r,int s){
        auto ends=graph.ends[edge];S ans=p.residue;
        if(ends[0]/3==ends[1]/3){
            int external=-1;for(int e:graph.slots[ends[0]/3])if(e!=edge)external=e;
            require(external>=0,"self-loop external edge missing");
            // The two sewn slots are at 1 and 0: the middle-slot Ward normalization gives (-1)^rs.
            return S(sign(r*s))*ans*fusion(p,r,s,external,shift[external],edge,shift[edge]+r*s)*fusion(p,r,s,external,shift[external],edge,shift[edge]);
        }
        for(int half:ends){auto es=graph.slots[half/3];int slot=half%3;
            int a=es[slot==2?0:2],b=es[slot==0?1:(slot==1?0:1)];
            int as=a<0?0:shift[a],bs=b<0?0:shift[b];if(a<0)a=graph.edges;if(b<0)b=graph.edges;
            ans*=fusion(p,r,s,a,as,b,bs);
        }return ans;
    }
public:
    size_t transitions=0,seed_terms=0;
    GraphCCY(Graph g,S central,std::vector<S>weights,S external=S(0)):graph(g),h_(weights),centers_{central}{h_.push_back(external);}
    Poly<S> reduced(std::vector<Key>indices){
        degree_sort(indices);std::set<Key>allowed(indices.begin(),indices.end());Key maximum{};
        for(auto n:indices)for(int e=0;e<graph.edges;e++){maximum[e]=std::max(maximum[e],n[e]);if(n[e]){auto lower=n;lower[e]--;require(allowed.count(lower),"graph CCY domain is not downward closed");}}
        std::map<Key,std::map<int,S>>amplitudes;amplitudes[Key{}][0]=S(1);Poly<S>totals,answer;
        for(auto shift:indices){auto found=amplitudes.find(shift);if(found==amplitudes.end())continue;auto incoming=std::move(found->second);amplitudes.erase(found);S sum=0;for(auto [id,v]:incoming)sum+=v;totals[shift]=sum;
            for(int e=0;e<graph.edges;e++)for(int r=2;r<=maximum[e]-shift[e];r++)for(int s=1;r*s<=maximum[e]-shift[e];s++){
                auto changed=shift;changed[e]+=r*s;if(!allowed.count(changed))continue;auto p=pole(e,shift[e],r,s);S res=residue(p,shift,e,r,s);if(res==S(0))continue;
                S v=0;for(auto [id,a]:incoming){v+=a*denominator(id,p.id);transitions++;}amplitudes[changed][p.id]+=res*v;
            }
        }
        for(auto n:indices){S v=0;for(auto [shift,a]:totals)if(leq(shift,n)){v+=a*global(shift,minus(n,shift));seed_terms++;}answer.emplace(n,v);}return answer;
    }
};
// Primitive Schottky classes of the actual ordered plumbing graph.
using QPoly=Poly<Rational>;using QMat=std::array<QPoly,4>;
inline QPoly constant(Rational q){QPoly p;if(q!=0)p[Key{}]=q;return p;}
inline QPoly add(QPoly a,const QPoly&b){for(auto[k,v]:b){a[k]+=v;if(a[k]==0)a.erase(k);}return a;}
inline QPoly scale(QPoly a,Rational b){for(auto&[k,v]:a)v*=b;return a;}
inline QPoly multiply(const QPoly&a,const QPoly&b,int cutoff){QPoly p;for(auto[ka,va]:a)for(auto[kb,vb]:b){auto k=plus(ka,kb);if(total(k)<=cutoff)p[k]+=va*vb;}for(auto it=p.begin();it!=p.end();)if(it->second==0)it=p.erase(it);else ++it;return p;}
inline int order(const QPoly&a,int cutoff){int n=cutoff+1;for(auto[k,v]:a)if(v!=0)n=std::min(n,total(k));return n;}
inline QPoly inverse(const QPoly&a,int cutoff){auto it=a.find(Key{});require(it!=a.end()&&it->second!=0,"nonunit Schottky trace");auto rem=add(constant(1),scale(a,-1/it->second));auto out=constant(1),p=out;for(int j=1;j<=cutoff/order(rem,cutoff);j++){p=multiply(p,rem,cutoff);out=add(out,p);}return scale(out,1/it->second);}
inline QMat multiply(const QMat&a,const QMat&b,int l){return {add(multiply(a[0],b[0],l),multiply(a[1],b[2],l)),add(multiply(a[0],b[1],l),multiply(a[1],b[3],l)),add(multiply(a[2],b[0],l),multiply(a[3],b[2],l)),add(multiply(a[2],b[1],l),multiply(a[3],b[3],l))};}
inline QPoly schottky(const Graph&g,int cutoff){
    auto one=constant(1);std::array<QMat,3>coordinate{QMat{QPoly{},one,one,QPoly{}},QMat{one,constant(-1),{},one},QMat{one,{}, {},one}};
    std::array<QMat,3>back=coordinate;back[1]={one,one,{},one};std::vector<QMat>maps;
    for(int e=0;e<g.edges;e++)for(int d=0;d<2;d++){
        Key k{};k[e]=1;QPoly q{{k,Rational(1)}};QMat inv{QPoly{},q,one,QPoly{}};
        maps.push_back(multiply(back[g.ends[e][1-d]%3],multiply(inv,coordinate[g.ends[e][d]%3],cutoff),cutoff));
    }
    std::set<Part>classes;Part word;
    std::function<void(int,int)>visit=[&](int start,int vertex){
        int n=word.size();if(n&&vertex==start&&word.back()!=(word.front()^1)){
            bool primitive=true;for(int p=1;p<n;p++)if(n%p==0){bool same=true;for(int j=0;j<n;j++)if(word[j]!=word[j%p])same=false;if(same)primitive=false;}
            if(primitive){Part inv;for(auto it=word.rbegin();it!=word.rend();++it)inv.push_back(*it^1);Part best=word;for(auto w:{word,inv})for(int r=0;r<n;r++){Part rot(w.begin()+r,w.end());rot.insert(rot.end(),w.begin(),w.begin()+r);best=std::min(best,rot);}classes.insert(best);}
        }
        if(n==cutoff/2)return;for(int e=0;e<g.edges;e++)for(int d=0;d<2;d++)if(g.ends[e][d]/3==vertex){int arc=2*e+d;if(n&&arc==(word.back()^1))continue;word.push_back(arc);visit(start,g.ends[e][1-d]/3);word.pop_back();}
    };
    for(int v=0;v<int(g.slots.size());v++)visit(v,v);
    QPoly logarithm;
    for(auto w:classes){QMat m{one,{}, {},one};QPoly det=one;for(int arc:w){m=multiply(maps[arc],m,cutoff);Key k{};k[arc/2]=1;int from=g.ends[arc/2][arc%2]%3,to=g.ends[arc/2][1-arc%2]%3;int ds=sign(1+(from==0)+(to==0));det=multiply(det,QPoly{{k,Rational(ds)}},cutoff);}
        auto inv=inverse(add(m[0],m[3]),cutoff),ratio=multiply(det,multiply(inv,inv,cutoff),cutoff);int o=order(ratio,cutoff);QPoly multiplier,p=one;
        for(int j=1;j<=cutoff/o;j++){p=multiply(p,ratio,cutoff);mpz_class c;mpz_bin_uiui(c.get_mpz_t(),2*j,j);c/=j+1;multiplier=add(multiplier,scale(p,Rational(c)));}
        p=multiplier;for(int s=2;s<=cutoff/o;s++){p=multiply(p,multiplier,cutoff);int d=0;for(int a=2;a<=s;a++)if(s%a==0)d+=a;logarithm=add(logarithm,scale(p,Rational(d,s)));}
    }
    auto out=one,p=one;for(int j=1;j<=cutoff/order(logarithm,cutoff);j++){p=scale(multiply(p,logarithm,cutoff),Rational(1,j));out=add(out,p);}return out;
}
}
