#include "ns_local.hpp"
#include "graph_ccy.hpp"
#include "ramond/branching.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace level6;
using S=MP;
using Row=std::map<int,S>;
using Block=std::map<Key,Row>;
struct Case {std::vector<int>f,eta,marked;};
struct Setup {
    std::string name;Graph graph;std::vector<bool>r;std::vector<S>p;std::vector<Case>cases;
    Setup(std::string n):name(n),graph(3,{{0,1,2},{0,1,2}},{{0,3},{1,4},{2,5}}),r(3,false){
        p={parse<S>("11/23"),parse<S>("13/29"),parse<S>("17/31")};
        if(n=="mercedes"){
            graph=Graph(6,{{0,1,2},{0,3,5},{1,4,3},{2,5,4}},{{0,3},{1,6},{2,9},{4,8},{7,11},{10,5}});
            r={false,false,false,true,true,true};p.insert(p.end(),{parse<S>("19/37"),parse<S>("23/41"),parse<S>("29/43")});
            for(int fm=0;fm<8;fm++)for(int signs=0;signs<8;signs++){
                Case c;c.f={__builtin_popcount(unsigned(fm))%2,fm&1,(fm>>1)&1,(fm>>2)&1};c.eta={1,sign(signs&1),sign((signs>>1)&1),sign((signs>>2)&1)};
                if(sign(c.f[0])*c.eta[1]*c.eta[2]*c.eta[3]<0)c.marked={3};cases.push_back(c);
            }
        }else if(n=="theta_ns"){for(int f=0;f<2;f++)cases.push_back({{f,f},{1,1},{}});}
        else{
            require(n=="glasses_rr"||n=="glasses_nr"||n=="glasses_nn","unknown graph");graph=Graph(3,{{0,1,1},{0,2,2}},{{0,3},{1,2},{4,5}});
            r={false,n=="glasses_rr",n!="glasses_nn"};
            for(int f=0;f<2;f++)for(int a=0;a<(r[1]?2:1);a++)for(int b=0;b<(r[2]?2:1);b++){
                Case c{{f,f},{sign(a),sign(b)},{}};for(int i=0;i<2;i++)if(r[i+1]&&sign(f)*c.eta[i]<0)c.marked.push_back(i+1);cases.push_back(c);
            }
        }
    }
    int transport(int mask,const Case&c)const{int ex=0;for(int v=0;v<int(graph.slots.size());v++){auto e=graph.slots[v];if(r[e[1]])ex+=c.f[v]*(((mask>>e[0])&1)+((mask>>e[1])&1));}return sign(ex);}
    int loop_count()const{return name=="mercedes"?1:int(r[1])+int(r[2]);}
};
S getrow(const Row&r,int k){auto it=r.find(k);return it==r.end()?S(0):it->second;}
void addstar(Row&out,const Row&a,const Row&b,const Graph&g,S scale=S(1)){for(auto[x,v]:a)if(v!=S(0))for(auto[y,w]:b)if(w!=S(0))out[x^y]+=scale*S(g.kernel(x,y))*v*w;}
std::vector<Key> indices(const Setup&s,int level,const std::vector<int>&marked={}){
    std::vector<Key>out;Key k{};
    std::function<void(int,int)>visit=[&](int e,int budget){if(e==s.graph.edges){out.push_back(k);return;}auto it=std::find(marked.begin(),marked.end(),e);
        for(int n=0;n<=budget;n+=(s.r[e]?2:1)){
            k[e]=n;if(it==marked.end())visit(e+1,budget-n);
            else {int other=s.graph.edges+int(it-marked.begin());for(int r=0;r<=budget;r+=2){k[other]=r;visit(e+1,budget-std::max(n,r));}}
        }
    };visit(0,2*level);degree_sort(out);return out;
}
struct Matrix {int n=0,m=0;std::vector<S>x;Matrix()=default;Matrix(int a,int b):n(a),m(b),x(a*b){}S&operator()(int i,int j){return x[i*m+j];}const S&operator()(int i,int j)const{return x[i*m+j];}};
Matrix mul(const Matrix&a,const Matrix&b){require(a.m==b.n,"matrix contraction dimensions");Matrix c(a.n,b.m);S scratch;for(int i=0;i<a.n;i++)for(int k=0;k<a.m;k++)if(a(i,k)!=S(0))for(int j=0;j<b.m;j++)pbw_madd(c(i,j),a(i,k),b(k,j),scratch);return c;}
S trace(const Matrix&a){require(a.n==a.m,"trace dimension");S ans=0;for(int j=0;j<a.n;j++)ans+=a(j,j);return ans;}
std::vector<S> axis_transform(std::vector<S>x,std::array<int,3>d,const std::array<const Matrix*,3>&m){
    int volume=d[0]*d[1]*d[2];std::vector<S>y(volume);S scratch;
    for(int axis=0;axis<3;axis++){
        int stride=axis==0?d[1]*d[2]:axis==1?d[2]:1,n=d[axis];
        for(int base=0;base<volume;base+=n*stride)for(int off=0;off<stride;off++)for(int i=0;i<n;i++){
            S&v=y[base+i*stride+off];v=S(0);for(int j=0;j<n;j++)pbw_madd(v,(*m[axis])(i,j),x[base+j*stride+off],scratch);
        }x.swap(y);
    }return x;
}
struct StateData {int id,pbw=0;AuxState aux;};
struct EdgeData{std::vector<StateData>states;Matrix inverse;};
class Sewing {
    const Setup&setup;bool fermion;ScaWords<S>w;S b=parse<S>("7/5"),q=b+S(1)/b;
    std::vector<std::unique_ptr<ScaModule<S>>>gm,wm;
    std::map<std::array<int,3>,EdgeData>edge_cache;
    std::map<std::array<int,5>,Matrix>prop_cache;
    std::map<std::array<int,6>,S>vertex_cache;
    std::map<std::array<int,3>,std::unique_ptr<ScaWard<S>>>rw;
    std::vector<std::unique_ptr<NSWard<S>>>nw;
    FermionForm ff;int nextid=0;
    const EdgeData&edge(int e,int l,int p){
        auto key=std::array<int,3>{e,l,p};auto old=edge_cache.find(key);if(old!=edge_cache.end())return old->second;EdgeData out;bool r=setup.r[e];
        if(fermion){for(auto modes:partitions(r?l/2:l,true,!r)){
            int ground=r?(p+int(modes.size()))%2:0;if((int(modes.size())+ground)%2!=p)continue;out.states.push_back({nextid++,0,{modes,ground}});
        }}else for(int ll=0;2*ll<=l;ll++)for(auto ls:partitions(ll))for(auto gs:partitions(r?l/2-ll:l-2*ll,true,!r)){
            ScaWord word;for(int n:ls)word.push_back({0,-2*n});for(int n:gs)word.push_back({1,r?-2*n:-n});int id=w.intern(word),ground=r?(p+int(gs.size()))%2:0;
            if((w.words[id].parity+ground)%2==p)out.states.push_back({nextid++,2*id+ground,{}});
        }
        int n=out.states.size();out.inverse=Matrix(n,n);
        if(fermion){for(int j=0;j<n;j++)out.inverse(j,j)=S(sign(p));}
        else {for(int i=0;i<n;i++)for(int j=0;j<n;j++)out.inverse(i,j)=gm[e]->inner(out.states[i].pbw,out.states[j].pbw);if(n)out.inverse.x=pbw_inverse(std::move(out.inverse.x),n);}
        return edge_cache.emplace(key,std::move(out)).first->second;
    }
    const Matrix&prop(int e,int l,int r,int p,bool insert){
        if(!insert){require(l==r,"unsplit levels differ");return edge(e,l,p).inverse;}
        require(fermion,"physical edge never split");auto key=std::array<int,5>{e,l,r,p,1};auto old=prop_cache.find(key);if(old!=prop_cache.end())return old->second;
        auto&a=edge(e,l,p).states;auto&z=edge(e,r,p).states;Matrix out(a.size(),z.size());int mode=(l-r)/2;
        for(int i=0;i<int(a.size());i++)for(int j=0;j<int(z.size());j++){
            S v=0;auto left=a[i].aux,right=z[j].aux;
            if(!mode){if(left==right)v=q/root(S(2))*S(sign(left.ground));}
            else {auto action=aux_act(1,mode,left);if(action.second!=0&&action.first.modes==right.modes&&1-action.first.ground==right.ground)v=q*from_rational<S>(action.second)*S(sign(int(right.modes.size())+left.ground+1));}
            out(i,j)=S(sign(p))*v;
        }return prop_cache.emplace(key,std::move(out)).first->second;
    }
    S vertex(int v,const StateData&a,const StateData&b,const StateData&c,int f,int eta){
        auto key=std::array<int,6>{v,a.id,b.id,c.id,f,eta};auto old=vertex_cache.find(key);if(old!=vertex_cache.end())return old->second;auto e=setup.graph.slots[v];S ans=0;
        if(fermion)ans=setup.r[e[1]]?ff.complex_value<S>({a.aux,b.aux,c.aux}):S(ns_fermion({a.aux,b.aux,c.aux}));
        else if(!setup.r[e[1]])ans=nw[v]->value({a.pbw,b.pbw,c.pbw});
        else {
            auto k=std::array<int,3>{v,f,eta};auto it=rw.find(k);if(it==rw.end())it=rw.emplace(k,std::make_unique<ScaWard<S>>(w,std::array<ScaModule<S>*,3>{wm[e[0]].get(),wm[e[1]].get(),wm[e[2]].get()},0,f,eta)).first;
            ans=power((S(-1)+S(Machine(0,1)))/root(S(2)),b.pbw%2+c.pbw%2)*it->second->value({a.pbw,b.pbw,c.pbw});
        }return vertex_cache.emplace(key,ans).first->second;
    }
public:
    Sewing(const Setup&s,bool ff):setup(s),fermion(ff){
        S central=rational<S>(3,2)+S(3)*q*q;for(int e=0;e<s.graph.edges;e++){
            S h=q*q/S(8)-s.p[e]*s.p[e]/S(2)+(s.r[e]?rational<S>(1,16):S(0));
            gm.push_back(std::make_unique<ScaModule<S>>(w,h,central,s.p[e]/root(S(2)),s.r[e],true));
            wm.push_back(std::make_unique<ScaModule<S>>(w,h,central,s.p[e]/root(S(2)),s.r[e],false));
        }
        for(auto e:s.graph.slots)nw.push_back(s.r[e[1]]?nullptr:std::make_unique<NSWard<S>>(w,std::array<ScaModule<S>*,3>{wm[e[0]].get(),wm[e[1]].get(),wm[e[2]].get()}));
    }
    Row coefficient(const Key&k,const Case&original,const std::vector<int>&marked={}){
        Case c=original;if(fermion){std::fill(c.f.begin(),c.f.end(),0);std::fill(c.eta.begin(),c.eta.end(),1);}
        Row out;int num=setup.graph.edges;
        for(int mask=0;mask<(1<<num);mask++){
            bool ok=true;for(int e=0;e<num;e++)if(!setup.r[e]&&((mask>>e)&1)!=k[e]%2)ok=false;
            for(int v=0;v<int(setup.graph.slots.size());v++){int par=0;for(int e:setup.graph.slots[v])par^=(mask>>e)&1;if(par!=c.f[v])ok=false;}if(!ok)continue;
            std::array<const EdgeData*,8> left{},right{};std::array<const Matrix*,8>props{};
            for(int e=0;e<num;e++){
                auto it=std::find(marked.begin(),marked.end(),e);int r=it==marked.end()?k[e]:k[num+int(it-marked.begin())];int par=(mask>>e)&1;
                left[e]=&edge(e,k[e],par);right[e]=&edge(e,r,par);props[e]=&prop(e,k[e],r,par,it!=marked.end());if(left[e]->states.empty()||right[e]->states.empty())ok=false;
            }if(!ok)continue;S ans=0;
            if(setup.name=="mercedes"){
                std::array<int,3>d{int(left[0]->states.size()),int(left[1]->states.size()),int(left[2]->states.size())};std::vector<S>center;
                for(auto&a:left[0]->states)for(auto&b:left[1]->states)for(auto&cc:left[2]->states)center.push_back(vertex(0,a,b,cc,c.f[0],1));
                center=axis_transform(std::move(center),d,{props[0],props[1],props[2]});int ix=0;
                for(auto&s0:right[0]->states)for(auto&s1:right[1]->states)for(auto&s2:right[2]->states){S cv=center[ix++];if(cv==S(0))continue;
                    Matrix v1(right[5]->states.size(),left[3]->states.size()),v2(right[3]->states.size(),left[4]->states.size()),v3(right[4]->states.size(),left[5]->states.size());
                    for(int i=0;i<v1.n;i++)for(int j=0;j<v1.m;j++)v1(i,j)=vertex(1,s0,left[3]->states[j],right[5]->states[i],c.f[1],c.eta[1]);
                    for(int i=0;i<v2.n;i++)for(int j=0;j<v2.m;j++)v2(i,j)=vertex(2,s1,left[4]->states[j],right[3]->states[i],c.f[2],c.eta[2]);
                    for(int i=0;i<v3.n;i++)for(int j=0;j<v3.m;j++)v3(i,j)=vertex(3,s2,left[5]->states[j],right[4]->states[i],c.f[3],c.eta[3]);
                    ans+=cv*trace(mul(mul(mul(mul(mul(v1,*props[3]),v2),*props[4]),v3),*props[5]));
                }
            }else if(setup.name=="theta_ns"){
                std::array<int,3>d{int(left[0]->states.size()),int(left[1]->states.size()),int(left[2]->states.size())};std::vector<S>tensor;
                for(auto&a:left[0]->states)for(auto&b:left[1]->states)for(auto&cc:left[2]->states)tensor.push_back(vertex(0,a,b,cc,c.f[0],1));
                auto transformed=axis_transform(tensor,d,{props[0],props[1],props[2]});for(int j=0;j<int(tensor.size());j++)ans+=tensor[j]*transformed[j];
            }else{
                std::vector<S>t[2];for(int v=0;v<2;v++)for(auto&a:left[0]->states){S z=0;int e=v+1;
                    for(int i=0;i<int(left[e]->states.size());i++)for(int j=0;j<int(right[e]->states.size());j++)z+=(*props[e])(i,j)*vertex(v,a,left[e]->states[i],right[e]->states[j],c.f[v],c.eta[v]);t[v].push_back(z);
                }for(int i=0;i<int(t[0].size());i++)for(int j=0;j<int(t[1].size());j++)ans+=t[0][i]*(*props[0])(i,j)*t[1][j];
            }
            ans*=S(setup.graph.sign_of(mask));if(fermion)for(auto e:setup.graph.slots)if(setup.r[e[1]])ans*=power(S(Machine(0,1)),(mask>>e[0])&1);
            if(ans!=S(0))out[mask]=ans;
        }return out;
    }
};

class DoubleVirasoro {
    const Setup&s;int level;S b=parse<S>("7/5"),q=b+S(1)/b;
    ScaWords<S>w;std::vector<std::unique_ptr<ScaModule<S>>>modules;
    std::vector<std::unique_ptr<NSWard<S>>>nsforms;
    std::vector<std::unique_ptr<NSBranches<S>>>nsbranch;
    std::vector<std::array<std::unique_ptr<OuterBranching<S>>,2>>outer;
    std::map<int,std::unique_ptr<RamondActions<S>>>actions;
    std::map<int,std::unique_ptr<MiddleBranching<S>>>middle;
    QPoly vacuum;
    S local(int v,const Key&labels,int mask,const Case&c,const Graph&g){
        auto e=g.slots[v];std::array<int,3>n{labels[e[0]],labels[e[1]],labels[e[2]]};auto original=s.graph.slots[v];
        if(!s.r[original[1]])return nsbranch[v]->raw(n);
        int a=(mask>>original[1])&1,cc=(mask>>original[2])&1,f=c.f[v];
        if(!outer[v][f]){
            std::cerr<<"branching vertex="<<v<<" f="<<f<<" start\n";
            outer[v][f]=std::make_unique<OuterBranching<S>>(b,std::array<S,3>{s.p[original[0]],s.p[original[1]],s.p[original[2]]},level,f,0,false,true);
            outer[v][f]->prepare(1,-1);std::cerr<<"branching vertex="<<v<<" f="<<f<<" done\n";
        }
        return outer[v][f]->raw(n[0],n[1],n[2],a,cc,c.eta[v]);
    }
public:
    size_t branches=0,ccy_transitions=0;
    DoubleVirasoro(const Setup&setup,int l):s(setup),level(l){
        S c=rational<S>(3,2)+S(3)*q*q;
        for(int e=0;e<s.graph.edges;e++)modules.push_back(std::make_unique<ScaModule<S>>(w,q*q/S(8)-s.p[e]*s.p[e]/S(2),c,s.p[e]/root(S(2)),false,true));
        outer.resize(s.graph.slots.size());
        for(auto es:s.graph.slots){
            if(s.r[es[1]]){nsforms.push_back(nullptr);nsbranch.push_back(nullptr);}
            else{nsforms.push_back(std::make_unique<NSWard<S>>(w,std::array<ScaModule<S>*,3>{modules[es[0]].get(),modules[es[1]].get(),modules[es[2]].get()}));nsbranch.push_back(std::make_unique<NSBranches<S>>(w,b,std::array<S,3>{s.p[es[0]],s.p[es[1]],s.p[es[2]]},*nsforms.back()));}
        }
        vacuum=schottky(s.graph,level);vacuum=multiply(vacuum,vacuum,level);
    }
    std::vector<Block> compute(const std::vector<int>&marked,const std::vector<int>&case_ids){
        Graph g=s.graph.split(marked);auto targets=indices(s,level,marked);std::set<Key>targetset(targets.begin(),targets.end());
        std::vector<Block>answer(s.cases.size());Key labels{},base{};std::vector<std::vector<int>> choices(g.edges);
        for(int e=0;e<g.edges;e++){
            int original=e<s.graph.edges?e:marked[e-s.graph.edges];
            for(int n=-20;n<=20;n++)if(s.r[original]?n%2!=0:n%2==0){int shift=s.r[original]?(n*n-1)/4:n*n/4;if(shift<=2*level)choices[e].push_back(n);}
        }
        auto budget=[&](const Key&k){int sum=0;for(int e=0;e<s.graph.edges;e++){auto it=std::find(marked.begin(),marked.end(),e);sum+=it==marked.end()?k[e]:std::max(k[e],k[s.graph.edges+int(it-marked.begin())]);}return sum;};
        std::function<void(int)>visit=[&](int edge){
            if(edge<g.edges){int original=edge<s.graph.edges?edge:marked[edge-s.graph.edges];for(int n:choices[edge]){
                if(edge>=s.graph.edges&&std::abs(n-labels[original])!=2)continue;
                labels[edge]=n;base[edge]=s.r[original]?(n*n-1)/4:n*n/4;
                if(budget(base)<=2*level)visit(edge+1);
            }base[edge]=0;labels[edge]=0;return;}
            std::vector<Key>allowed,output;
            if(marked.empty() && std::none_of(s.r.begin(),s.r.end(),[](bool r){return r;})){
                Key k{};int remaining=(2*level-total(base))/2;
                std::function<void(int,int)> enumerate=[&](int e,int budget){if(e==g.edges){allowed.push_back(k);Key target=base;for(int j=0;j<g.edges;j++)target[j]+=2*k[j];output.push_back(target);return;}for(int n=0;n<=budget;n++){k[e]=n;enumerate(e+1,budget-n);}k[e]=0;};enumerate(0,remaining);
            }else for(auto target:targets)if(leq(base,target)){
                Key k=minus(target,base);bool integral=true;for(int e=0;e<g.edges;e++){if(k[e]%2)integral=false;k[e]/=2;}if(integral){allowed.push_back(k);output.push_back(target);}
            }if(allowed.empty())return;
            // The retained target lattice has fixed branch parity; it is downward closed.
            std::vector<Row>factors(s.cases.size());bool nonzero=false;
            for(int ci:case_ids)for(int mask=0;mask<(1<<s.graph.edges);mask++){
                const auto&c=s.cases[ci];bool ok=true;
                for(int e=0;e<s.graph.edges;e++)if(!s.r[e]&&((mask>>e)&1)!=((labels[e]/2)%2+2)%2)ok=false;
                for(int v=0;v<int(s.graph.slots.size());v++){int p=0;for(int e:s.graph.slots[v])p^=(mask>>e)&1;if(p!=c.f[v])ok=false;}if(!ok)continue;
                S factor=S(s.graph.sign_of(mask));for(int v=0;v<int(s.graph.slots.size());v++){
                    factor*=local(v,labels,mask,c,g);auto es=s.graph.slots[v];if(s.r[es[1]])factor*=power(S(Machine(0,1)),(mask>>es[0])&1);
                }if(factor==S(0))continue;
                for(int e=0;e<g.edges;e++){int original=e<s.graph.edges?e:marked[e-s.graph.edges];factor/=s.r[original]?r_norm(labels[e],(mask>>original)&1,b,s.p[original]):ns_norm(labels[e],b,s.p[original]);}
                for(int j=0;j<int(marked.size());j++){
                    int e=marked[j];if(!middle.count(e)){actions[e]=std::make_unique<RamondActions<S>>(b,s.p[e],false);middle[e]=std::make_unique<MiddleBranching<S>>(b,s.p[e],*actions[e]);}
                    factor*=middle[e]->raw(labels[s.graph.edges+j],labels[e],(mask>>e)&1);
                }
                if(factor!=S(0)){factors[ci][mask]=factor;nonzero=true;}
            }if(!nonzero)return;branches++;
            std::array<Poly<S>,2>vir;
            for(int copy=0;copy<2;copy++){
                std::vector<S>h;for(int e=0;e<g.edges;e++){int original=e<s.graph.edges?e:marked[e-s.graph.edges];h.push_back(branch_weight(copy,labels[e],b,s.p[original]));}
                S ext=copy?(b*b+S(2))/(S(2)*(S(1)-b*b)):-(S(1)+S(2)*b*b)/(S(2)*(S(1)-b*b));
                GraphCCY<S>engine(g,branch_central(copy,b),h,ext);vir[copy]=engine.reduced(allowed);ccy_transitions+=engine.transitions;
            }
            Poly<S> convolution;
            if(marked.empty()){
                int remaining=(2*level-total(base))/2;
                std::vector<std::pair<Key,S>> right(vir[1].begin(),vir[1].end());
                std::sort(right.begin(),right.end(),[](const auto&a,const auto&b){return total(a.first)<total(b.first);});
                for(const auto&[a,x]:vir[0])if(x!=S(0))for(const auto&[b,y]:right){if(total(a)+total(b)>remaining)break;if(y!=S(0))convolution[plus(a,b)]+=x*y;}
            }
            for(size_t j=0;j<allowed.size();j++){
                auto k=allowed[j],target=output[j];S value=0;
                if(marked.empty()){auto it=convolution.find(k);if(it!=convolution.end())value=it->second;}
                else for(auto&[a,x]:vir[0])if(leq(a,k)){auto it=vir[1].find(minus(k,a));if(it!=vir[1].end())value+=x*it->second;}
                if(value!=S(0))for(int ci:case_ids)for(auto [mask,f]:factors[ci])answer[ci][target][mask]+=value*f;
            }
            if(branches%100==0)std::cerr<<"DV branches="<<branches<<" CCY transitions="<<ccy_transitions<<"\n";
        };visit(0);
        for(int ci:case_ids){Block full;for(auto&[k,row]:answer[ci])for(auto[v,c]:vacuum){Key shift{};for(int e=0;e<s.graph.edges;e++)shift[e]=2*v[e];for(int j=0;j<int(marked.size());j++)shift[s.graph.edges+j]=shift[marked[j]];auto key=plus(k,shift);if(targetset.count(key))for(auto[mask,x]:row)full[key][mask]+=x*from_rational<S>(c);}answer[ci]=std::move(full);}
        return answer;
    }
};
struct Error {
    long components=0,failed=0;double scaled=0,absolute=0;Key worst{};int parity=0;S actual=0,expected=0;
    void check(const Row&a,const Row&b,const Key&k,int bits){components+=1<<bits;for(int p=0;p<(1<<bits);p++){
        S x=getrow(a,p),y=getrow(b,p);double e=magnitude(x-y),r=e/std::max({1.,magnitude(x),magnitude(y)});if(!std::isfinite(e))throw std::runtime_error("nonfinite comparison");absolute=std::max(absolute,e);if(r>scaled){scaled=r;worst=k;parity=p;actual=x;expected=y;}if(r>1e-18)failed++;
    }}
    void json(std::ostream&o)const{o<<"{\"components\":"<<components<<",\"failed_components\":"<<failed<<",\"max_scaled\":"<<scaled<<",\"max_absolute\":"<<absolute<<",\"worst_level2\":[";for(int e=0;e<8;e++){if(e)o<<',';o<<worst[e];}o<<"],\"parity\":"<<parity<<"}";}
};
void row_json(std::ostream&o,const Row&r){o<<'[';bool first=true;for(auto[k,x]:r)if(x!=S(0)){if(!first)o<<',';first=false;o<<'['<<k<<",\""<<decimal(mpc_realref(x.data()))<<"\",\""<<decimal(mpc_imagref(x.data()))<<"\"]";}o<<']';}
