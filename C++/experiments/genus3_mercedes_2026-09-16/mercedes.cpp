#include "ramond/direct_pbw.hpp"
#include "ramond/branching.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace ramond;
using S=Machine;
#include "central_data.hpp"
using Levels=std::array<int,6>;
using Row=std::array<S,64>;
using GraphSeries=std::map<Levels,Row>;
constexpr int slots[4][3]={{0,1,2},{0,3,5},{1,4,3},{2,5,4}};
constexpr int pairs[6][2]={{0,3},{1,6},{2,9},{4,8},{7,11},{10,5}};
int bit(int x,int e){return (x>>e)&1;}
int graph_sign(int mask){
    int p[12],order[12],t=0,exponent=0;
    for(int e=0;e<6;e++)for(int h:pairs[e]){p[h]=bit(mask,e);order[t++]=h;}
    for(int i=0;i<12;i++)for(int j=i+1;j<12;j++)
        if(order[i]>order[j])exponent+=p[order[i]]*p[order[j]];
    return sign(exponent);
}
int kernel(int a,int b){
    int local=0;for(auto &v:slots)local+=bit(a,v[1])*bit(b,v[2]);
    return graph_sign(a^b)*graph_sign(a)*graph_sign(b)*sign(local);
}
int transport(int mask,const std::array<int,3>&f){
    int exponent=0;for(int i=0;i<3;i++)exponent+=f[i]*(bit(mask,i)+bit(mask,3+i));
    return sign(exponent);
}
S virnorm(S h,int n){return n?S(2)*h:S(1);}
// Exact L_-1 global vertex; all Virasoro descendant levels are <=1.
S vrho(int i,int j,int k,std::array<S,3>h){
    S ans=0;for(int p=0;p<=std::min(i,k);p++){
        S t=S(binomial(i,p))*rising(S(2)*h[2]+S(k-p),p);
        for(int u=0;u<p;u++)t*=S(k-u);
        t*=rising(h[2]+h[1]-h[0],k-p)*rising(h[0]+h[1]-h[2]+S(-k+p),i-p);ans+=t;
    }return ans*rising(h[0]-h[1]-h[2]+S(i-j+1-k),j);
}
struct Mat{
    int n=0,m=0;std::vector<S>v;
    Mat()=default;Mat(int a,int b):n(a),m(b),v(a*b){}
    S& operator()(int i,int j){return v[i*m+j];}
    const S& operator()(int i,int j)const{return v[i*m+j];}
};
Mat mul(const Mat&a,const Mat&b){
    require(a.m==b.n,"network matrix dimensions");Mat c(a.n,b.m);
    for(int i=0;i<a.n;i++)for(int k=0;k<a.m;k++)if(a(i,k)!=S(0))
        for(int j=0;j<b.m;j++)c(i,j)+=a(i,k)*b(k,j);
    return c;
}
S tr(const Mat&a){require(a.n==a.m,"network trace dimensions");S z=0;for(int i=0;i<a.n;i++)z+=a(i,i);return z;}
struct St{int id,lev,par,pbw=0,n4=0,d0=0,d1=0;AuxState aux{};};
enum Kind{PHYSICAL,DV,FERMION};
class Network{
    Kind kind;S b=S(1.4),q=b+S(1)/b,c=S(1.5)+S(3)*q*q;
    std::array<S,6>p{11./23,13./29,17./31,19./37,23./41,29./43};
    ScaWords<S>w;
    std::array<std::unique_ptr<ScaModule<S>>,6>grammod,wardmod;
    std::array<std::array<std::array<std::unique_ptr<ScaWard<S>>,2>,2>,3>ward;
    std::array<std::unique_ptr<LowAnchors<S>>,3>anchors;
    RamondActions<S>actions{b,p[3],false};MiddleBranching<S>middle{b,p[3],actions};
    FermionForm ff;
    std::array<std::vector<St>,6>states;
    std::array<std::array<std::array<std::vector<St>,2>,3>,6>basis;
    std::map<std::array<int,5>,Mat>prop_cache;
    std::map<std::array<int,6>,S>vertex_cache;
    std::map<std::array<int,7>,S>branch_cache;
    void add(int e,St s){s.id=states[e].size();states[e].push_back(s);basis[e][s.lev][s.par].push_back(s);}
    S inverse_norm(int e,const St&s){
        if(kind==FERMION)return S(sign(s.par));
        S n=e<3?ns_norm(s.n4,b,p[e]):r_norm(s.n4,s.par,b,p[e]);
        return S(1)/(n*virnorm(branch_weight(0,s.n4,b,p[e]),s.d0)*virnorm(branch_weight(1,s.n4,b,p[e]),s.d1));
    }
    const Mat&prop(int e,int l,int r,int par,bool ins){
        std::array<int,5>key{e,l,r,par,int(ins)};auto found=prop_cache.find(key);if(found!=prop_cache.end())return found->second;
        auto &left=basis[e][l][par],&right=basis[e][r][par];Mat m(left.size(),right.size());
        if(kind==PHYSICAL){
            require(!ins&&l==r,"physical propagator is unsplit");
            for(int i=0;i<m.n;i++)for(int j=0;j<m.n;j++)m(i,j)=grammod[e]->inner(left[i].pbw,left[j].pbw);
            m.v=pbw_inverse(m.v,m.n);
        }else if(!ins){
            require(l==r,"ordinary edge levels differ");for(int i=0;i<m.n;i++)m(i,i)=inverse_norm(e,left[i]);
        }else if(kind==FERMION){
            require(e==3&&m.n==1&&m.m==1,"split auxiliary basis");
            m(0,0)=l==r?S(sign(par+left[0].aux.ground))*q/root(S(2)):q;
        }else{
            require(e==3,"split DV edge");
            for(int i=0;i<m.n;i++)for(int j=0;j<m.m;j++){
                auto &a=left[i],&z=right[j];if(std::abs(a.n4-z.n4)!=2)continue;
                S val=middle.raw(z.n4,a.n4,par)*inverse_norm(e,a)*inverse_norm(e,z);
                for(int cp=0;cp<2;cp++){
                    S ext=cp?(b*b+S(2))/(S(2)*(S(1)-b*b)):-(S(1)+S(2)*b*b)/(S(2)*(S(1)-b*b));
                    val*=vrho(cp?z.d1:z.d0,0,cp?a.d1:a.d0,{branch_weight(cp,z.n4,b,p[e]),ext,branch_weight(cp,a.n4,b,p[e])});
                }m(i,j)=val;
            }
        }return prop_cache.emplace(key,std::move(m)).first->second;
    }
    S vertex(int v,const St&a,const St&z,const St&t,int f,int eta){
        std::array<int,6>key{v,a.id,z.id,t.id,f,eta};auto found=vertex_cache.find(key);if(found!=vertex_cache.end())return found->second;
        if((a.par+z.par+t.par)%2!=f)return vertex_cache[key]=S(0);
        S val=0;
        if(v==0){
            if(kind==PHYSICAL)val=central_physical[9*a.lev+3*z.lev+t.lev];
            else if(kind==FERMION)val=central_fermion[4*a.lev+2*z.lev+t.lev];
            else val=central_branch[9*(a.n4/2+1)+3*(z.n4/2+1)+(t.n4/2+1)];
        }else if(kind==PHYSICAL){
            auto &form=ward[v-1][f][eta==1];
            if(!form)form=std::make_unique<ScaWard<S>>(w,std::array<ScaModule<S>*,3>{wardmod[slots[v][0]].get(),wardmod[slots[v][1]].get(),wardmod[slots[v][2]].get()},0,f,eta);
            val=power((S(-1)+S(0,1))/root(S(2)),z.pbw%2+t.pbw%2)*form->value({a.pbw,z.pbw,t.pbw});
        }else if(kind==FERMION)val=ff.complex_value<S>({a.aux,z.aux,t.aux});
        else{
            std::array<int,7>bk{v,a.n4,z.n4,t.n4,z.par,t.par,sign(f)*eta};
            auto it=branch_cache.find(bk);
            if(it==branch_cache.end())it=branch_cache.emplace(bk,anchors[v-1]->raw({a.n4,z.n4,t.n4},z.par,t.par,sign(f)*eta)).first;
            val=it->second;
        }
        if(kind==DV)for(int cp=0;cp<2;cp++)val*=vrho(cp?a.d1:a.d0,cp?z.d1:z.d0,cp?t.d1:t.d0,
            {branch_weight(cp,a.n4,b,p[slots[v][0]]),branch_weight(cp,z.n4,b,p[slots[v][1]]),branch_weight(cp,t.n4,b,p[slots[v][2]])});
        return vertex_cache[key]=val;
    }
public:
    Network(Kind k):kind(k){
        for(int e=0;e<6;e++){
            if(kind==PHYSICAL){
                S h=q*q/S(8)-p[e]*p[e]/S(2)+(e<3?S(0):S(1./16));
                grammod[e]=std::make_unique<ScaModule<S>>(w,h,c,p[e]/root(S(2)),e>=3,true);
                wardmod[e]=std::make_unique<ScaModule<S>>(w,h,c,p[e]/root(S(2)),e>=3,false);
                for(int level=0;level<=2;level+=(e<3?1:2))for(int ground=0;ground<(e<3?1:2);ground++){
                    std::vector<ScaWord>words;
                    if(level==0)words.push_back({});
                    if(level==1)words.push_back({{1,-1}});
                    if(level==2){words.push_back({{0,-2}});if(e>=3)words.push_back({{1,-2}});}
                    for(auto word:words){int id=w.intern(word);St s{};s.lev=level;s.par=(w.words[id].parity+ground)%2;s.pbw=2*id+ground;add(e,s);}
                }
            }else if(kind==FERMION){
                for(int occ=0;occ<2;occ++)for(int g=0;g<(e<3?1:2);g++){
                    St s{};s.lev=occ*(e<3?1:2);s.par=(occ+g)%2;s.aux={occ?Part{1}:Part{},g};add(e,s);
                }
            }else{
                std::vector<int>ns=e<3?std::vector<int>{-2,0,2}:std::vector<int>{-3,-1,1,3};
                for(int n:ns)for(int par=0;par<(e<3?1:2);par++)for(int a=0;a<2;a++)for(int z=0;z<2;z++){
                    int lev=(e<3?n*n/4:2*ramond_level(n))+2*(a+z);if(lev>2)continue;
                    St s{};s.n4=n;s.lev=lev;s.par=e<3?((n/2)%2+2)%2:par;s.d0=a;s.d1=z;add(e,s);
                }
            }
        }
        if(kind==DV)for(int v=1;v<4;v++)anchors[v-1]=std::make_unique<LowAnchors<S>>(b,std::array<S,3>{p[slots[v][0]],p[slots[v][1]],p[slots[v][2]]},0);
    }
    Row coefficient(Levels lev,std::array<int,3>f,std::array<int,3>eta,bool insert,int right=-1){
        if(right<0)right=lev[3];Row out{};
        if(kind==FERMION){f={0,0,0};eta={1,1,1};}
        int nspar[3]={lev[0]%2,lev[1]%2,lev[2]%2};
        int a=(f[0]+f[1]+f[2])%2;if((nspar[0]+nspar[1]+nspar[2])%2!=a)return out;
        for(int i=0;i<3;i++)if(basis[i][lev[i]][nspar[i]].empty())return out;
        for(int rim=0;rim<8;rim++){
            int rp[3]={bit(rim,0),bit(rim,1),bit(rim,2)};bool ok=true;
            for(int i=0;i<3;i++)if((nspar[i]+rp[i]+rp[(i+2)%3])%2!=f[i])ok=false;
            if(!ok)continue;
            int mask=nspar[0]|(nspar[1]<<1)|(nspar[2]<<2)|(rim<<3);
            const auto &k3=prop(3,lev[3],right,rp[0],insert),&k4=prop(4,lev[4],lev[4],rp[1],false),&k5=prop(5,lev[5],lev[5],rp[2],false);
            const auto &r3l=basis[3][lev[3]][rp[0]],&r3r=basis[3][right][rp[0]],&r4=basis[4][lev[4]][rp[1]],&r5=basis[5][lev[5]][rp[2]];
            S result=0;
            for(auto &s0:basis[0][lev[0]][nspar[0]])for(auto &s1:basis[1][lev[1]][nspar[1]])for(auto &s2:basis[2][lev[2]][nspar[2]]){
                S central=vertex(0,s0,s1,s2,a,1);if(central==S(0))continue;
                const St ns[3]={s0,s1,s2};
                for(int e=0;e<3;e++){
                    auto &bs=basis[e][lev[e]][nspar[e]];int index=0;while(bs[index].id!=ns[e].id)index++;
                    central*=prop(e,lev[e],lev[e],nspar[e],false)(index,index);
                }
                Mat v1(r5.size(),r3l.size()),v2(r3r.size(),r4.size()),v3(r4.size(),r5.size());
                for(int i=0;i<v1.n;i++)for(int j=0;j<v1.m;j++)v1(i,j)=vertex(1,s0,r3l[j],r5[i],f[0],eta[0]);
                for(int i=0;i<v2.n;i++)for(int j=0;j<v2.m;j++)v2(i,j)=vertex(2,s1,r4[j],r3r[i],f[1],eta[1]);
                for(int i=0;i<v3.n;i++)for(int j=0;j<v3.m;j++)v3(i,j)=vertex(3,s2,r5[j],r4[i],f[2],eta[2]);
                result+=central*tr(mul(mul(mul(mul(mul(v1,k3),v2),k4),v3),k5));
            }
            out[mask]=S(graph_sign(mask))*result;
            if(kind!=PHYSICAL)out[mask]*=power(S(0,1),nspar[0]+nspar[1]+nspar[2]);
        }return out;
    }
};
void addstar(Row&out,const Row&a,const Row&b,S scale=S(1)){
    for(int x=0;x<64;x++)if(a[x]!=S(0))for(int y=0;y<64;y++)if(b[y]!=S(0))out[x^y]+=scale*S(kernel(x,y))*a[x]*b[y];
}
struct Error{
    double abs=0,scaled=0;long components=0,nonzero=0,failed=0;Levels worst{};int parity=0,right=-1;S x{},y{};
    void check(const Row&a,const Row&b,Levels l,int r=-1){for(int k=0;k<64;k++){
        double e=std::abs(a[k]-b[k]),s=e/std::max({1.,std::abs(a[k]),std::abs(b[k])});components++;if(std::max(std::abs(a[k]),std::abs(b[k]))>1e-12)nonzero++;
        abs=std::max(abs,e);if(s>scaled){scaled=s;worst=l;parity=k;right=r;x=a[k];y=b[k];}if(s>1e-8)failed++;
    }}
    void json()const{std::cout<<"{\"components\":"<<components<<",\"nonzero_components\":"<<nonzero<<",\"failed_components\":"<<failed<<",\"max_absolute\":"<<abs<<",\"max_scaled\":"<<scaled<<",\"worst_level2\":[";
        for(int i=0;i<6;i++){if(i)std::cout<<",";std::cout<<worst[i];}std::cout<<"],\"right_level2\":"<<right<<",\"worst_parity\":"<<parity<<",\"actual\":["<<x.real()<<","<<x.imag()<<"],\"expected\":["<<y.real()<<","<<y.imag()<<"]}";}
};
void sparse_json(std::ostream&o,const Row&r){bool first=true;o<<"[";for(int k=0;k<64;k++)if(r[k]!=S(0)){if(!first)o<<",";first=false;o<<"["<<k<<","<<r[k].real()<<","<<r[k].imag()<<"]";}o<<"]";}
int main(int argc,char**argv){try{
    bool quick=argc>1&&std::string(argv[1])=="--quick";double start=seconds();
    std::vector<Levels> levels;
    for(int a=0;a<3;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++)for(int d=0;d<3;d+=2)for(int e=0;e<3;e+=2)for(int f=0;f<3;f+=2)levels.push_back({a,b,c,d,e,f});
    std::sort(levels.begin(),levels.end(),[](auto a,auto b){int sa=0,sb=0;for(int x:a)sa+=x;for(int x:b)sb+=x;return sa==sb?a<b:sa<sb;});
    std::ofstream coefficients(quick?"quick_coefficients.jsonl":"coefficients.jsonl");coefficients<<std::setprecision(17);
    Network physical(PHYSICAL),dv(DV),fermion(FERMION);std::array<GraphSeries,2>aux;
    for(int ins=0;ins<2;ins++)for(auto l:levels)aux[ins][l]=fermion.coefficient(l,{0,0,0},{1,1,1},ins);
    double prep=seconds()-start;int cases=0,failures=0;std::array<int,2>counts{};
    std::cout<<std::setprecision(17)<<"{\"precision\":\"machine complex double\",\"individual_edge_level\":1,\"graph\":\"K4: NS spokes 0,1,2; R rim 3,4,5\",\"case_results\":[\n";
    for(int fm=0;fm<8;fm++)for(int signs=0;signs<8;signs++){
        if(quick&&(fm!=0&&fm!=1))continue;if(quick&&signs>1)continue;
        std::array<int,3>f{bit(fm,0),bit(fm,1),bit(fm,2)},eta{sign(bit(signs,0)),sign(bit(signs,1)),sign(bit(signs,2))};
        int ins=sign(f[0]+f[1]+f[2])*eta[0]*eta[1]*eta[2]<0;
        S constant=S(2)*(ins?(S(1.4)+S(1)/S(1.4))/root(S(2)):S(1));
        GraphSeries ph,hat,rec;Error forward,recovery,vanishing,sector,split,spin;double t=seconds();
        for(auto l:levels){ph[l]=physical.coefficient(l,f,eta,false);hat[l]=dv.coefficient(l,f,eta,ins);}
        for(auto l:levels){Row expected{},remainder=hat[l];
            for(auto r:levels){Levels s;bool ok=true,zero=true;for(int e=0;e<6;e++){s[e]=l[e]-r[e];if(s[e]<0)ok=false;if(s[e])zero=false;}if(!ok)continue;
                auto it=aux[ins].find(s);if(it==aux[ins].end())continue;Row transported=ph[r];for(int k=0;k<64;k++)transported[k]*=S(transport(k,f));
                addstar(expected,transported,it->second);
                if(!zero){auto old=rec.find(r);if(old!=rec.end())addstar(remainder,old->second,it->second,S(-1));}
            }
            forward.check(hat[l],expected,l);for(auto &x:remainder)x/=constant;rec[l]=remainder;
            Row recovered=remainder;for(int k=0;k<64;k++)recovered[k]*=S(transport(k,f));recovery.check(recovered,ph[l],l);
            Row spin_got{},spin_expected{};for(int lift=0;lift<64;lift++)for(int k=0;k<64;k++){
                int sgn=sign(__builtin_popcount(unsigned(lift&k)));spin_got[lift]+=S(sgn)*recovered[k];spin_expected[lift]+=S(sgn)*ph[l][k];
            }spin.check(spin_got,spin_expected,l);
            coefficients<<"{\"case\":"<<cases<<",\"level2\":[";for(int e=0;e<6;e++){if(e)coefficients<<",";coefficients<<l[e];}
            coefficients<<"],\"physical_pbw\":";sparse_json(coefficients,ph[l]);coefficients<<",\"enlarged_dv\":";sparse_json(coefficients,hat[l]);coefficients<<",\"recovered\":";sparse_json(coefficients,recovered);coefficients<<"}\n";
            Row jj{};S phase=power(S(0,1),-3)*S(graph_sign(56));for(int k=0;k<64;k++)jj[k^56]=phase*S(kernel(56,k))*ph[l][k]*S(eta[0]*eta[1]*eta[2]);sector.check(jj,ph[l],l);
            if(ins)vanishing.check(dv.coefficient(l,f,eta,false),Row{},l);
            if(ins)for(int right:{0,2})if(right!=l[3]){
                Row expect{};
                for(auto r:levels){Levels s;bool ok=true;for(int e=0;e<6;e++){s[e]=l[e]-r[e];if(s[e]<0)ok=false;}int sr=right-r[3];if(sr<0)ok=false;if(!ok)continue;
                    auto ff=fermion.coefficient(s,{0,0,0},{1,1,1},true,sr);Row x=ph[r];for(int k=0;k<64;k++)x[k]*=S(transport(k,f));addstar(expect,x,ff);
                }split.check(dv.coefficient(l,f,eta,true,right),expect,l,right);
            }
        }
        if(cases++)std::cout<<",\n";counts[ins]++;bool bad=forward.failed||recovery.failed||vanishing.failed||sector.failed||split.failed||spin.failed;failures+=bad;
        std::cout<<"{\"f\":["<<f[0]<<","<<f[1]<<","<<f[2]<<"],\"central_a\":"<<((f[0]+f[1]+f[2])%2)<<",\"eta\":["<<eta[0]<<","<<eta[1]<<","<<eta[2]<<"],\"inserted\":"<<(ins?"true":"false")<<",\"forward\":";forward.json();std::cout<<",\"recovery\":";recovery.json();std::cout<<",\"uninserted_vanishing\":";vanishing.json();std::cout<<",\"physical_sector\":";sector.json();std::cout<<",\"unequal_split\":";split.json();std::cout<<",\"spin_evaluations\":";spin.json();std::cout<<",\"elapsed_seconds\":"<<seconds()-t<<"}";
        std::cerr<<"case "<<cases<<" f="<<fm<<" eta="<<signs<<" insert="<<ins<<" error="<<recovery.scaled<<" split="<<split.scaled<<" time="<<seconds()-t<<"\n";
    }
    std::cout<<"\n],\"ordinary_cases\":"<<counts[0]<<",\"inserted_cases\":"<<counts[1]<<",\"failed_cases\":"<<failures<<",\"preparation_seconds\":"<<prep<<",\"runtime_seconds\":"<<seconds()-start<<"}\n";
    return failures?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
