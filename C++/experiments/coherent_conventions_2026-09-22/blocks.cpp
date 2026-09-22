#include "unified_ward.hpp"
#include "ramond/pipeline.hpp"
#include <fstream>
#include <iostream>
using namespace coherent;
struct Audit {size_t count=0,failed=0;double maximum=0;void check(MP x,MP y){double e=magnitude(x-y)/std::max({1.,magnitude(x),magnitude(y)});count++;maximum=std::max(maximum,e);if(e>1e-25)failed++;}};
struct Edge {std::vector<int> basis;std::vector<MP> inverse;};
class Sew {
    ScaWords<MP> w;std::array<std::unique_ptr<ScaModule<MP>>,3> mod;
    std::array<std::unique_ptr<Physical<MP>>,2> rho;
    std::map<std::array<int,3>,Edge> edges;
    std::map<std::array<int,3>,std::array<MP,8>> physical_cache;
    Auxiliary<MP> aux;
    int f;bool opposite;MP Q;
    const Edge& edge(int slot,int lev2,int parity){
        std::array<int,3>key{slot,lev2,parity};auto old=edges.find(key);if(old!=edges.end())return old->second;
        Edge e;int level=slot?lev2/2:lev2;
        for(int l=0;l<=(slot?level:level/2);l++)for(auto ls:partitions(l))for(auto gs:partitions(level-(slot?l:2*l),true,!slot)){
            int ground=slot?(parity+gs.size())%2:0;if((ground+gs.size())%2!=parity)continue;
            ScaWord word;for(int n:ls)word.push_back({0,-2*n});for(int n:gs)word.push_back({1,-(slot?2:1)*n});e.basis.push_back(2*w.intern(word)+ground);
        }
        int n=e.basis.size();std::vector<MP>g(n*n);MP I(Machine(0,1));
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)g[i*n+j]=power(I,e.basis[i]%2)*mod[slot]->inner(e.basis[i],e.basis[j]);
        if(n)e.inverse=pbw_inverse(std::move(g),n);return edges.emplace(key,std::move(e)).first->second;
    }
    static MP contract(const std::array<const Edge*,3>&e,const std::vector<MP>&a,const std::vector<MP>&b){
        size_t d[3]={e[0]->basis.size(),e[1]->basis.size(),e[2]->basis.size()},volume=a.size();
        std::vector<MP>x=b,y(volume);
        for(int axis=0;axis<3;axis++){
            size_t stride=axis==0?d[1]*d[2]:axis==1?d[2]:1,n=d[axis];
            for(size_t base=0;base<volume;base+=n*stride)for(size_t off=0;off<stride;off++)for(size_t i=0;i<n;i++){
                MP v=0;for(size_t j=0;j<n;j++)v+=e[axis]->inverse[i*n+j]*x[base+j*stride+off];y[base+i*stride+off]=v;
            }x.swap(y);
        }MP ans=0;for(size_t i=0;i<volume;i++)ans+=a[i]*x[i];return ans;
    }
public:
    Sew(MP b,std::array<MP,3>P,int ff,bool opp):f(ff),opposite(opp),Q(b+MP(1)/b){
        MP c=rational<MP>(3,2)+MP(3)*Q*Q;
        for(int j=0;j<3;j++)mod[j]=std::make_unique<ScaModule<MP>>(w,Q*Q/MP(8)-P[j]*P[j]/MP(2)+(j?rational<MP>(1,16):MP(0)),c,P[j]/root(MP(2)),j!=0,false);
        std::array<ScaModule<MP>*,3>mm{mod[0].get(),mod[1].get(),mod[2].get()};
        rho[0]=std::make_unique<Physical<MP>>(w,mm,f,1);rho[1]=std::make_unique<Physical<MP>>(w,mm,f,opposite?-1:1);
    }
    std::array<MP,8> physical(std::array<int,3>l){
        auto old=physical_cache.find(l);if(old!=physical_cache.end())return old->second;std::array<MP,8>out{};
        for(int p2=0;p2<2;p2++){
            int p1=l[0]%2,p3=(f+p1+p2)%2,index=p1+2*p2+4*p3;
            std::array<const Edge*,3>e{&edge(0,l[0],p1),&edge(1,l[1],p2),&edge(2,l[2],p3)};
            std::vector<MP>a,b;for(int x:e[0]->basis)for(int y:e[1]->basis)for(int z:e[2]->basis){a.push_back(rho[0]->value({x,y,z}));b.push_back(rho[1]->value({x,y,z}));}
            if(!a.empty())out[index]=MP(theta_sign(index))*contract(e,a,b);
        }physical_cache[l]=out;return out;
    }
    std::array<MP,8> fermion(std::array<int,3>l,bool inserted){
        std::array<MP,8>out{};
        for(auto a:partitions(l[0],true,true))for(auto b:partitions(l[1]/2,true,false))for(auto c:partitions(l[2]/2,true,false))for(int ground=0;ground<2;ground++){
            int cg=(a.size()+b.size()+c.size()+ground)%2;
            AuxTriple s{AuxState{a,0},AuxState{b,ground},AuxState{c,cg}};
            int p=aux_parity(0,s[0])+2*aux_parity(1,s[1])+4*aux_parity(2,s[2]);
            MP v=aux.value(s);MP weight=MP(theta_sign(p)*sign(aux_parity(0,s[0])+aux_parity(1,s[1])+aux_parity(2,s[2])));
            if(inserted)weight*=Q/ root(MP(2))*MP(sign(ground));out[p]+=weight*v*v;
        }return out;
    }
    // Tensor-PBW sewing: inverse auxiliary Gram signs, tensor parity
    // permutation, and the two local middle-to-ket signs (which cancel).
    std::array<MP,8> enlarged(std::array<int,3>l,bool inserted){
        std::array<MP,8>out{};
        for(int a=0;a<=l[0];a++)for(int b=0;b<=l[1];b+=2)for(int c=0;c<=l[2];c+=2){
            auto p=physical({a,b,c});std::array<int,3>al{l[0]-a,l[1]-b,l[2]-c};
            for(auto A:partitions(al[0],true,true))for(auto B:partitions(al[1]/2,true,false))for(auto C:partitions(al[2]/2,true,false))for(int g=0;g<2;g++){
                AuxTriple s{AuxState{A,0},AuxState{B,g},AuxState{C,int((A.size()+B.size()+C.size()+g)%2)}};
                int ap=aux_parity(0,s[0]),bp=aux_parity(1,s[1]),cp=aux_parity(2,s[2]),bits=ap+2*bp+4*cp;
                MP v=aux.value(s),factor=MP(sign(ap+bp+cp))*v*v;if(inserted)factor*=Q/root(MP(2))*MP(sign(g));
                for(int k=0;k<8;k++)out[k^bits]+=MP(theta_sign(k^bits)*theta_sign(k))*factor*p[k];
            }
        }return out;
    }
};
int main(){try{
    MP::precision(40);double start=seconds();Settings settings;settings.level=3;settings.dps=40;settings.record_sector=true;
    MP b=parse<MP>(settings.b);std::array<MP,3>P;for(int j=0;j<3;j++)P[j]=parse<MP>(settings.momenta[j]);
    Audit physical_ccy,aux_reference,convolution,enlarged_ccy,vanishing;std::ofstream out("block_results.json");
    out<<std::setprecision(17)<<"{\"dps\":40,\"total_level\":3,\"runs\":[";bool comma=false;
    auto vacuum=qmultiply(schottky_vacuum(SeriesDomain(3)),schottky_vacuum(SeriesDomain(3)),SeriesDomain(3));
    for(int f=0;f<2;f++)for(bool opposite:{false,true}){
        double tick=seconds();Sew sew(b,P,f,opposite);ParitySeries<MP>phys,ff,hat;
        for(auto k:physical_indices(6)){
            std::array<int,3>l{k[0],2*k[1],k[3]};phys[k]=sew.physical(l);ff[k]=sew.fermion(l,opposite);hat[k]=sew.enlarged(l,opposite);
            if(opposite){auto zero=sew.enlarged(l,false);for(auto v:zero)vanishing.check(v,MP(0));}
        }
        double sew_seconds=seconds()-tick;auto oldff=numerical_fermion(3,opposite,b+MP(1)/b);
        for(auto &[k,row]:ff)for(int i=0;i<8;i++)aux_reference.check(row[i],oldff[k][i]);
        ParitySeries<MP>product;
        for(auto &[a,x]:phys)for(auto &[d,y]:ff)if(degree(a)+degree(d)<=6)for(int i=0;i<8;i++)for(int j=0;j<8;j++)product[a+d][i^j]+=MP(star_sign(i,j))*x[i]*y[j];
        for(auto &[k,row]:hat)for(int i=0;i<8;i++)convolution.check(row[i],product[k][i]);
        settings.f=f;settings.inserted=opposite;auto dv=pipeline<MP>(settings);ParitySeries<MP>full;
        for(auto &[v,coeff]:vacuum){Index shift{2*v[0],v[1],v[1],2*v[2]};for(auto &[k,row]:dv.numerator)if(degree(k+shift)<=6)for(int i=0;i<8;i++)full[k+shift][i]+=from_rational<MP>(coeff)*row[i];}
        for(auto &[k,row]:phys)for(int i=0;i<8;i++){physical_ccy.check(row[i],dv.physical[k][i]);enlarged_ccy.check(hat[k][i],full[k][i]);}
        if(comma)out<<',';comma=true;out<<"{\"f\":"<<f<<",\"eta_product\":"<<(opposite?-1:1)<<",\"new_PBW_seconds\":"<<sew_seconds<<",\"CCY_seconds\":"<<dv.timing.total<<",\"physical\":";encode_series(out,phys);out<<",\"auxiliary\":";encode_series(out,ff);out<<",\"enlarged\":";encode_series(out,hat);out<<'}';
        std::cerr<<"block f="<<f<<" opposite="<<opposite<<" done\n";
    }
    out<<"],\"checks\":{";comma=false;size_t failed=0;
    for(auto item:std::vector<std::pair<std::string,Audit>>{{"physical_PBW_vs_CCY",physical_ccy},{"auxiliary_PBW_vs_reference",aux_reference},{"PBW_convolution",convolution},{"enlarged_PBW_vs_CCY",enlarged_ccy},{"odd_loop_without_insertion",vanishing}}){
        if(comma)out<<',';comma=true;auto&a=item.second;failed+=a.failed;out<<'"'<<item.first<<"\":{\"cases\":"<<a.count<<",\"failed\":"<<a.failed<<",\"maximum_scaled_error\":"<<a.maximum<<'}';std::cout<<item.first<<": "<<a.count<<" cases; "<<a.failed<<" failed; max "<<a.maximum<<'\n';
    }out<<"},\"seconds\":"<<seconds()-start<<",\"failed\":"<<failed<<"}\n";return failed?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
