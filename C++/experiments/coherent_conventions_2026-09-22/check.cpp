#include "unified_ward.hpp"
#include "../total_level6_2026-09-16/ns_local.hpp"
#include <fstream>
#include <iostream>
using namespace coherent;
struct Audit {size_t count=0,failed=0;double maximum=0;void check(MP x,MP y){
    double e=magnitude(x-y)/std::max({1.,magnitude(x),magnitude(y)});count++;maximum=std::max(maximum,e);if(e>1e-26)failed++;
}};
int main(){try{
    MP::precision(40);double start=seconds();MP b=parse<MP>("7/5"),Q=b+MP(1)/b,c=MP(3)*Q*Q+rational<MP>(3,2),I(Machine(0,1));
    MP t=(MP(-1)+I)/root(MP(2));std::array<MP,3>P{parse<MP>("11/23"),parse<MP>("13/29"),parse<MP>("17/31")};
    ScaWords<MP> words;std::array<std::unique_ptr<ScaModule<MP>>,3> modules;
    std::array<std::unique_ptr<FreeField<MP>>,3> free;
    std::array<std::unique_ptr<PBWModule<MP>>,3> pbw;
    std::array<std::vector<int>,3> states;std::array<std::vector<AuxState>,3> auxstates;
    for(int j=0;j<3;j++){
        MP h=Q*Q/MP(8)-P[j]*P[j]/MP(2)+(j?rational<MP>(1,16):MP(0));
        modules[j]=std::make_unique<ScaModule<MP>>(words,h,c,P[j]/root(MP(2)),j!=0,false);
        free[j]=std::make_unique<FreeField<MP>>(j!=0,b,P[j]);pbw[j]=std::make_unique<PBWModule<MP>>(*free[j]);
        for(int level2=0;level2<=4;level2+=j?2:1){
            int level=j?level2/2:level2;
            for(int l=0;l<=(j?level:level/2);l++)for(auto ls:partitions(l))for(auto gs:partitions(level-(j?l:2*l),true,!j)){
                ScaWord w;for(int n:ls)w.push_back({0,-2*n});for(int n:gs)w.push_back({1,-(j?2:1)*n});
                int id=words.intern(w);for(int ground=0;ground<(j?2:1);ground++)states[j].push_back(2*id+ground);
            }
            for(auto gs:partitions(level,true,!j))for(int ground=0;ground<(j?2:1);ground++)auxstates[j].push_back({gs,ground});
        }
    }
    std::array<ScaModule<MP>*,3> mm{modules[0].get(),modules[1].get(),modules[2].get()};
    Auxiliary<MP> auxiliary;Audit gward,fward,production_physical,production_aux,virasoro,cycle;
    for(auto a:auxstates[0])for(auto bb:auxstates[1])for(auto cc:auxstates[2]){
        AuxTriple s{a,bb,cc};if(sum(a.modes)+2*sum(bb.modes)+2*sum(cc.modes)>4)continue;
        for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++){
            MP r=0;double norm=0;for(auto [state,v]:auxiliary.equation(s,m,n))if(v!=MP(0)){MP z=v*auxiliary.value(state);r+=z;norm+=magnitude(z);}
            fward.check(r/MP(std::max(1.,norm)),MP(0));
        }
        FermionForm production;production_aux.check(auxiliary.value(s),production.complex_value<MP>(s));
    }
    for(int f=0;f<2;f++)for(int eta:{-1,1}){
        Physical<MP> physical(words,mm,f,eta);ScaWard<MP> production(words,mm,0,f,eta);
        for(int a:states[0])for(int bb:states[1])for(int cc:states[2]){
            std::array<int,3>s{a,bb,cc};if(words.words[a/2].level2+words.words[bb/2].level2+words.words[cc/2].level2>4)continue;
            production_physical.check(physical.value(s),production.value(s));
            auto odd_map=[&](int state){
                MP ground=state%2?-(MP(1)+I)/root(MP(2)):(MP(1)-I)/root(MP(2));
                return MP(sign(words.words[state/2].parity))*ground;
            };
            cycle.check(odd_map(bb)*odd_map(cc)*physical.value({a,bb^1,cc^1}),
                        -I*MP(eta*sign(f+physical.parity(bb)))*physical.value(s));
            for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++){
                MP r=0;double norm=0;for(auto [state,v]:physical.equation(s,m,n))if(v!=MP(0)){MP z=v*physical.value(state);r+=z;norm+=magnitude(z);}
                gward.check(r/MP(std::max(1.,norm)),MP(0));
            }
        }
        struct Term {AuxState aux;int phys;MP c;};
        auto expand=[&](int slot,const Sparse<MP>&expr){
            std::map<AuxState,Sparse<MP>> groups;std::vector<Term>out;
            for(auto [id,v]:expr){auto s=free[slot]->states.at(id);AuxState a;
                for(int bit=63;bit>=0;bit--)if((s.auxiliary>>bit)&1)a.modes.push_back(slot?bit+1:2*bit+1);
                a.ground=s.auxiliary_ground;s.auxiliary=0;s.auxiliary_ground=0;groups[a][free[slot]->intern(s)]+=v;
            }
            for(auto &[a,v]:groups)for(auto &[s,coef]:pbw[slot]->from_fock(v)){
                ScaWord w;for(int n:s.l)w.push_back({0,-2*n});for(int n:s.g)w.push_back({1,-(slot?2:1)*n});
                out.push_back({a,2*words.intern(w)+s.ground,coef*power(t,s.ground)});
            }return out;
        };
        auto vertex=[&](const std::array<Sparse<MP>,3>&s){
            std::array<std::vector<Term>,3> terms;for(int j=0;j<3;j++)terms[j]=expand(j,s[j]);MP ans=0;
            for(auto&a:terms[0])for(auto&bb:terms[1])for(auto&cc:terms[2]){
                MP v=auxiliary.value({a.aux,bb.aux,cc.aux});if(v==MP(0))continue;
                ans+=a.c*bb.c*cc.c*MP(sign(physical.parity(bb.phys)*aux_parity(2,cc.aux)))*v*physical.value({a.phys,bb.phys,cc.phys});
            }return ans;
        };
        for(int ns:{0,2})for(int r2:{1,3})for(int r3:{1,3})for(int alpha=0;alpha<2;alpha++)for(int gamma=0;gamma<2;gamma++){
            std::array<Sparse<MP>,3>s{free[0]->primary(ns,0),free[1]->primary(r2,alpha),free[2]->primary(r3,gamma)};
            // Check full descendant transport, not only the primary weight formula.
            for(int copy=0;copy<2;copy++)for(int n:{1,2}){
                auto changed=s;changed[0]=free[0]->apply(4+copy,-n,s[0]);MP lhs=vertex(changed);
                changed=s;changed[2]=free[2]->apply(4+copy,n,s[2]);MP rhs=vertex(changed);
                for(int j=-1;j<=n;j++){
                    changed=s;
                    // FreeField::embedded is a nonzero-mode helper: L0 is
                    // supplied by the exact branch weight, as in production.
                    if(j==0){changed[1]=s[1];for(auto &[id,v]:changed[1])v*=branch_weight(copy,r2,b,P[1]);}
                    else changed[1]=free[1]->apply(4+copy,j,s[1]);
                    rhs+=MP(binomial(n+1,j+1))*vertex(changed);
                }
                if(magnitude(lhs-rhs)>1e-20 && virasoro.failed<12)std::cerr<<"V failure f="<<f<<" eta="<<eta<<" labels="<<ns<<','<<r2<<','<<r3<<" grounds="<<alpha<<','<<gamma<<" copy="<<copy<<" n="<<n<<" lhs="<<json_number(lhs)<<" rhs="<<json_number(rhs)<<'\n';
                virasoro.check(lhs,rhs);
            }
            for(int copy=0;copy<2;copy++)for(int slot:{1,2})for(int n:{1,2}){
                auto changed=s;changed[slot]=free[slot]->apply(4+copy,-n,s[slot]);
                std::array<MP,3>h{branch_weight(copy,ns,b,P[0]),branch_weight(copy,r2,b,P[1]),branch_weight(copy,r3,b,P[2])};
                virasoro.check(vertex(changed),vertex_word(slot,Part{n},h)*vertex(s));
            }
        }
        std::cerr<<"local f="<<f<<" eta="<<eta<<" complete, "<<physical.size()<<" cached components\n";
    }
    // The all-NS forms retain the same BPZ convention, with no square root
    // multiplier in the contour. Check the existing all-NS forms against
    // those untwisted residues as part of the uniform sewing convention.
    Audit ns_gward,ns_fward;
    std::array<std::unique_ptr<ScaModule<MP>>,3> nsmod;
    for(int j=0;j<3;j++)nsmod[j]=std::make_unique<ScaModule<MP>>(words,Q*Q/MP(8)-P[j]*P[j]/MP(2),c,MP(0),false,false);
    level6::NSWard<MP> nsrho(words,{nsmod[0].get(),nsmod[1].get(),nsmod[2].get()});
    for(int a:states[0])for(int bb:states[0])for(int cc:states[0]){
        std::array<int,3>s{a,bb,cc};if(words.words[a/2].level2+words.words[bb/2].level2+words.words[cc/2].level2>4)continue;
        for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++){
            MP res=0;double scale=0;
            auto add=[&](int slot,int mode,MP factor){for(auto [t,v]:nsmod[slot]->act(1,mode,s[slot])){auto changed=s;changed[slot]=t;MP z=factor*v*nsrho.value(changed);res+=z;scale+=magnitude(z);}};
            for(int j=0;j<10;j++){
                MP a=from_rational<MP>(half_binomial(2*n,j));
                add(0,2*j-2*m-2*n+1,-MP(sign(j))*a);
                add(1,2*n+2*j-1,from_rational<MP>(half_binomial(2*m,j)));
                add(2,2*m+2*j-1,MP(sign(words.words[bb/2].parity+n+j))*a);
            }ns_gward.check(res/MP(std::max(1.,scale)),MP(0));
        }
    }
    for(auto a:auxstates[0])for(auto bb:auxstates[0])for(auto cc:auxstates[0]){
        AuxTriple s{a,bb,cc};if(sum(a.modes)+sum(bb.modes)+sum(cc.modes)>4)continue;
        for(int m=-1;m<=1;m++)for(int n=-1;n<=1;n++){
            MP res=0;double scale=0;
            auto add=[&](int slot,int mode,MP factor){auto [t,v]=aux_act(0,mode,s[slot]);auto changed=s;changed[slot]=t;MP z=factor*from_rational<MP>(v)*MP(level6::ns_fermion(changed));res+=z;scale+=magnitude(z);};
            for(int j=0;j<10;j++){
                MP a=from_rational<MP>(half_binomial(2*n,j));
                add(0,2*j-2*m-2*n-1,MP(sign(j))*a);
                add(1,2*n+2*j+1,from_rational<MP>(half_binomial(2*m,j)));
                add(2,2*m+2*j+1,MP(sign(bb.modes.size()+n+j))*a);
            }ns_fward.check(res/MP(std::max(1.,scale)),MP(0));
        }
    }
    std::ofstream out("local_results.json");out<<std::setprecision(17)<<"{\"dps\":40,\"seconds\":"<<seconds()-start<<",\"checks\":{";
    bool comma=false;size_t failed=0;for(auto item:std::vector<std::pair<std::string,Audit>>{{"physical_residue_identities",gward},{"auxiliary_residue_identities",fward},{"physical_vs_production",production_physical},{"auxiliary_vs_production",production_aux},{"double_Virasoro_Ward",virasoro},{"Ramond_odd_ground_identity",cycle},{"all_NS_physical_residues",ns_gward},{"all_NS_auxiliary_residues",ns_fward}}){
        if(comma)out<<',';comma=true;auto&a=item.second;failed+=a.failed;
        out<<'"'<<item.first<<"\":{\"cases\":"<<a.count<<",\"failed\":"<<a.failed<<",\"maximum_scaled_error\":"<<a.maximum<<'}';
        std::cout<<item.first<<": "<<a.count<<" cases; "<<a.failed<<" failed; max "<<a.maximum<<'\n';
    }out<<"},\"failed\":"<<failed<<"}\n";return failed?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
