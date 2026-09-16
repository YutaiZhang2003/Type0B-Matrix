// External-state BPZ projection: the final coefficients are overlap/Gram,
// never a fit of the input state against the branching basis.
#include "ramond/free_field.hpp"
#include "ramond/direct_pbw.hpp"
#include <fstream>
#include <iostream>
using namespace ramond;
struct TensorPBW { int physical; uint64_t aux; int ground; Sparse<MP> vector; };
struct Branch { int n4,parity,copy; MP gram; Sparse<MP> vector; std::vector<MP> coordinates; };
struct Case { std::string sector;int twice_level,dimension,branches,gram_checks;double gram_error,reconstruction_error,coordinate_error; };

Case check(bool r,int twice_level) {
    MP b=parse<MP>("7/5"),p=parse<MP>("13/29"),q=b+MP(1)/b;
    MP c=rational<MP>(3,2)+MP(3)*q*q,h=q*q/MP(8)-p*p/MP(2)+(r?rational<MP>(1,16):MP(0));
    FreeField<MP> module(r,b,p);ScaWords<MP> words;ScaModule<MP> physical(words,h,c,p/root(MP(2)),r,true);
    int level=r?twice_level/2:twice_level;
    std::vector<TensorPBW> tensor;
    for(int physical_level=0;physical_level<=level;physical_level++) {
        auto states=module.pbw_basis(physical_level);
        for(const auto &auxiliary:partitions(level-physical_level,true,!r))
        for(int ground=0;ground<(r?2:1);ground++)for(const auto &[pbw,vector]:states) {
            uint64_t mask=0;for(int n:auxiliary)mask|=UINT64_C(1)<<(r?n-1:n/2);
            Sparse<MP> complete;
            for(const auto &[id,v]:vector) {
                State state=module.states[id];state.auxiliary=mask;state.auxiliary_ground=ground;
                complete[module.intern(state)]+=v;
            }
            ScaWord word;for(int n:pbw.l)word.push_back({0,-2*n});for(int n:pbw.g)word.push_back({1,r?-2*n:-n});
            tensor.push_back({2*words.intern(word)+pbw.ground,mask,ground,std::move(complete)});
        }
    }
    size_t dimension=tensor.size();std::vector<MP> gram(dimension*dimension);
    for(size_t a=0;a<dimension;a++)for(size_t bindex=0;bindex<dimension;bindex++)
        if(tensor[a].aux==tensor[bindex].aux&&tensor[a].ground==tensor[bindex].ground)
            gram[a*dimension+bindex]=MP(sign(__builtin_popcountll(tensor[a].aux)+tensor[a].ground))
                *physical.inner(tensor[a].physical,tensor[bindex].physical);
    std::vector<Branch> branches;
    auto add=[&](int n4,int parity,int copy,Sparse<MP> vector) {
        MP primary_norm=r?r_norm(n4,parity,b,p):ns_norm(n4,b,p);
        MP rootnorm=root(primary_norm);
        for(auto &[id,value]:vector)value/=rootnorm;
        MP descendant_norm=copy<0?MP(1):MP(2)*branch_weight(copy,n4,b,p);
        branches.push_back({n4,parity,copy,descendant_norm,std::move(vector),{}});
    };
    if(r) {
        for(int n4:{-3,-1,1,3})for(int parity=0;parity<2;parity++) {
            int shift=(n4*n4-1)/4;
            if(shift>twice_level)continue;
            auto primary=module.primary(n4,parity);
            if(shift==twice_level)add(n4,parity,-1,primary);
            else if(shift+2==twice_level)for(int copy=0;copy<2;copy++)add(n4,parity,copy,module.apply(copy+4,-1,primary));
        }
    } else if(twice_level==0) add(0,0,-1,module.primary(0));
    else if(twice_level==1) {
        add(2,0,-1,module.primary(2));
        auto negative=module.apply(1,-1,Sparse<MP>{{0,MP(1)}});
        State aux{};aux.auxiliary=1;negative[module.intern(aux)]+=q/MP(2)-p;
        add(-2,0,-1,std::move(negative));
    } else if(twice_level==2) {
        auto primary=module.primary(0);
        for(int copy=0;copy<2;copy++)add(0,0,copy,module.apply(copy+4,-1,primary));
    }
    require(branches.size()==dimension,"incomplete branch basis");
    double coordinate_error=0;
    std::vector<Sparse<MP>> columns;for(const auto &state:tensor)columns.push_back(state.vector);
    for(auto &branch:branches) {
        Fit fit;branch.coordinates=span_solve(branch.vector,columns,fit);
        coordinate_error=std::max(coordinate_error,fit.relative_residual);
    }
    auto inner=[&](const std::vector<MP> &a,const std::vector<MP> &bb) {
        MP result(0);for(size_t i=0;i<dimension;i++)for(size_t j=0;j<dimension;j++)result+=a[i]*gram[i*dimension+j]*bb[j];return result;
    };
    double gram_error=0,reconstruction_error=0;
    for(size_t a=0;a<branches.size();a++)for(size_t bb=0;bb<branches.size();bb++) {
        MP observed=inner(branches[a].coordinates,branches[bb].coordinates),expected=a==bb?branches[a].gram:MP(0);
        gram_error=std::max(gram_error,magnitude(observed-expected)/std::max({1.,magnitude(observed),magnitude(expected)}));
    }
    for(size_t source=0;source<dimension;source++) {
        Sparse<MP> reconstructed;
        std::vector<MP> unit(dimension);unit[source]=MP(1);
        for(const auto &branch:branches) {
            // Normalized double-Virasoro basis, contracted with its independently
            // known inverse Gram: 1 on a primary, 1/(2h) on L_-1 descendants.
            MP coefficient=inner(branch.coordinates,unit)/branch.gram;
            for(const auto &[id,value]:branch.vector)reconstructed[id]+=coefficient*value;
        }
        std::set<uint32_t> keys;for(const auto &[id,v]:reconstructed)keys.insert(id);for(const auto &[id,v]:tensor[source].vector)keys.insert(id);
        for(auto id:keys) {
            MP a=get(reconstructed,id),bb=get(tensor[source].vector,id);
            reconstruction_error=std::max(reconstruction_error,magnitude(a-bb)/std::max({1.,magnitude(a),magnitude(bb)}));
        }
    }
    return {r?"R":"NS",twice_level,int(dimension),int(branches.size()),int(branches.size()*branches.size()),gram_error,reconstruction_error,coordinate_error};
}
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: external_state OUTPUT");MP::precision(40);
        std::vector<Case> results;for(int level:{0,1,2})results.push_back(check(false,level));for(int level:{0,2})results.push_back(check(true,level));
        bool passed=true;std::ofstream out(argv[1]);out<<std::setprecision(17)<<"{\"dps\":40,\"method\":\"BPZ overlaps evaluated in physical PBW times auxiliary Gram, followed by normalized double-Virasoro inverse Gram; no final branching-basis fit\",\"cases\":[";
        for(size_t j=0;j<results.size();j++) {
            const auto &r=results[j];if(j)out<<',';out<<"{\"sector\":\""<<r.sector<<"\",\"twice_level\":"<<r.twice_level<<",\"states\":"<<r.dimension<<",\"branches\":"<<r.branches<<",\"Gram_entries\":"<<r.gram_checks<<",\"max_Gram_error\":"<<r.gram_error<<",\"max_reconstruction_error\":"<<r.reconstruction_error<<",\"PBW_coordinate_conversion_error\":"<<r.coordinate_error<<'}';
            passed=passed&&r.gram_error<1e-25&&r.reconstruction_error<1e-25;
            std::cout<<r.sector<<" level="<<r.twice_level/2.<<" dim="<<r.dimension<<" Gram="<<r.gram_error<<" reconstruction="<<r.reconstruction_error<<'\n';
        }
        out<<"],\"passed\":"<<(passed?"true":"false")<<"}\n";return passed?0:1;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
