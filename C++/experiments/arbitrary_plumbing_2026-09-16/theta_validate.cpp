// Directed genus-two theta graph validation. No production code is modified.
#include "ramond/pipeline.hpp"
#include "ramond/literal_enlarged.hpp"
#include "theta_literal_split.hpp"
#include "theta_full_split_dv.hpp"
#include <fstream>
using namespace ramond;

template<class S> ParitySeries<S> restore_vacuum(const ParitySeries<S> &reduced,int level,bool split=false) {
    auto vacuum=schottky_vacuum(SeriesDomain(level));
    auto squared=qmultiply(vacuum,vacuum,SeriesDomain(level));
    ParitySeries<S> full;
    for(const auto &[k,exact]:squared) {
        Index shift{2*k[0],k[1],k[1],2*k[2]};
        S scalar=from_rational<S>(exact);
        for(const auto &[n,row]:reduced) if(degree(n+shift)<=2*level && (split||(n+shift)[1]==(n+shift)[2]))
            for(int i=0;i<8;i++) full[n+shift][i]+=scalar*row[i];
    }
    return full;
}

int main(int argc,char **argv) {
    try {
        require(argc==6||argc==7,"usage: theta_validate LEVEL F ETA ordinary|inserted OUTPUT [NS_PRIMARY_PARITY]");
        Settings s; s.level=std::stoi(argv[1]);s.f=std::stoi(argv[2]);s.eta=std::stoi(argv[3]);
        s.inserted=std::string(argv[4])=="inserted";s.dps=40;s.record_sector=true;
        if(argc==7)s.p=std::stoi(argv[6]);
        require(s.level>=0&&s.level<=3,"directed scope is total level 0..3");
        require((s.f==0||s.f==1)&&(s.p==0||s.p==1)&&std::abs(s.eta)==1,"invalid signs");
        MP::precision(s.dps);
        unphased_vertex=true;apply_vertex_transport=true;keep_ns_branch_sign=true;
        auto dv=pipeline<MP>(s);
        auto full=restore_vacuum(dv.numerator,s.level);
        MP b=parse<MP>(s.b);std::array<MP,3> p;
        for(int j=0;j<3;j++)p[j]=parse<MP>(s.momenta[j]);
        DirectPBW<MP> physical(b,p,s.p,s.f,s.eta,s.inserted);
        LiteralEnlargedPBW<MP> enlarged(b,p,s.p,s.f,s.eta,s.inserted);
        ParitySeries<MP> direct,raw;
        double start=seconds();
        for(auto k:physical_indices(2*s.level))
            direct[k]=physical.coefficient({k[0],2*k[1],k[3]});
        double physical_seconds=seconds()-start;start=seconds();
        for(auto k:physical_indices(2*s.level))
            raw[k]=enlarged.coefficient({k[0],2*k[1],k[3]});
        double enlarged_seconds=seconds()-start;
        auto split=numerical_fermion(s.level,s.inserted,b+MP(1)/b,true);
        ParitySeries<MP> split_raw,opposite_uninserted,split_dv;
        if(s.inserted) {
            split_dv=restore_vacuum(theta_full_split_dv<MP>(s),s.level,true);
            ThetaSplitPBW<MP> split_engine(b,p,s.p,s.f,s.eta,true);
            for(int a=0;a<=2*s.level;a++)
            for(int l=0;l<=2*s.level-a;l++)
            for(int r=0;r<=2*s.level-a-l;r++)
            for(int d=0;d<=2*s.level-a-l-r;d+=2)
                split_raw[{a,l,r,d}]=split_engine.coefficient({a,l,r,d});
            for(auto k:physical_indices(2*s.level))
                opposite_uninserted[k]=split_engine.coefficient(k,false);
        }
        double raw_residual=0;
        auto raw_recovered=recover(raw,dv.auxiliary,2*s.level,s.inserted,true,raw_residual);
        std::ostringstream encoded;encode_result(encoded,s,dv);
        std::string base=encoded.str();auto end=base.find_last_of('}');require(end!=std::string::npos,"JSON encoding failed");
        std::ofstream out(argv[5]);require(bool(out),"cannot open output");out<<base.substr(0,end);
        out<<",\"dv_full_numerator\":";encode_series(out,full);
        out<<",\"direct_physical_pbw\":";encode_series(out,direct);
        out<<",\"direct_enlarged_pbw\":";encode_series(out,raw);
        out<<",\"direct_enlarged_recovery\":";encode_series(out,raw_recovered);
        out<<",\"full_split_auxiliary\":";encode_series(out,split);
        out<<",\"direct_split_enlarged_pbw\":";encode_series(out,split_raw);
        out<<",\"full_split_dv_numerator\":";encode_series(out,split_dv);
        out<<",\"opposite_uninserted_enlarged_pbw\":";encode_series(out,opposite_uninserted);
        out<<",\"direct_timing_seconds\":{\"physical\":"<<physical_seconds<<",\"enlarged\":"<<enlarged_seconds
           <<"},\"direct_recovery_sector_residual\":"<<raw_residual<<"}\n";
        require(bool(out),"output write failed");
        std::cerr<<"physical PBW: "<<physical_seconds<<" s; enlarged PBW: "<<enlarged_seconds<<" s\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
