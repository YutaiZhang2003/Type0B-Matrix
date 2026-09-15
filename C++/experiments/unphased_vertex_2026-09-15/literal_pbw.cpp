#include "ramond/pipeline.hpp"
#include "ramond/literal_enlarged.hpp"
#include <fstream>
using namespace ramond;
int main(int argc,char **argv) {
    try {
        require(argc==5,"usage: literal_pbw LEVEL F ordinary|inserted OUTPUT");
        Settings s; s.level=std::stoi(argv[1]); s.f=std::stoi(argv[2]); s.dps=40;
        s.inserted=std::string(argv[3])=="inserted"; s.box=false; s.record_sector=true;
        require(s.level>=0 && s.level<=3 && s.f==0,"this directed probe is limited to level 0..3, f=0");
        MP::precision(s.dps); MP b=parse<MP>(s.b);
        std::array<MP,3> p; for(int j=0;j<3;j++) p[j]=parse<MP>(s.momenta[j]);
        double start=seconds();
        LiteralEnlargedPBW<MP> engine(b,p,s.p,s.f,s.eta,s.inserted);
        Result<MP> r;
        for(auto k:physical_indices(2*s.level)) r.numerator[k]=engine.coefficient({k[0],2*k[1],k[3]});
        r.timing.assembly=seconds()-start;
        double tick=seconds(); r.auxiliary=numerical_fermion(s.level,s.inserted,b+MP(1)/b);
        r.timing.auxiliary=seconds()-tick; tick=seconds();
        r.physical=recover(r.numerator,r.auxiliary,2*s.level,s.inserted,true,r.sector_residual);
        r.timing.division=seconds()-tick; r.timing.total=seconds()-start;
        std::ostringstream encoded; encode_result(encoded,s,r);
        std::string data=encoded.str();
        auto replace=[&](const std::string &from,const std::string &to) {
            auto pos=data.find(from); require(pos!=std::string::npos,"missing JSON field"); data.replace(pos,from.size(),to);
        };
        replace("\"branching_method\":\"stored_recursion\"","\"branching_method\":\"direct_literal_PBW\"");
        replace("\"reduced_numerator\":","\"full_numerator\":");
        std::ofstream out(argv[4]); require(bool(out),"cannot open output"); out<<data;
        std::cerr<<"literal enlarged PBW total: "<<r.timing.total<<" s\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
