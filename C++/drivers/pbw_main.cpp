#include "ramond/direct_pbw.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
using namespace ramond;

template<class S> void run(int level,int dps,int p,int f,int eta,bool opposite,const std::string &output) {
    double start=seconds();
    S b=parse<S>("7/5"); std::array<S,3> momenta{parse<S>("11/23"),parse<S>("13/29"),parse<S>("17/31")};
    DirectPBW<S> engine(b,momenta,p,f,eta,opposite);
    double setup=seconds()-start;
    std::vector<std::array<int,3>> levels;
    for(int a=0;a<=2*level;a++) for(int l=0;l<=level;l++) for(int d=0;d<=level;d++) levels.push_back({a,2*l,2*d});
    std::sort(levels.begin(),levels.end(),[](auto a,auto b){
        int x=a[0]+a[1]+a[2],y=b[0]+b[1]+b[2]; return x==y?a<b:x<y; });
    std::vector<std::array<S,8>> coefficients; coefficients.reserve(levels.size());
    for(size_t j=0;j<levels.size();j++) {
        coefficients.push_back(engine.coefficient(levels[j]));
        if((j+1)%50==0) std::cerr<<"PBW "<<(opposite?"inserted":"ordinary")<<": "<<j+1<<'/'<<levels.size()<<", "<<seconds()-start<<" s\n";
    }
    double total=seconds()-start; const auto &t=engine.times; const auto &n=engine.counts;
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::filesystem::path path(output); if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(output); require(bool(out),"cannot open PBW output");
    out<<std::setprecision(17)<<"{\"status\":\"computed\",\"implementation\":\"C++17 direct SCA PBW\",\"mode\":\""
       <<(opposite?"inserted":"ordinary")<<"\",\"truncation\":\"per-edge\",\"q_level_cutoffs\":["<<level<<','<<level<<','<<level
       <<"],\"dps\":"<<dps<<",\"precision_bits\":"<<(dps?MP::bits:53)<<",\"b\":\"7/5\",\"momenta\":[\"11/23\",\"13/29\",\"17/31\"],\"p\":"<<p<<",\"f\":"<<f
       <<",\"etas\":["<<eta<<','<<(opposite?-eta:eta)<<"],\"timing_seconds\":{\"setup\":"<<setup<<",\"metadata\":"<<t.metadata
       <<",\"gram_entries\":"<<t.gram_entries<<",\"gram_inverse\":"<<t.gram_inverse<<",\"vertices\":"<<t.vertices<<",\"contractions\":"<<t.contractions<<",\"total\":"<<total
       <<"},\"counts\":{\"monomials\":"<<levels.size()<<",\"parity_slots\":"<<8*levels.size()<<",\"gram_entries\":"<<n.gram_entries<<",\"gram_matrices\":"<<n.gram_matrices
       <<",\"vertex_entries\":"<<n.vertex_entries<<",\"reused_vertex_tensors\":"<<n.reused_vertex_tensors<<",\"contractions\":"<<n.contractions<<",\"contraction_products\":"<<n.contraction_products
       <<",\"ward_entries\":"<<engine.ward_entries()<<",\"ward_hits\":"<<engine.ward_hits()<<",\"ward_action_terms\":"<<engine.ward_terms()<<",\"ward_buckets\":"<<engine.ward_buckets()
       <<",\"interned_words\":"<<engine.word_count()<<",\"mode_action_entries\":"<<engine.action_entries()
       <<"},\"memory\":{\"peak_rss_native_units\":"<<usage.ru_maxrss<<",\"peak_rss_unit\":\""
#ifdef __APPLE__
       <<"bytes"
#else
       <<"KiB"
#endif
       <<"\",\"scalar_object_bytes\":"<<sizeof(S)<<",\"ward_key_bytes\":"<<sizeof(std::array<int,3>)
       <<",\"ward_entry_pair_bytes\":"<<ScaWard<S>::entry_pair_bytes()<<"},\"coefficients\":[";
    for(size_t i=0;i<levels.size();i++) {
        auto l=levels[i]; if(i) out<<',';
        out<<"{\"exponents\":["<<l[0]<<','<<l[1]/2<<','<<l[1]/2<<','<<l[2]<<"],\"values\":[";
        for(int j=0;j<8;j++) { if(j) out<<','; out<<json_number(coefficients[i][j]); }
        out<<"]}";
    }
    out<<"]}\n"; out.close(); require(bool(out),"failed PBW output write");
    std::cerr<<"PBW total: "<<total<<" s; vertices "<<t.vertices<<" s; contractions "<<t.contractions<<" s\n";
}

int main(int argc,char **argv) {
    try {
        int level=5,dps=40,p=0,f=0,eta=1; bool opposite=false; std::string output;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"missing PBW option value"); std::string key=argv[i],value=argv[i+1];
            if(key=="--level") level=std::stoi(value);
            else if(key=="--dps") dps=std::stoi(value);
            else if(key=="--p") p=std::stoi(value);
            else if(key=="--f") f=std::stoi(value);
            else if(key=="--eta") eta=std::stoi(value);
            else if(key=="--json") output=value;
            else if(key=="--mode") { require(value=="ordinary"||value=="inserted","invalid PBW mode"); opposite=value=="inserted"; }
            else throw std::invalid_argument("unknown PBW option: "+key);
        }
        require(level>=0&&!output.empty(),"need nonnegative --level and --json");
        require((p==0||p==1)&&(f==0||f==1)&&std::abs(eta)==1,"invalid physical PBW signs");
        if(dps) { MP::precision(dps); run<MP>(level,dps,p,f,eta,opposite,output); }
        else run<Machine>(level,dps,p,f,eta,opposite,output);
        return 0;
    } catch(const std::exception &e) { std::cerr<<"pbw: "<<e.what()<<'\n'; return 1; }
}
