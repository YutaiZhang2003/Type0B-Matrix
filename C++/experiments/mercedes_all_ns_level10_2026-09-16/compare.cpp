#include "dv_engine.hpp"
#include "ns_recursion.hpp"

Setup all_ns_setup(){
    Setup s("theta_ns");s.name="mercedes";s.graph=Graph(6,{{0,1,2},{0,3,5},{1,4,3},{2,5,4}},{{0,3},{1,6},{2,9},{4,8},{7,11},{10,5}});
    s.r.assign(6,false);s.p.insert(s.p.end(),{parse<S>("19/37"),parse<S>("23/41"),parse<S>("29/43")});s.cases.clear();
    for(int fm=0;fm<8;fm++)s.cases.push_back({{__builtin_popcount(unsigned(fm))%2,fm&1,(fm>>1)&1,(fm>>2)&1},{1,1,1,1},{}});return s;
}
int case_for(const Graph&g,int mask){int ci=0;for(int v=1;v<4;v++){int p=0;for(int e:g.slots[v])p^=(mask>>e)&1;ci|=p<<(v-1);}return ci;}
void scalar_json(std::ostream&o,int ci,const Key&k,const S&x,const char*field){
    o<<"{\"case\":"<<ci<<",\"level2\":[";for(int e=0;e<6;e++){if(e)o<<',';o<<k[e];}
    o<<"],\""<<field<<"\":[["<<parity_mask(k)<<",\""<<decimal(mpc_realref(x.data()))<<"\",\""<<decimal(mpc_imagref(x.data()))<<"\"]]}\n";
}
int main(int argc,char**argv){try{
    require(argc>=2,"usage: compare dv|ns [TOTAL_LEVEL=10]");std::string mode=argv[1];int level=argc>2?std::stoi(argv[2]):10;MP::precision(40);
    Setup s=all_ns_setup();auto targets=indices(s,level);std::cerr<<"mode="<<mode<<" total level="<<level<<" multidegrees="<<targets.size()<<" dps=40\n";
    double start=seconds();std::string suffix="_L"+std::to_string(level);std::ofstream timing("timing_"+mode+suffix+".json");timing<<std::setprecision(17);
    if(mode=="dv"){
        DoubleVirasoro dv(s,level);auto hats=dv.compute({},{0,1,2,3,4,5,6,7});double dvtime=seconds()-start;
        std::cerr<<"double Virasoro enlarged blocks complete: "<<dvtime<<" s\n";
        Sewing ff(s,true);std::vector<std::pair<Key,S>>aux;
        for(const auto&k:targets){Row r=ff.coefficient(k,s.cases[0]);if(!total(k)){require(getrow(r,0)==S(1),"auxiliary constant is not one");continue;}
            for(const auto&[mask,x]:r){require(mask==parity_mask(k),"auxiliary parity mismatch");if(x!=S(0))aux.emplace_back(k,x);}}
        double fft=seconds()-start-dvtime;std::cerr<<"auxiliary complete: "<<fft<<" s, "<<aux.size()<<" nonconstant coefficients\n";
        double assembly_start=seconds();Poly<S>remainder;for(int ci=0;ci<8;ci++)for(const auto&[k,row]:hats[ci])for(const auto&[mask,x]:row){require(mask==parity_mask(k)&&ci==case_for(s.graph,mask),"enlarged sector mismatch");remainder[k]+=x;}
        hats.clear();int kernels[64][64];for(int a=0;a<64;a++)for(int b=0;b<64;b++)kernels[a][b]=s.graph.kernel(a,b);
        double assembly_seconds=seconds()-assembly_start;double recoverstart=seconds();std::ofstream out("coefficients"+suffix+".jsonl");int n=0;
        for(const auto&k:targets){S value=remainder[k];int km=parity_mask(k);scalar_json(out,case_for(s.graph,km),k,value,"double_virasoro");
            if(value!=S(0))for(const auto&[a,f]:aux){if(total(k)+total(a)>2*level)break;remainder[plus(k,a)]-=S(kernels[km][parity_mask(a)])*value*f;}
            if(++n%20000==0)std::cerr<<"recovered "<<n<<"/"<<targets.size()<<"\n";
        }
        double rt=seconds()-recoverstart;timing<<"{\"level\":"<<level<<",\"dps\":40,\"multidegrees\":"<<targets.size()<<",\"dv_seconds\":"<<dvtime<<",\"fermion_seconds\":"<<fft<<",\"assembly_seconds\":"<<assembly_seconds<<",\"recovery_and_output_seconds\":"<<rt<<",\"total_seconds\":"<<seconds()-start<<",\"branches\":"<<dv.branches<<",\"ccy_transitions\":"<<dv.ccy_transitions<<"}\n";
    }else if(mode=="ns"){
        S b=parse<S>("7/5"),Q=b+S(1)/b;std::vector<S>h;for(const auto&p:s.p)h.push_back(Q*Q/S(8)-p*p/S(2));
        NSRecursion engine(s.graph,rational<S>(3,2)+S(3)*Q*Q,h,level);double seedtime=seconds()-start;
        std::ofstream out("c_recursion"+suffix+".jsonl");int n=0;
        for(const auto&k:targets){S value=engine.coefficient(k);scalar_json(out,case_for(s.graph,parity_mask(k)),k,value,"c_recursion");if(++n%5000==0)std::cerr<<"NS recursion "<<n<<"/"<<targets.size()<<" elapsed="<<seconds()-start<<" s\n";}
        timing<<"{\"level\":"<<level<<",\"dps\":40,\"multidegrees\":"<<targets.size()<<",\"schottky_seed_seconds\":"<<seedtime<<",\"recursion_and_output_seconds\":"<<seconds()-start-seedtime<<",\"total_seconds\":"<<seconds()-start<<"}\n";
    }else throw std::runtime_error("mode must be dv or ns");
    std::cerr<<mode<<" complete in "<<seconds()-start<<" s\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}}
