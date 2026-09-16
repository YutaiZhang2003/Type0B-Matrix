// Directed all-NS Mercedes comparison. Reuse the frozen double-Virasoro engine.
#define main frozen_level6_main
#include "../total_level6_2026-09-16/graph_validate.cpp"
#undef main
#include "ns_recursion.hpp"

int main(int argc,char**argv){try{
    int level=argc>1?std::stoi(argv[1]):6; MP::precision(40);
    Setup setup("theta_ns");setup.name="mercedes";
    setup.graph=Graph(6,{{0,1,2},{0,3,5},{1,4,3},{2,5,4}},{{0,3},{1,6},{2,9},{4,8},{7,11},{10,5}});
    setup.r.assign(6,false);setup.p.insert(setup.p.end(),{parse<S>("19/37"),parse<S>("23/41"),parse<S>("29/43")});
    setup.cases.clear();for(int fm=0;fm<8;fm++)setup.cases.push_back({{__builtin_popcount(unsigned(fm))%2,fm&1,(fm>>1)&1,(fm>>2)&1},{1,1,1,1},{}});
    auto targets=indices(setup,level);std::vector<int>ids{0,1,2,3,4,5,6,7};
    double start=seconds();DoubleVirasoro dv(setup,level);auto hats=dv.compute({},ids);
    double dv_seconds=seconds()-start;std::cerr<<"double Virasoro complete: "<<dv_seconds<<" s\n";
    Sewing fermion(setup,true);Block aux;for(auto k:targets){auto row=fermion.coefficient(k,setup.cases[0]);if(!row.empty())aux[k]=row;}
    double ff_seconds=seconds()-start-dv_seconds;std::cerr<<"fermion complete: "<<ff_seconds<<" s, "<<aux.size()<<" nonzero multidegrees\n";
    std::vector<Block>recovered(8);
    for(int ci=0;ci<8;ci++)for(auto k:targets){Row rem=hats[ci][k];for(auto&[v,row]:aux)if(total(v)&&leq(v,k)){auto it=recovered[ci].find(minus(k,v));if(it!=recovered[ci].end())addstar(rem,it->second,row,setup.graph,S(-1));}if(!rem.empty())recovered[ci].emplace(k,std::move(rem));}
    double inverse_seconds=seconds()-start-dv_seconds-ff_seconds;
    std::ofstream co("coefficients_L"+std::to_string(level)+".jsonl");
    // Save the recovered coefficients before the independent recursion starts.
    for(int ci=0;ci<8;ci++)for(auto&[k,row]:recovered[ci]){co<<"{\"case\":"<<ci<<",\"level2\":[";for(int e=0;e<6;e++){if(e)co<<',';co<<k[e];}co<<"],\"double_virasoro\":";row_json(co,row);co<<"}\n";}co.close();
    S b=parse<S>("7/5"),Q=b+S(1)/b;std::vector<S>h;for(auto p:setup.p)h.push_back(Q*Q/S(8)-p*p/S(2));
    NSRecursion engine(setup.graph,rational<S>(3,2)+S(3)*Q*Q,h,level);
    std::vector<Error>errors(8);std::ofstream cr("c_recursion_L"+std::to_string(level)+".jsonl");int count=0;
    for(auto k:targets){int mask=0;for(int e=0;e<6;e++)mask|=(k[e]%2)<<e;int ci=0;for(int v=1;v<4;v++){int p=0;for(int e:setup.graph.slots[v])p^=(mask>>e)&1;ci|=p<<(v-1);}S value=engine.coefficient(k);Row expected{{mask,value}};
        for(int c=0;c<8;c++)errors[c].check(recovered[c][k],c==ci?expected:Row{},k,6);
        cr<<"{\"case\":"<<ci<<",\"level2\":[";for(int e=0;e<6;e++){if(e)cr<<',';cr<<k[e];}cr<<"],\"c_recursion\":";row_json(cr,expected);cr<<"}\n";
        if(++count%2000==0)std::cerr<<"NS recursion "<<count<<"/"<<targets.size()<<" coefficients\n";
    }
    double ns_seconds=seconds()-start-dv_seconds-ff_seconds-inverse_seconds;std::ofstream out("results_L"+std::to_string(level)+".json");out<<std::setprecision(17)<<"{\"level\":"<<level<<",\"dps\":40,\"multidegrees\":"<<targets.size()<<",\"dv_seconds\":"<<dv_seconds<<",\"fermion_seconds\":"<<ff_seconds<<",\"recovery_seconds\":"<<inverse_seconds<<",\"ns_c_recursion_seconds\":"<<ns_seconds<<",\"total_seconds\":"<<seconds()-start<<",\"branches\":"<<dv.branches<<",\"virasoro_transitions\":"<<dv.ccy_transitions<<",\"cases\":[";
    int failures=0;for(int ci=0;ci<8;ci++){if(ci)out<<',';errors[ci].json(out);failures+=errors[ci].failed;std::cerr<<"case "<<ci<<" error="<<errors[ci].scaled<<" absolute="<<errors[ci].absolute<<"\n";}out<<"],\"failed_coefficients\":"<<failures<<"}\n";return failures?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
