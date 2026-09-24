#include "scblocks/partition_sewing.hpp"
#include "ramond/fermion.hpp"
#include <chrono>
#include <iostream>
using namespace ramond;
namespace nc = scblocks::nonchiral;

// Direct audit of the paper's BPZ sewing in its original PBW basis.
// See PARTITION_NO_M_AUDIT.md.
int mod4(int n) { return (n % 4 + 4) % 4; }
template<class S> S ipow(int n) {
    switch (mod4(n)) { case 0:return S(1); case 1:return S(Machine(0,1));
        case 2:return S(-1); default:return S(Machine(0,-1)); }
}

template<class S> void check(double tolerance) {
    S I(Machine(0,1)), rt=root(S(2)), u=(S(1)-I)/rt, ub=(S(1)+I)/rt;
    double error=0;
    auto equal=[&](S x,S y) { error=std::max(error,magnitude(x-y)/std::max(1.,magnitude(y))); };
    // B_geo(v,w)=i^|v| B_alg(v,w) implements graded adjointness
    // B_geo(Gv,w)=(-1)^|v| B_geo(v,i G_-r w), for either parity.
    for(int p=0;p<2;p++) equal(ipow<S>(p^1),S(sign(p))*I*ipow<S>(p));
    // Paper Ramond ground action and its two metrics.
    S beta=I*S(2)/S(5), Gplus=I*beta*u, Gminus=I*beta*ub;
    equal(Gplus*I,Gminus);                 // algebraic metric diag(1,i)
    equal(-Gplus,I*Gminus);               // geometric metric diag(1,-1)

    // Full NS state sewing: the original K, graded product metric,
    // geometric inverse-Gram phases, and both infinity-slot factors.
    // Geometry-reference lifts differ by eta_1=-1 in this BPZ branch.
    // Check the exponent exactly modulo four, before numerical evaluation.
    int ns_cases=0;
    for(int e=0;e<8;e++) for(int t=0;t<8;t++) {
        int ne=__builtin_popcount(unsigned(e)),nt=__builtin_popcount(unsigned(t));
        if(ne%2!=nt%2)continue;
        int phase=2*(nc::K(e^t)+__builtin_popcount(unsigned(e&t)))-ne+nt
                  +2*((e&1)+(t&1)) +2*((e&1)+(t&1));
        require(mod4(phase)==2*(ne%2),"geometric NS state sewing failed");
        ns_cases++;
    }
    int relabelings=0;
    for(int e=0;e<8;e++) {
        std::array<int,3> s{sign(e&1),sign((e>>1)&1),sign((e>>2)&1)};
        if(nc::character(s,3)==-1 && nc::character(s,5)==-1 && nc::character(s,6)==-1)relabelings++;
    }
    require(relabelings==0,"three pair-loop signs unexpectedly are a character");

    // Physical w^+,w^- kets exactly as written in the paper, in the
    // ordered product basis (++,+-,-+,--); no rephasing or basis map.
    std::array<std::array<S,2>,4> E{{{S(1)/rt,0},{0,S(1)/rt},{0,S(1)/rt},{-I/rt,0}}};
    std::array<S,4> algebraic_metric{1,-I,I,-1};
    std::array<S,4> bpz_metric{1,-1,-1,-1};
    for(int e=0;e<2;e++)for(int f=0;f<2;f++) {
        S algebraic_pair=0,bpz_pair=0;
        for(int j=0;j<4;j++) {
            algebraic_pair+=E[j][e]*algebraic_metric[j]*E[j][f];
            bpz_pair+=E[j][e]*bpz_metric[j]*E[j][f];
        }
        equal(algebraic_pair,S(e==0 && f==0));
        equal(bpz_pair,S(e==f?sign(e):0));
    }
    require(error<tolerance,"paper BPZ convention audit failed");
    std::cout<<"{\"NS_exact_parity_cases\":"<<ns_cases<<",\"R_ground_metric_cases\":4"
             <<",\"edge_sign_relabelings\":"<<relabelings
             <<",\"algebraic_R_restricted_metric\":[1,0],\"paper_BPZ_R_restricted_metric\":[1,-1]"
             <<",\"maximum_scaled_error\":"<<error<<"}\n";
}

int main() {
    try {
        auto start=std::chrono::steady_clock::now();
        MP::precision(40);check<Machine>(1e-13);check<MP>(1e-38);
        // Independent R free-fermion geometry check input. These coefficients
        // retain the actual Ward vertex and both ground states. The geometric
        // BPZ conversion is performed before evaluating any q or theta function.
        auto F=fermion_series(8,false);std::map<std::array<int,3>,std::array<Rational,2>> coefficients;
        for(const auto& [level,row]:F)for(int e=0;e<8;e++) {
            int a=e&1,b=(e>>1)&1,ne=__builtin_popcount(unsigned(e));
            require(ne%2==0 || row[e]==0,"fermion vertex must be even");
            // i^sum epsilon from the Li-Z inverse metric; (-1)^a from
            // the two infinity bra factors; (-1)^b from the R tube lift.
            int phase=ne+2*a+2*b;
            if(row[e]==0)continue;
            require(mod4(phase)%2==0,"nonreal even fermion BPZ coefficient");
            for(int spin=0;spin<2;spin++)
                coefficients[{level[0],level[1],level[3]/2}][spin]
                    +=sign(mod4(phase)/2+spin*a)*row[e];
        }
        std::cout<<"{\"R_free_fermion_level\":8,\"coefficients\":[";
        bool first=true;
        for(const auto& [level,row]:coefficients) {
            if(!first)std::cout<<',';first=false;
            std::cout<<'['<<level[0]<<','<<level[1]<<','<<level[2]<<",\""<<row[0]<<"\",\""<<row[1]<<"\"]";
        }
        std::cout<<"],\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
