#include "scblocks/partition_sewing.hpp"
#include "scblocks/json_input.hpp"
#include "ramond/storage.hpp"
#include <iostream>
using namespace ramond;
namespace nc=scblocks::nonchiral;

template<class S> double check(double tolerance) {
    S I(Machine(0,1)),rt=root(S(2));
    // The paper's w^+,w^- embedding in (++,+-,-+,--), unchanged.
    std::array<std::array<S,2>,4> E{{{S(1)/rt,0},{0,S(1)/rt},{0,S(1)/rt},{-I/rt,0}}};
    double error=0;
    auto equal=[&](S a,S b) {error=std::max(error,magnitude(a-b)/std::max(1.,magnitude(b)));};
    for(int e=0;e<2;e++)for(int f=0;f<2;f++) {
        S pair=0;for(int j=0;j<4;j++)pair+=E[j][e]*S(j? -1:1)*E[j][f];
        equal(pair,S(e==f?sign(e):0));
    }
    // The full-state graded sewing reduces to the two chiral K signs plus a.
    auto K=[](int e) {return ((e&1)*((e>>1)&1)+(e&1)*((e>>2)&1)+((e>>1)&1)*((e>>2)&1))%2;};
    int graded_cases=0;
    for(int e=0;e<8;e++)for(int t=0;t<8;t++)if(__builtin_popcount(unsigned(e))%2==__builtin_popcount(unsigned(t))%2) {
        int a=__builtin_popcount(unsigned(e))%2;
        equal(S(sign(K(e^t)+__builtin_popcount(unsigned(e&t)))),S(sign(K(e)+K(t)+a)));
        graded_cases++;
    }
    // In the oscillator recursion, each pair of fermionic edges has K=1. The
    // product of the three relative signs is -1 for every paper eta_e.
    // This excludes the previously assumed linear map to the charge-lattice
    // theta signs at fixed q; it does not refute the SCblock factorization.
    for(int spin=0;spin<4;spin++) {
        int loop_product=1;
        for(int e:{3,5,6})loop_product*=sign(K(e))*nc::character(nc::eta_e[spin],e);
        require(loop_product==-1,"literal paper F changed its level-one signs");
    }
    // BPZ sewing is part of the block and gives the Ramond trace 2 in
    // the NS identity limit at either fixed set of physical tube signs.
    std::array<S,2> constants{S(1),S(0)};
    std::array<S,8> even{},odd{};
    even[0]=S(1);even[6]=S(1);
    odd[2]=I;odd[4]=-I;
    for(auto signs:{std::array<int,3>{1,-1,1},std::array<int,3>{-1,-1,1}}) {
        S f0=0,f1=0;
        for(int epsilon=0;epsilon<8;epsilon++) {
            S phase=theta_bpz_weight<S>(epsilon)*S(nc::character(signs,epsilon));
            f0+=phase*even[epsilon];f1+=phase*odd[epsilon];
        }
        equal(f0,S(2));equal(f1,-f0);
        std::array<S,2> amplitude{f0,S(0)},anti{conjugate(f0),S(0)};
        equal(nc::fixed_R_partition(amplitude,anti,constants,constants,S(1)),S(2));
    }
    for(int epsilon=0;epsilon<8;epsilon++) {
        S phase=theta_bpz_weight<S>(epsilon);
        equal(phase*conjugate(phase),S(1));
    }

    require(error<tolerance,"partition sewing validation failed");
    std::cout<<"{\"paper_R_ground_metric_cases\":4,\"NS_graded_sign_cases\":"<<graded_cases
             <<",\"NS_linear_theta_map_obstruction_cases\":4,\"maximum_scaled_error\":"<<error<<"}"<<std::endl;
    return error;
}
int main() {
    try {
        MP::precision(40);check<Machine>(1e-13);check<MP>(1e-38);
        auto x=scblocks::Json::parse("{\"x\":1.234567890123456789012345678901234567890e-40,\"s\":\"a\\nb\"}");
        require(x.at("x").scalar()=="1.234567890123456789012345678901234567890e-40","decimal input lost precision");
        for(auto bad:{"{\"x\":1,\"x\":2}","[1,]","01","1e","{\"x\":NaN}"}) {
            bool rejected=false;try{scblocks::Json::parse(bad);}catch(const std::exception&){rejected=true;}
            require(rejected,"invalid JSON accepted");
        }
    std::cout<<"{\"NS_literal_matches_assumed_linear_theta_map\":false}\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
