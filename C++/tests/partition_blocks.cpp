#include "scblocks/ns_recursion.hpp"
#include "ramond/pipeline.hpp"
#include "ramond/direct_pbw.hpp"
#include "scblocks/partition_sewing.hpp"
#include <iostream>
using namespace ramond;
int main() {
    try {
        MP::precision(40);double start=seconds(),maximum=0;size_t count=0;
        auto equal=[&](const MP&a,const MP&b) {
            maximum=std::max(maximum,magnitude(a-b)/std::max({1.,magnitude(a),magnitude(b)}));count++;
        };
        Settings settings;settings.level=3;settings.dps=40;settings.b="1.4";
        settings.momenta={"0,0.17282371376357447","0,0.13570429546907137","0,0.14019878891744741"};
        settings.inserted=false;settings.p=0;
        for(int eta:{1,-1})for(int f=0;f<2;f++) {
            settings.eta=eta;settings.f=f;auto F=pipeline<MP>(settings).physical;
            std::array<MP,3> P;for(int e=0;e<3;e++)P[e]=parse<MP>(settings.momenta[e]);
            DirectPBW<MP> pbw(parse<MP>(settings.b),P,0,f,eta,false);
            for(const auto &[k,row]:F) {
                auto direct=pbw.coefficient({k[0],2*k[1],k[3]});
                for(auto signs:scblocks::nonchiral::eta_e)
                    equal(scblocks::nonchiral::evaluate_parity(row,signs),scblocks::nonchiral::evaluate_parity(direct,signs));
            }
        }
        double identity_error=maximum;size_t identity_count=count;maximum=0;count=0;
        scblocks::Setup setup("theta_ns");setup.b=parse<MP>(settings.b);
        for(int j=0;j<3;j++)setup.p[j]=parse<MP>(settings.momenta[j]);
        MP Q=setup.b+MP(1)/setup.b;std::vector<MP>weights;
        for(auto p:setup.p)weights.push_back(Q*Q/MP(8)-p*p/MP(2));
        auto vacuum=scblocks::ns_schottky(setup.graph,setup.domain(3));
        // The three first closed supercurrent links have the paper's Koszul minus.
        for(int i=0;i<3;i++)for(int j=i+1;j<3;j++) {
            scblocks::Key k{};k[i]=k[j]=3;
            require(vacuum.at(k)==Rational(-1),"wrong Schottky leading spin lift");
        }
        scblocks::NSRecursion recursion(setup.graph,rational<MP>(3,2)+MP(3)*Q*Q,weights,vacuum);
        auto coefficients=recursion.coefficients(scblocks::indices(setup,3));
        scblocks::Sewing physical(setup,false);
        for(auto &[k,value]:coefficients) {
            auto row=physical.coefficient(k,setup.cases[scblocks::total(k)%2]);
            equal(value,scblocks::getrow(row,scblocks::parity_mask(k)));
        }
        require(identity_error<1e-25 && maximum<1e-30,"partition block convention check failed");
        std::cout<<std::setprecision(17)<<"{\"dps\":40,\"level\":3,\"R_literal_F_vs_PBW_components\":"<<identity_count
            <<",\"R_literal_F_vs_PBW_maximum_scaled_error\":"<<identity_error
            <<",\"NS_vs_PBW_components\":"<<count<<",\"NS_vs_PBW_maximum_scaled_error\":"<<maximum
            <<",\"seconds\":"<<seconds()-start<<"}\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
