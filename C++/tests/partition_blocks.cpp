#include "scblocks/ns_recursion.hpp"
#include "ramond/pipeline.hpp"
#include "ramond/direct_pbw.hpp"
#include "scblocks/partition_sewing.hpp"
#include <iostream>
using namespace ramond;
int main() {
    try {
        MP::precision(40);double start=seconds(),maximum=0,native_maximum=0,ns_native_maximum=0;
        size_t count=0,native_count=0,ns_native_count=0,convolution_count=0;
        double convolution_maximum=0;
        auto equal=[&](const MP&a,const MP&b) {
            maximum=std::max(maximum,magnitude(a-b)/std::max({1.,magnitude(a),magnitude(b)}));count++;
        };
        auto equal_native=[&](const MP&a,const MP&b) {
            native_maximum=std::max(native_maximum,magnitude(a-b)/std::max({1.,magnitude(a),magnitude(b)}));
            native_count++;
        };
        Settings settings;settings.level=3;settings.dps=40;settings.b="1.4";
        settings.momenta={"0,0.17282371376357447","0,0.13570429546907137","0,0.14019878891744741"};
        settings.p=0;
        for(int primary_parity=0;primary_parity<2;primary_parity++)
        for(bool inserted:{false,true})for(int eta:{1,-1})for(int f=0;f<2;f++) {
            settings.p=primary_parity;
            settings.inserted=inserted;settings.eta=eta;settings.f=f;
            auto result=pipeline<MP>(settings);
            Settings native_settings=settings;native_settings.native_bpz=true;
            auto native_result=pipeline<MP>(native_settings);
            auto &F=result.physical;
            std::array<MP,3> P;for(int e=0;e<3;e++)P[e]=parse<MP>(settings.momenta[e]);
            DirectPBW<MP> pbw(parse<MP>(settings.b),P,primary_parity,f,eta,inserted);
            DirectPBW<MP> native(parse<MP>(settings.b),P,primary_parity,f,eta,inserted,true);
            for(const auto &[k,row]:F) {
                auto direct=pbw.coefficient({k[0],2*k[1],k[3]});
                auto intrinsic=native.coefficient({k[0],2*k[1],k[3]});
                for(int epsilon=0;epsilon<8;epsilon++) {
                    equal_native(direct[epsilon],intrinsic[epsilon]);
                    equal_native(native_result.physical.at(k)[epsilon],intrinsic[epsilon]);
                }
                for(auto signs:scblocks::nonchiral::eta_e)
                    equal(scblocks::nonchiral::evaluate_parity(row,signs),scblocks::nonchiral::evaluate_parity(direct,signs));
            }
            // Check the BPZ-native diagonal convolution, including the
            // additional tensor-metric sign (-1)^(physical . auxiliary).
            for(const auto &[k,hat]:native_result.numerator) {
                if(k[1]!=k[2])continue;
                for(int total=0;total<8;total++) {
                    MP rhs=0;
                    for(const auto &[pk,pv]:native_result.physical)
                        for(const auto &[ak,av]:native_result.auxiliary)
                        if(pk+ak==k)for(int physical=0;physical<8;physical++) {
                            int auxiliary=physical^total;
                            if(pv[physical]==MP(0) || av[auxiliary]==MP(0))continue;
                            int dot=__builtin_popcount(unsigned(physical&auxiliary))%2;
                            rhs+=MP(sign(dot)*star_sign(auxiliary,physical))*
                                pv[physical]*av[auxiliary];
                        }
                    MP lhs=hat[total];
                    convolution_maximum=std::max(convolution_maximum,
                        magnitude(lhs-rhs)/std::max({1.,magnitude(lhs),magnitude(rhs)}));
                    convolution_count++;
                }
            }
        }
        double identity_error=maximum;size_t identity_count=count;maximum=0;count=0;
        scblocks::Setup setup("theta_ns");setup.b=parse<MP>(settings.b);
        for(int parity=0;parity<8;parity++)
            require(theta_bpz_weight<MP>(parity)==
                    plumbing_bpz_weight<MP>(setup.graph,parity),
                    "theta and graph BPZ sewing disagree");
        for(int j=0;j<3;j++)setup.p[j]=parse<MP>(settings.momenta[j]);
        MP Q=setup.b+MP(1)/setup.b;std::vector<MP>weights;
        for(auto p:setup.p)weights.push_back(Q*Q/MP(8)-p*p/MP(2));
        auto vacuum=scblocks::ns_schottky(setup.graph,setup.domain(3));
        // The oscillator recursion has the Koszul minus before BPZ sewing.
        for(int i=0;i<3;i++)for(int j=i+1;j<3;j++) {
            scblocks::Key k{};k[i]=k[j]=3;
            require(vacuum.at(k)==Rational(-1),"wrong Schottky leading spin lift");
        }
        scblocks::NSRecursion recursion(setup.graph,rational<MP>(3,2)+MP(3)*Q*Q,weights,vacuum);
        auto coefficients=recursion.coefficients(scblocks::indices(setup,3));
        scblocks::Sewing physical(setup,false),native_ns(setup,false,true);
        for(auto &[k,value]:coefficients) {
            auto row=physical.coefficient(k,setup.cases[scblocks::total(k)%2]);
            auto native_row=native_ns.coefficient(k,setup.cases[scblocks::total(k)%2]);
            equal(value,theta_bpz_weight<MP>(scblocks::parity_mask(k))
                            *scblocks::getrow(row,scblocks::parity_mask(k)));
            MP native_value=scblocks::getrow(native_row,scblocks::parity_mask(k));
            ns_native_maximum=std::max(ns_native_maximum,
                magnitude(value-native_value)/std::max({1.,magnitude(value),magnitude(native_value)}));
            ns_native_count++;
        }
        std::cerr<<"identity_error="<<identity_error<<" native_maximum="<<native_maximum
                 <<" NS_maximum="<<maximum<<" convolution_maximum="<<convolution_maximum
                 <<" ns_native_maximum="<<ns_native_maximum<<'\n';
        require(identity_error<1e-25 && native_maximum<1e-25 && maximum<1e-30 &&
                convolution_maximum<1e-25 &&
                ns_native_maximum<1e-30,
                "partition block convention check failed");
        std::cout<<std::setprecision(17)<<"{\"dps\":40,\"level\":3,\"R_F_vs_PBW_components\":"<<identity_count
            <<",\"R_F_vs_PBW_maximum_scaled_error\":"<<identity_error
            <<",\"R_native_BPZ_vs_PBW_components\":"<<native_count
            <<",\"R_native_BPZ_vs_PBW_maximum_scaled_error\":"<<native_maximum
            <<",\"NS_vs_PBW_components\":"<<count<<",\"NS_vs_PBW_maximum_scaled_error\":"<<maximum
            <<",\"NS_native_BPZ_vs_recursion_components\":"<<ns_native_count
            <<",\"NS_native_BPZ_vs_recursion_maximum_scaled_error\":"<<ns_native_maximum
            <<",\"BPZ_native_convolution_components\":"<<convolution_count
            <<",\"BPZ_native_convolution_maximum_scaled_error\":"<<convolution_maximum
            <<",\"seconds\":"<<seconds()-start<<"}\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
