#include "ramond/number.hpp"
#include "ramond/theta_bpz.hpp"
#include <array>
#include <iostream>
using namespace ramond;

int main() {
    try {
        MP::precision(40);
        const MP I(Machine(0,1)), beta=parse<MP>("13/29"), rt=root(MP(2));
        const MP a=I*beta*(MP(1)-I)/rt, b=I*beta*(MP(1)+I)/rt;
        // Tensor order is (u^0 w^+,u^0 w^-,u^1 w^+,u^1 w^-).
        auto parity=[](int s){return ((s>>1)+(s&1))%2;};
        auto metric=[&](int s){
            int af=s>>1,ps=s&1;
            return (af&&ps?MP(-1):MP(1))*(af?-I:MP(1))*(ps?MP(-1):MP(1));
        };
        auto G=[&](int s){
            int af=s>>1,ps=s&1;
            return std::pair<int,MP>{s^1,(af?MP(-1):MP(1))*(ps?b:a)};
        };
        auto psi=[&](int s){return std::pair<int,MP>{s^2,MP(1)/rt};};
        double maximum=0;size_t checks=0;
        auto equal=[&](MP x,MP y){
            maximum=std::max(maximum,magnitude(x-y)/std::max({1.,magnitude(x),magnitude(y)}));
            checks++;
        };
        for(int x=0;x<4;x++)for(int y=0;y<4;y++) {
            auto [gx,gc]=G(x);auto [gy,gd]=G(y);
            auto [px,pc]=psi(x);auto [py,pd]=psi(y);
            equal(gx==y?gc*metric(y):MP(0),
                  MP(parity(x)?-1:1)*I*(x==gy?gd*metric(x):MP(0)));
            equal(px==y?pc*metric(y):MP(0),
                  MP(parity(x)?-1:1)*(-I)*(x==py?pd*metric(x):MP(0)));
            auto [ugx,ugc]=G(x);auto [ux,upc]=psi(ugx);
            auto [ugy,ugd]=G(y);auto [uy,upd]=psi(ugy);
            // psi*G is even and is BPZ self-adjoint.
            equal(ux==y?ugc*upc*metric(y):MP(0),
                  x==uy?ugd*upd*metric(x):MP(0));
            equal(metric(x)*(x==y?MP(1):MP(0)),
                  (x==y?MP((x>>1)*(x&1)?-1:1)*
                          (x>>1?-I:MP(1))*(x&1?MP(-1):MP(1)):MP(0)));
        }
        size_t convolution_cases=0;
        for(int p=0;p<8;p++)for(int a=0;a<8;a++) {
            equal(theta_bpz_weight<MP>(p^a),
                  MP(__builtin_popcount(unsigned(p&a))%2?-1:1)*
                  theta_bpz_weight<MP>(p)*theta_bpz_weight<MP>(a));
            convolution_cases++;
        }
        require(maximum<1e-38,"graded tensor BPZ identity failed");
        std::cout<<"{\"ground_tensor_checks\":"<<checks
                 <<",\"theta_convolution_phase_checks\":"<<convolution_cases
                 <<",\"maximum_scaled_error\":"<<maximum<<"}\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
