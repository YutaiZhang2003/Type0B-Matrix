#include "ramond/resummed_pipeline.hpp"
#include <iostream>
using namespace ramond;
int main() {
    MP::precision(65);
    GlobalControls controls{1e-25, 80};
    GlobalDiagnostics diagnostic;
    std::array<MP,3> q{parse<MP>("-.029248438120443598,.019339142235614115"),
                      parse<MP>("-.01117750080023636,.00811019727805815"),
                      parse<MP>("-.056249225334039255,-.036215035442636094")};
    std::vector<MP> weights{parse<MP>("-.7052074942378977"),
                            parse<MP>("-.43221127907076157,.3279585744440939"),
                            parse<MP>("-.43172991162098456,.3272085430013835")};
    auto value=resummed_ccy(3,MP(24),weights,MP(0),q,0,controls,diagnostic).at(0);
    auto reference=parse<MP>("1.0081200330358524,-.035666827425668575");
    require(magnitude(value-reference)<3e-15,"complex theta 65-digit reference failed");
    // Both copies of the previously troublesome higher-primary branch
    // (n1,n2,n3)*4=(0,7,7), checked against the independent MP65 audit.
    for (const auto &fixture : std::vector<std::array<std::string,4>>{
            {"-.7052074942378977", "11.817788720929238,2.295710021108657",
             "11.818270088379016,2.2904598010096846", ".12775105672025958,-.18508708290158618"},
            {"1.3822066887062792", "-5.032865893021309,-2.2957100211086567",
             "-5.03380937322287,-2.2904598010096846", "1.6090568897891389,1.2235637525970475"}}) {
        std::array<MP,3> h{parse<MP>(fixture[0]),parse<MP>(fixture[1]),parse<MP>(fixture[2])};
        auto actual=theta_global_resummed(h,q,controls,diagnostic);
        require(magnitude(actual-parse<MP>(fixture[3]))<5e-14,"higher-primary MP65 reference failed");
    }
    // Independent direct unnormalized rho implementation from the polynomial
    // backend checks every punctured global coefficient in the resummed sum.
    std::array<MP,3> smallq{q[0]/MP(100),q[1]/MP(100),q[2]/MP(100)};
    for(int dim : {3,4}) {
        auto h=weights;
        if(dim==4) h.insert(h.begin()+2,parse<MP>("-.27,.51"));
        CCY<MP> engine(dim,MP(24),h,MP(-1));
        for(int order : {0,2,3,8}) {
            auto actual=resummed_ccy(dim,MP(24),h,MP(-1),smallq,order,controls,diagnostic);
            auto indices=virasoro_box_indices(Index{12,12,12,dim==4 ? 12:0});
            indices.erase(std::remove_if(indices.begin(),indices.end(),[](auto n){return degree(n)>12;}),indices.end());
            auto nulls=virasoro_indices(dim,order,order);
            auto amplitudes=engine.pole_amplitudes(nulls);
            Laurent<MP> direct;
            for(auto n:indices) {
                MP coefficient;
                for(const auto &[shift,amplitude]:amplitudes)
                    if(below(shift,n)) coefficient+=amplitude*engine.global_coefficient(shift,n-shift);
                if(dim==3) direct[0]+=coefficient*power(smallq[0],n[0])*power(smallq[1],n[1])*power(smallq[2],n[2]);
                else direct[n[1]-n[2]]+=coefficient*power(smallq[0],n[0])*power(root(smallq[1]),n[1]+n[2])*power(smallq[2],n[3]);
            }
            double difference=0;
            for(const auto &[k,v]:actual) difference+=magnitude(v-direct[k]);
            require(difference<2e-22,"resummed CCY differs from independent direct global coefficients");
            std::cout << "dimension=" << dim << " residue_order=" << order << " difference=" << difference << '\n';
        }
    }
    // Exact Laurent extraction catches unequal-level contamination and has
    // no finite Fourier-grid aliasing.
    Laurent<MP> a{{-2,MP(3)},{0,MP(5)},{2,MP(7)}}, b{{-1,MP(11)},{1,MP(13)}};
    require(diagonal_laurent_product(a,b,1)==MP(142),"shifted diagonal extraction failed");
    bool rejected=false;
    try { GlobalDiagnostics d; theta_global_resummed(std::array<MP,3>{weights[0],weights[1],weights[2]},q,GlobalControls{1e-30,4},d); }
    catch(const std::runtime_error &) { rejected=true; }
    require(rejected,"unconverged safety ceiling was silently accepted");
    std::cout << "global resummation, independent null order, complex roots and rejection: passed\n";
}
