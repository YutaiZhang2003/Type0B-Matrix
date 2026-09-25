#pragma once
#include "ramond/number.hpp"
#include "ramond/theta_bpz.hpp"
#include <array>

namespace scblocks::nonchiral {
using ramond::Machine;
using ramond::require;
inline constexpr const char* coefficient_convention="unit_identity_upsilon_ordered_plus_i_eta_2026-09-25";
inline constexpr const char* sewing_convention="theta_BPZ_state_sum_in_block_2026-09-25";
// Paper edge order: (1,2,3)=(infinity,1,0). Fix eta_1=+1.
inline constexpr std::array<std::array<int,3>,4> eta_e{{{1,1,1},{1,-1,1},{1,1,-1},{1,-1,-1}}};
template<class S> using F_NS = std::array<std::array<S,4>,2>; // [a][eta_e]
template<class S> using F_R = std::array<std::array<S,4>,8>; // [f,eta,eta'][eta_e]
template<class S> using C_f_eta = std::array<std::array<S,2>,2>; // [f][eta=+,-]
template<class S> using Fixed_R = std::array<std::array<std::array<S,2>,2>,2>; // [fixed spin][f][eta=+,-]
inline int channel(int f,int eta,int etap) {
    require((f==0 || f==1) && (eta==1 || eta==-1) && (etap==1 || etap==-1),"invalid channel");
    return 4*f+2*(eta==-1)+(etap==-1);
}
inline int character(const std::array<int,3>& signs,int epsilon) {
    int value=1;
    for(int e=0;e<3;e++) {
        require(signs[e]==1 || signs[e]==-1,"eta_e must be +/-1");
        if(epsilon&(1<<e))value*=signs[e];
    }
    return value;
}
template<class S> S evaluate_parity(const std::array<S,8>& F,const std::array<int,3>& signs) {
    S value=0;
    for(int epsilon=0;epsilon<8;epsilon++)value+=S(character(signs,epsilon))*F[epsilon];
    return value;
}
// Koszul sign in the paper's three-edge PBW order.
inline int K(int e) {
    return ((e&1)*((e>>1)&1)+(e&1)*((e>>2)&1)+((e>>1)&1)*((e>>2)&1))%2;
}
// Physical two-family Ramond pairing for a fixed spin. The second form is computed and retained
// for an independent sector-identity check; it is not summed as another spin.
template<class S> S fixed_R_partition(const std::array<S,2>& F,
                                      const std::array<S,2>& Ftilde,
                                      const std::array<S,2>& C_left,
                                      const std::array<S,2>& C_right,
                                      S propagation) {
    S value=0;
    for(int eta=0;eta<2;eta++)
        value+=C_left[eta]*C_right[eta]*F[eta]*Ftilde[eta];
    return propagation*value/S(2);
}
template<class S,size_t N> std::array<std::array<S,4>,N> conjugated(const std::array<std::array<S,4>,N>& F) {
    auto Ftilde=F;
    for(auto& row:Ftilde)for(auto& value:row)value=ramond::conjugate(value);
    return Ftilde;
}
} // namespace scblocks::nonchiral
