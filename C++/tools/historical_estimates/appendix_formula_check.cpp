#include "scblocks/partition_sewing.hpp"
#include <iostream>
using namespace ramond;
namespace nc=scblocks::nonchiral;
int sign(int n){return n%2?-1:1;}
int main(){
 using S=Machine; double error=0;
 auto eq=[&](S x,S y){error=std::max(error,std::abs(x-y)/std::max(1.,std::abs(y)));};
 for(int a:{0,1}){
  std::array<S,4> f{S(.2,.7),S(1.2,-.3),S(-.4,.9),S(.8,1.1)};
  auto exact=nc::reflected(f,a);
  for(int i=0;i<4;i++)eq(exact[i],S(.5)*(S(sign(a+1))*f[i]+f[i^1]+f[i^2]+S(sign(a))*f[i^3]));
 }
 nc::Channels<S> plus{},minus{},h{},anti{};
 for(int i=0;i<8;i++){plus[i]=S(.1*(i+1),.2*(i+2));minus[i]=S(.3*(i+2),-.1*(i+3));h[i]=(plus[i]+minus[i])/std::sqrt(2.);anti[i]=std::conj(h[i]);}
 std::array<S,2> c{S(.7),S(1.3)};
 for(int x:{-1,1})for(int y:{-1,1})for(int z:{-1,1}){
  S formula=0;
  for(int i=0;i<2;i++)for(int j=0;j<2;j++){
   int eta=i?-1:1,etap=j?-1:1;
   if(eta*etap!=-y*z)continue;
   for(int f=0;f<2;f++)for(int af=0;af<2;af++){
    int k=nc::channel(f,eta,etap),l=nc::channel(af,eta,etap);
    formula+=c[i]*c[j]*S((f+af)%2?x*z:1)*(plus[k]+minus[k])*std::conj(plus[l]+minus[l])/S(4);
   }
  }
  eq(formula,nc::contract(h,anti,c,c,{x,y,z}));
 }
 std::array<std::array<S,4>,2> ns{{{S(.2,.7),S(1.2,-.3),S(-.4,.9),S(.8,1.1)}, {S(.5,-.1),S(-.3,.6),S(1.1,.4),S(.4,-.8)}}};
 auto z=nc::observables(nc::target_density(ns,c,S(1)));
 for(int x:{1,-1}){
  S formula=0;
  for(int a:{0,1}){auto r=nc::reflected(ns[a],a);S Ca=(a?S(0,1):S(1))*c[a];formula+=S(sign(a))*Ca*Ca*std::norm(r[2]-S(0,x)*r[0])/S(2);}
  eq(formula,z[x==1?0:1]);
 }
 std::cout<<"{\"maximum_scaled_error\":"<<error<<",\"NS_lift_cases\":8,\"NSRR_tube_signs\":8,\"transported_target_cases\":2}"<<std::endl;
 return error<1e-13?0:1;
}
