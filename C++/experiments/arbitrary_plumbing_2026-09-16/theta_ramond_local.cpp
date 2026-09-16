#include "ramond/direct_pbw.hpp"
#include <fstream>
#include <iostream>
using namespace ramond;
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: theta_ramond_local OUTPUT");MP::precision(40);
        ScaWords<MP> words;MP b=parse<MP>("7/5"),q=b+MP(1)/b,c=rational<MP>(3,2)+MP(3)*q*q;
        std::array<MP,3> p{parse<MP>("11/23"),parse<MP>("13/29"),parse<MP>("17/31")};
        std::array<std::unique_ptr<ScaModule<MP>>,3> modules;
        for(int j=0;j<3;j++)modules[j]=std::make_unique<ScaModule<MP>>(words,q*q/MP(8)-p[j]*p[j]/MP(2)+(j?rational<MP>(1,16):MP(0)),c,p[j]/root(MP(2)),j!=0,false);
        std::array<ScaModule<MP>*,3> ptr{modules[0].get(),modules[1].get(),modules[2].get()};
        std::array<int,3> ns{0,words.intern({{1,-1}}),words.intern({{0,-2}})};
        std::array<int,3> ramond{0,words.intern({{1,-2}}),words.intern({{0,-2}})};
        MP imag(Machine(0,1));std::array<MP,2> odd{(MP(1)-imag)/root(MP(2)),-(MP(1)+imag)/root(MP(2))};
        double worst=0;int count=0,failed=0;
        for(int primary=0;primary<2;primary++)for(int f=0;f<2;f++)for(int eta:{-1,1}) {
            ScaWard<MP> rho(words,ptr,primary,f,eta);
            for(int a:ns)for(int bb:ramond)for(int cc:ramond)for(int alpha=0;alpha<2;alpha++)for(int gamma=0;gamma<2;gamma++) {
                MP left=MP(sign(words.words[bb].parity+words.words[cc].parity))*odd[alpha]*odd[gamma]
                    *rho.value({2*a,2*bb+1-alpha,2*cc+1-gamma});
                MP right=-imag*MP(eta*sign(words.words[bb].parity+alpha))*rho.value({2*a,2*bb+alpha,2*cc+gamma});
                double error=magnitude(left-right)/std::max({1.,magnitude(left),magnitude(right)});
                worst=std::max(worst,error);count++;if(error>1e-30)failed++;
            }
        }
        std::ofstream out(argv[1]);out<<std::setprecision(17)
          <<"{\"scope\":\"p=0,1; f=0,1; eta=+/-1; NS states primary,G_-1/2,L_-1; each R state ground,G_-1,L_-1 and both grounds\","
          <<"\"identity\":\"rho(x,Jy,Jz)=-i eta (-1)^epsilon_y rho(x,y,z)\","
          <<"\"dps\":40,\"cases\":"<<count<<",\"failed\":"<<failed<<",\"max_scaled\":"<<worst<<"}\n";
        std::cout<<count<<" local identities, failed="<<failed<<", maxscaled="<<worst<<'\n';return failed?1:0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
