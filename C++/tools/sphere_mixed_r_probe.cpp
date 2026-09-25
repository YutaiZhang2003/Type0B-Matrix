#include "ramond/direct_pbw.hpp"
#include <iomanip>
#include <iostream>
#include <string>
using namespace ramond;

int main(int argc, char** argv) {
    using S=Machine;
    if(argc!=1 && argc!=12) return 2;
    const S imaginary(0,1), c(argc==12?std::stod(argv[1]):3);
    auto hns=[&](S p) {return (c-S(1.5))/S(24)+p*p/S(2);};
    auto hr=[&](S p) {return hns(p)+S(0.0625);};
    std::array<S,5> p{{S(0.22),S(0.27),S(0.31),S(0.37),S(0.43)}};
    if(argc==12) for(int j=0;j<5;j++)
        p[j]=S(std::stod(argv[2+2*j]),std::stod(argv[3+2*j]));
    ScaWords<S> words;
    ScaModule<S> e1(words,hns(p[0]),c,0,false,false),
                 e2(words,hr(p[1]),c,-imaginary*S(p[1])/root(S(2)),true,false),
                 e3(words,hr(p[2]),c,-imaginary*S(p[2])/root(S(2)),true,false),
                 e4(words,hns(p[3]),c,0,false,false),
                 internal(words,hr(p[4]),c,-imaginary*S(p[4])/root(S(2)),true,false);
    int star=2*words.intern({{1,-1}});
    std::cout<<std::setprecision(17);
    for(int level=0;level<=2;level++)for(int k=0;k<2;k++) {
        std::vector<int> states;
        for(int l=0;l<=level;l++)
            for(const auto &ls:partitions(l))
                for(const auto &gs:partitions(level-l,true,false)) {
                    ScaWord word;
                    for(int n:ls)word.push_back({0,-2*n});
                    for(int n:gs)word.push_back({1,-2*n});
                    int g=(k+int(gs.size()))%2;
                    states.push_back(2*words.intern(word)+g);
                }
        int n=int(states.size());
        std::vector<S> gram(n*n);
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)
            gram[i*n+j]=(states[i]%2?imaginary:S(1))
                *internal.inner(states[i],states[j]);
        auto inverse=pbw_inverse(gram,n);
        for(int a1=0;a1<2;a1++)for(int a2=0;a2<2;a2++)
        for(int a3=0;a3<2;a3++)for(int a4=0;a4<2;a4++)
        for(int left_eta:{-1,1})for(int right_eta:{-1,1}) {
            int lf=(a4+a3+k)%2,rf=(k+a2+a1)%2;
            ScaWard<S> left(words,{{&e4,&e3,&internal}},0,lf,left_eta);
            ScaWard<S> right(words,{{&e1,&e2,&internal}},0,rf,right_eta);
            S value=0;
            for(int i=0;i<n;i++)for(int j=0;j<n;j++)
                value+=left.value({a4?star:0,a3,states[i]})
                    *inverse[i*n+j]
                    *right.value({a1?star:0,a2,states[j]});
            std::cout<<level<<' '<<k<<' '<<a1<<' '<<a2<<' '<<a3<<' '<<a4<<' '
                     <<left_eta<<' '<<right_eta<<' '
                     <<value.real()<<' '<<value.imag()<<'\n';
        }
    }
}
