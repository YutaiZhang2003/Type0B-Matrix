#include "ramond/resummed_pipeline.hpp"
using namespace ramond;
int main() {
    MP::precision(40);
    std::array<MP,3> q{parse<MP>("-.029248438120443598,.019339142235614115"),
                      parse<MP>("-.01117750080023636,.00811019727805815"),
                      parse<MP>("-.056249225334039255,-.036215035442636094")};
    ResummedSettings s;
    s.auxiliary_max_level=24;
    for(bool inserted : {false,true}) {
        s.branching.inserted=inserted;
        int reached=0; double change=0;
        std::cerr << "inserted=" << inserted << '\n';
        auto a=auxiliary_value(s,parse<MP>("74/35"),q,reached,change);
        for(const auto &x:a) std::cout << json_number(x) << '\n';
        auto spec=spectrum(a);
        auto back=inverse_spectrum(spec);
        for(int i=0;i<8;++i) require(magnitude(a[i]-back[i])<1e-35,"parity transform inverse");
    }
    int arc=0;double shell=0;
    auto vacuum=vacuum_value(s,q,arc,shell);
    std::cout << "vacuum " << json_number(vacuum) << " arc " << arc << " shell " << shell << '\n';
}
