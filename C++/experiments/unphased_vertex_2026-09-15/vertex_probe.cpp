#include "ramond/anchors.hpp"
#include <fstream>
#include <iostream>
using namespace ramond;
struct ProbeTerm { AuxState auxiliary; PBW physical; MP coefficient; };
int main(int argc,char **argv) {
    try {
        require(argc==2,"usage: vertex_probe OUTPUT"); MP::precision(40);
        MP b=parse<MP>("7/5"); std::array<MP,3> p{parse<MP>("11/23"),parse<MP>("13/29"),parse<MP>("17/31")};
        std::array<std::unique_ptr<FreeField<MP>>,3> free;
        std::array<std::unique_ptr<PBWModule<MP>>,3> pbw;
        for(int j=0;j<3;j++) { free[j]=std::make_unique<FreeField<MP>>(j!=0,b,p[j]); pbw[j]=std::make_unique<PBWModule<MP>>(*free[j]); }
        UnphasedPhysicalForm<MP> raw(b,p,0);
        PhysicalForm<MP> production({pbw[0].get(),pbw[1].get(),pbw[2].get()},0,1,0);
        FermionForm aux;
        auto expand=[&](int slot,const Sparse<MP> &expr) {
            std::map<AuxState,Sparse<MP>> groups;
            for(const auto &[id,v]:expr) {
                State state=free[slot]->states.at(id); AuxState a;
                for(int bit=63;bit>=0;bit--) if((state.auxiliary>>bit)&1) a.modes.push_back(slot?bit+1:2*bit+1);
                a.ground=state.auxiliary_ground; state.auxiliary=0; state.auxiliary_ground=0;
                groups[a][free[slot]->intern(state)]+=v;
            }
            std::vector<ProbeTerm> out;
            for(const auto &[a,v]:groups) for(const auto &[state,c]:pbw[slot]->from_fock(v)) out.push_back({a,state,c});
            return out;
        };
        auto evaluate=[&](int copy,int convention) {
            std::array<std::vector<ProbeTerm>,3> terms;
            for(int j=0;j<3;j++) {
                auto expr=free[j]->primary(j?1:0,0);
                if(j==0 && copy>=0) expr=free[j]->apply(4+copy,-1,expr);
                terms[j]=expand(j,expr);
            }
            MP value(0);
            for(const auto &a:terms[0]) for(const auto &b:terms[1]) for(const auto &c:terms[2]) {
                int phase=pbw[0]->parity(a.physical)*aux_parity(0,a.auxiliary)+pbw[1]->parity(b.physical)*aux_parity(2,c.auxiliary);
                MP physical=convention?raw.value({a.physical,b.physical,c.physical},0,1):production.value({a.physical,b.physical,c.physical});
                if(convention==2) physical*=vertex_transport<MP>(a.physical.g.size(),b.physical.g.size(),b.physical.ground,0,0,aux_parity(1,b.auxiliary));
                value+=a.coefficient*b.coefficient*c.coefficient*MP(sign(phase))*aux.complex_value<MP>({a.auxiliary,b.auxiliary,c.auxiliary})*physical;
            }
            return value;
        };
        std::ofstream out(argv[1]); out<<"{\"test\":\"NS-slot L_-1 Virasoro Ward identity\",\"labels_4n\":[0,1,1],\"alpha_gamma\":[0,0],\"f\":0,\"eta\":1,\"dps\":40,\"records\":[";
        bool comma=false;
        for(int convention:{0,1,2}) for(int copy=0;copy<2;copy++) {
            const char *name=convention==0?"production":convention==1?"unphased":"transported";
            MP actual=evaluate(copy,convention);
            MP expected=(branch_weight(copy,0,b,p[0])+branch_weight(copy,1,b,p[1])-branch_weight(copy,1,b,p[2]))*evaluate(-1,convention);
            if(comma) out<<','; comma=true;
            out<<"{\"vertex\":\""<<name<<"\",\"copy\":"<<copy+1<<",\"actual\":"<<json_number(actual)<<",\"expected\":"<<json_number(expected)<<",\"absolute_error\":"<<magnitude(actual-expected)<<'}';
            std::cout<<name<<" copy "<<copy+1<<" absolute error "<<magnitude(actual-expected)<<'\n';
        }
        out<<"]}\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
