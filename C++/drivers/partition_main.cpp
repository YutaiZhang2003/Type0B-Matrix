#include "scblocks/ns_recursion.hpp"
#include "scblocks/json_input.hpp"
#include "scblocks/partition_sewing.hpp"
#include "ramond/pipeline.hpp"
#include <filesystem>
#include <sys/wait.h>
#include <cerrno>
#include <unistd.h>

namespace fs=std::filesystem;
using namespace ramond;
using scblocks::Json;
using scblocks::json_quote;
namespace nc=scblocks::nonchiral;
using Scalar=MP;

Scalar log_value(const Scalar &x) { Scalar y; mpc_log(y.data(),x.data(),MPC_RNDNN); return y; }
Scalar exp_value(const Scalar &x) { Scalar y; mpc_exp(y.data(),x.data(),MPC_RNDNN); return y; }
Scalar number(const Json &j) {
    if(j.kind==Json::Array) {
        require(j.array.size()==2,"complex array must have two components");
        return parse<Scalar>(j.at(0).scalar()+","+j.at(1).scalar());
    }
    if(j.kind==Json::Object) return parse<Scalar>(j.at("real").scalar()+","+j.at("imag").scalar());
    auto s=j.scalar();
    if(s.find('j')==std::string::npos) return parse<Scalar>(s);
    if(s.front()=='(' && s.back()==')') s=s.substr(1,s.size()-2);
    require(s.back()=='j',"invalid complex plumbing input"); s.pop_back();
    size_t split=std::string::npos;
    for(size_t i=1;i<s.size();i++) if((s[i]=='+' || s[i]=='-') && s[i-1]!='e' && s[i-1]!='E') split=i;
    require(split!=std::string::npos,"complex plumbing needs real and imaginary parts");
    return parse<Scalar>(s.substr(0,split)+","+s.substr(split));
}
template<size_t N> std::array<Scalar,N> numbers(const Json &j) {
    require(j.kind==Json::Array && j.array.size()==N,"incorrect input vector size");
    std::array<Scalar,N> x; for(size_t i=0;i<N;i++) x[i]=number(j.at(i)); return x;
}
template<size_t N> void vector_json(std::ostream &o,const std::array<Scalar,N>& v) {
    o<<'['; for(size_t i=0;i<N;i++) {if(i)o<<',';o<<json_number(v[i]);}o<<']';
}
void validate_config(const Json &config) {
    require(config.at("schema").scalar()=="paper-partition-fixed-spin-input-v2", "fixed-spin configuration required");
    auto check_spins=[&](const char* key,const std::array<std::array<int,4>,2>& expected) {
        const auto& spins=config.at(key);
        require(spins.kind==Json::Array && spins.array.size()==2,"two geometric spin characteristics required");
        for(int row=0;row<2;row++)for(int bit=0;bit<4;bit++)
            require(spins.at(row).at(bit/2).at(bit%2).integer()==expected[row][bit],"unsupported geometric spin dictionary");
    };
    check_spins("source_marked_spins",{{{1,1,0,0},{1,1,1,1}}});
    check_spins("target_marked_spins",{{{0,0,0,0},{0,0,1,0}}});
    // Transport the actual marked characteristics through the supplied
    // symplectic homology change, rather than merely accepting two labels.
    const auto &transport=config.at("source_to_target");
    require(transport.kind==Json::Array && transport.array.size()==4,"genus-two transport must be 4 by 4");
    std::array<std::array<long long,4>,4> M{};
    for(int i=0;i<4;i++) {
        require(transport.at(i).kind==Json::Array && transport.at(i).array.size()==4,
                "genus-two transport must be 4 by 4");
        for(int j=0;j<4;j++)M[i][j]=transport.at(i).at(j).integer();
    }
    auto J=[](int i,int j) {return i<2 && j==i+2 ? 1 : i>=2 && j==i-2 ? -1 : 0;};
    for(int i=0;i<4;i++)for(int j=0;j<4;j++) {
        long long value=0;
        for(int k=0;k<4;k++)for(int l=0;l<4;l++)value+=M[k][i]*J(k,l)*M[l][j];
        require(value==J(i,j),"source-to-target homology map is not symplectic");
    }
    auto mod2=[](long long value) {return int((value%2+2)%2);};
    for(int spin=0;spin<2;spin++) {
        const auto &source=config.at("source_marked_spins").at(spin),
                   &target=config.at("target_marked_spins").at(spin);
        for(int i=0;i<2;i++) {
            long long alpha=0,beta=0;
            for(int j=0;j<2;j++) {
                long long a=source.at(0).at(j).integer(),b=source.at(1).at(j).integer();
                alpha+=M[i+2][j+2]*a-M[i+2][j]*b
                      +M[i+2][j]*M[i+2][j+2];
                beta+=-M[i][j+2]*a+M[i][j]*b+M[i][j]*M[i][j+2];
            }
            require(mod2(alpha)==target.at(0).at(i).integer()
                 && mod2(beta)==target.at(1).at(i).integer(),
                    "source and target marked spins are not related by the homology map");
        }
    }
    auto check_signs=[&](const char* key) {
        const auto& signs=config.at(key);
        require(signs.kind==Json::Array && signs.array.size()==2,"two fixed plumbing sign assignments required");
        for(int spin=0;spin<2;spin++)for(int edge=0;edge<3;edge++)
            require(std::abs(signs.at(spin).at(edge).integer())==1,"fixed-spin plumbing signs must be +/-1");
    };
    check_signs("source_eta_e");check_signs("target_eta_e");
    auto bit=[](int x) { return (x%2+2)%2; };
    for(int spin=0;spin<2;spin++) {
        const auto& source_branch=config.at("source_period_branch");
        const auto& source_spin=config.at("source_marked_spins").at(spin);
        int alpha0=source_spin.at(0).at(0).integer(),alpha1=source_spin.at(0).at(1).integer();
        int beta0=bit(source_spin.at(1).at(0).integer()+source_branch.at(0).at(0).integer()*alpha0
            +source_branch.at(0).at(1).integer()*alpha1+source_branch.at(0).at(0).integer());
        int beta1=bit(source_spin.at(1).at(1).integer()+source_branch.at(1).at(0).integer()*alpha0
            +source_branch.at(1).at(1).integer()*alpha1+source_branch.at(1).at(1).integer());
        require(beta0==beta1,"source spin is not one of the two even Ramond families");
        const auto& source_signs=config.at("source_eta_e").at(spin);
        require(source_signs.at(0).integer()==(beta0?-1:1)
             && source_signs.at(1).integer()==-1 && source_signs.at(2).integer()==1,
             "source tube signs disagree with the marked spin and period branch");
        const auto& target_branch=config.at("target_period_branch");
        const auto& target_spin=config.at("target_marked_spins").at(spin);
        int target_beta0=bit(target_spin.at(1).at(0).integer()+target_branch.at(0).at(0).integer());
        int target_beta1=bit(target_spin.at(1).at(1).integer()+target_branch.at(1).at(1).integer());
        const auto& target_signs=config.at("target_eta_e").at(spin);
        require(target_signs.at(0).integer()==-1 && target_signs.at(1).integer()==(target_beta1?-1:1)
             && target_signs.at(2).integer()==(target_beta0?-1:1),
             "target tube signs disagree with the marked spin and period branch");
    }
    Scalar b=number(config.at("b"));
    require(machine(b).real()>0 && machine(b).imag()==0,"positive real b required");
    const auto &point=config.at("point");
    for(auto name:{"q_source","q_target"})for(auto q:numbers<3>(point.at(name)))
        require(magnitude(q)>0 && magnitude(q)<1,"plumbing values must have 0 < |q| < 1");
    for(auto name:{"source_free","target_free"}) {
        Scalar f=number(point.at(name));require(machine(f).real()>0 && machine(f).imag()==0,"positive free-frame factor required");
    }

}
std::array<std::array<int,3>,2> fixed_signs(const Json& config,const char* key) {
    std::array<std::array<int,3>,2> out{};
    for(int spin=0;spin<2;spin++)for(int edge=0;edge<3;edge++)out[spin][edge]=config.at(key).at(spin).at(edge).integer();
    return out;
}
void atomic(const fs::path &path,const std::string &s) {
    if(!path.parent_path().empty())fs::create_directories(path.parent_path());
    std::ofstream out(path.string()+".partial"); require(bool(out),"cannot open output");
    out<<s;out.close();require(bool(out),"failed output write");fs::rename(path.string()+".partial",path);
}
struct Options {
    std::string channel,config,node_directory,output,source,target;
    int level=5,minimum=-1,dps=40,workers=1,index=-1;
};
std::vector<fs::path> node_paths(const fs::path &p) {
    std::vector<fs::path> out;
    for(auto &entry:fs::directory_iterator(p))
        if(entry.is_regular_file() && entry.path().extension()==".json" && entry.path().filename().string().rfind("node-",0)==0) out.push_back(entry.path());
    std::sort(out.begin(),out.end());require(!out.empty(),"no momentum nodes");return out;
}
Scalar primary(Scalar b,const std::array<Scalar,3>& p,const std::array<Scalar,3>& q,bool rr) {
    Scalar Q=b+Scalar(1)/b,ans=0;
    for(int e=0;e<3;e++) {
        Scalar h=Q*Q/Scalar(8)+p[e]*p[e]/Scalar(2)+(rr && e>0?rational<Scalar>(1,16):Scalar(0));
        ans+=h*log_value(q[e]);
    }
    return exp_value(ans);
}
struct SourceBlocks { nc::F_R<Scalar> F{}; nc::Fixed_R<Scalar> fixed{}; };
std::map<int,SourceBlocks> source_blocks(const Options& opts,Scalar b,const std::array<Scalar,3>& p,
                         const std::array<Scalar,3>& q,
                         const std::array<std::array<int,3>,2>& source_signs,
                         double &ward,double &sector) {
    std::map<int,SourceBlocks> out;
    for(int l=opts.minimum;l<=opts.level;l++)out[l]={};
    std::array<Scalar,3> logs{log_value(q[0]),log_value(q[1]),log_value(q[2])};
    // Both f are computed directly. No projected-block identity is used.
    // Both fixed source spins have the same R-family pairing, eta'=eta.
    for(int eta:{1,-1})for(int f=0;f<2;f++) {
        Settings s;s.inserted=false;s.box=false;s.record_sector=false;s.native_bpz=true;
        s.p=0;s.f=f;s.eta=eta;
        s.level=opts.level;s.dps=opts.dps;s.b=decimal(mpc_realref(b.data()));
        for(int e=0;e<3;e++)s.momenta[e]="0,"+decimal(mpc_realref(p[e].data()));
        auto result=pipeline<Scalar>(s);
        ward=std::max(ward,result.ward_residual);sector=std::max(sector,result.sector_residual);
        for(const auto &[k,row]:result.physical) {
            require(k[1]==k[2],"source must use original physical edge degrees");
            Scalar monomial=exp_value(Scalar(k[0])*logs[0]/Scalar(2)+Scalar(k[1])*logs[1]+Scalar(k[3])*logs[2]/Scalar(2));
            for(int lift=0;lift<4;lift++) {
                Scalar value=nc::evaluate_parity(row,nc::eta_e[lift])*monomial;
                for(auto &[l,F]:out)if(degree(k)<=2*l)F.F[nc::channel(f,eta,eta)][lift]+=value;
            }
            for(auto &[l,F]:out)if(degree(k)<=2*l)
                for(int spin=0;spin<2;spin++)for(int epsilon=0;epsilon<8;epsilon++)
                    F.fixed[spin][f][eta==-1]
                        +=Scalar(nc::character(source_signs[spin],epsilon))
                          *row[epsilon]*monomial;
        }
    }
    return out;
}
using LiftBlocks=nc::F_NS<Scalar>;
struct TargetBlocks { LiftBlocks F{}; std::array<std::array<Scalar,2>,2> fixed{}; };
std::map<int,TargetBlocks> target_blocks(const Options& opts,Scalar b,const std::array<Scalar,3>& p,
                                     const std::array<Scalar,3>& q,
                                     const std::array<std::array<int,3>,2>& target_signs,
                                     const scblocks::QPoly& vacuum,
                                     size_t &transitions,size_t &seed_terms) {
    scblocks::Setup setup("theta_ns");
    Scalar Q=b+Scalar(1)/b,c=rational<Scalar>(3,2)+Scalar(3)*Q*Q;
    std::vector<Scalar> h;
    std::array<Scalar,3> logs{log_value(q[0]),log_value(q[1]),log_value(q[2])};
    for(int e=0;e<3;e++)h.push_back(Q*Q/Scalar(8)+p[e]*p[e]/Scalar(2));
    scblocks::NSRecursion recursion(setup.graph,c,h,vacuum);
    auto coefficients=recursion.coefficients(scblocks::indices(setup,opts.level));
    transitions=recursion.transitions;seed_terms=recursion.seed_terms;
    std::map<int,TargetBlocks> out;
    for(int l=opts.minimum;l<=opts.level;l++)out[l]={};
    for(const auto &[k,co]:coefficients) {
        int form=scblocks::total(k)%2;
        int epsilon=(k[0]%2)+2*(k[1]%2)+4*(k[2]%2);
        Scalar x=co*exp_value((Scalar(k[0])*logs[0]+Scalar(k[1])*logs[1]+Scalar(k[2])*logs[2])/Scalar(2));
        // The CCY output is the block in the theta BPZ sewing convention.
        for(auto &[l,F]:out)if(scblocks::total(k)<=2*l)
            for(int lift=0;lift<4;lift++) {
                Scalar term=Scalar(nc::character(nc::eta_e[lift],epsilon))*x;
                F.F[form][lift]+=term;
            }
        for(auto &[l,F]:out)if(scblocks::total(k)<=2*l)
            for(int spin=0;spin<2;spin++) {
                Scalar term=Scalar(nc::character(target_signs[spin],epsilon))*x;
                F.fixed[form][spin]+=term;
            }
    }
    return out;
}
void calculate_node(const Options& opts,const Json& config,const fs::path& input,const scblocks::QPoly& vacuum) {
    double started=seconds();
    Json node=Json::read(input.string());
    // Only these physical input fields participate in the new calculation.
    auto p=numbers<3>(node.at("momenta"));
    require(node.at("schema").scalar()=="paper-partition-node-input-v1","input convention missing");
    std::array<Scalar,2> C_a{}; nc::C_f_eta<Scalar> C_f_eta{};
    if(opts.channel=="target") C_a=numbers<2>(node.at("C_a"));
    else for(int f=0;f<2;f++) C_f_eta[f]=numbers<2>(node.at("C_f_eta").at(f));
    if(opts.channel=="source")for(int eta=0;eta<2;eta++)
        require(C_f_eta[0][eta]==C_f_eta[1][eta],"this ordered NS-R-R contraction requires C_1,eta=C_0,eta");
    require(node.at("N").integer()>0 && node.at("index").integer()>=0,"invalid quadrature node labels");
    if(node.has("channel"))require(node.at("channel").scalar()==opts.channel,"input node is from the other channel");
    Scalar measure=number(node.at("measure")),b=number(config.at("b"));
    require(machine(measure).real()>0 && machine(measure).imag()==0,"invalid momentum measure");
    for(auto &x:p)require(machine(x).real()>0 && machine(x).imag()==0,"positive continuum momenta required");
    auto q=numbers<3>(config.at("point").at(opts.channel=="source"?"q_source":"q_target"));
    Scalar prim=primary(b,p,q,opts.channel=="source");
    Scalar propagation=measure*prim*conjugate(prim);
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"schema\":\"ordered-fixed-spin-node-v5\",\"conventions\":\"ordered_yutai_bpz_sewing_2026-09-25\","
       <<"\"coefficient_convention\":"<<json_quote(nc::coefficient_convention)<<','
       <<"\"sewing_convention\":"<<json_quote(nc::sewing_convention)<<','
       <<"\"spin_status\":\"one fixed-sign evaluation of each algorithm block in each channel\","
       <<"\"implementation\":\"C++17\",\"normalization_factor\":1,\"chiral_blocks_fresh\":true,"
       <<"\"source_input\":"<<json_quote(input.string())<<",\"channel\":"<<json_quote(opts.channel)
       <<",\"index\":"<<node.at("index").integer()<<",\"N\":"<<node.at("N").integer()
       <<",\"dps\":"<<opts.dps<<",\"level\":"<<opts.level<<",\"minimum_level\":"<<opts.minimum
       <<",\"truncation\":\"total descendant level in both channels\",\"momenta\":";
    vector_json(out,p);out<<",\"edge_order\":[1,2,3],\"eta_e\":[";
    auto signs=fixed_signs(config,opts.channel=="source"?"source_eta_e":"target_eta_e");
    for(int spin=0;spin<2;spin++) {
        if(spin)out<<',';out<<'[';
        for(int edge=0;edge<3;edge++){if(edge)out<<',';out<<signs[spin][edge];}
        out<<']';
    }
    out<<']';
    if(opts.channel=="target") {out<<",\"C_a\":";vector_json(out,C_a);}
    else {out<<",\"C_f_eta\":[";vector_json(out,C_f_eta[0]);out<<',';vector_json(out,C_f_eta[1]);out<<']';}
    out<<",\"measure\":"<<json_number(measure)<<",\"primary\":"<<json_number(prim)<<",\"rows\":[";
    double ward=0,sector=0;size_t transitions=0,seed_terms=0;
    if(opts.channel=="source") {
        auto all=source_blocks(opts,b,p,q,signs,ward,sector);
        int count=0;
        for(auto &[l,blocks]:all) {
            std::array<Scalar,2> Z{};
            double maximum_form_identity_error=0;
            for(int spin=0;spin<2;spin++) {
                require(signs[spin][1]*signs[spin][2]==-1,"this source run requires the even R-cycle contraction");
                // The two ordered parity forms enter with the same reduced
                // coefficient. Each fixed spin is evaluated alone.
                for(int f=0;f<2;f++)for(int eta=0;eta<2;eta++) {
                    Scalar F=blocks.fixed[spin][f][eta];
                    Z[spin]+=propagation*C_f_eta[f][eta]*C_f_eta[f][eta]
                             *F*conjugate(F)/Scalar(4);
                }
                for(int eta=0;eta<2;eta++)
                    maximum_form_identity_error=std::max(maximum_form_identity_error,
                        magnitude(blocks.fixed[spin][0][eta]+blocks.fixed[spin][1][eta]));
            }
            if(count++)out<<',';
            out<<"{\"level\":"<<l<<",\"F\":[";
            int entry=0;
            for(int f=0;f<2;f++)for(int eta:{1,-1}) {
                if(entry++)out<<',';
                out<<"{\"f\":"<<f<<",\"eta\":"<<eta<<",\"eta_prime\":"<<eta<<",\"values\":";
                vector_json(out,blocks.F[nc::channel(f,eta,eta)]);out<<'}';
            }
            out<<"],\"F_fixed\":[";
            for(int spin=0;spin<2;spin++) {
                if(spin)out<<',';out<<'[';
                vector_json(out,blocks.fixed[spin][0]);out<<',';
                vector_json(out,blocks.fixed[spin][1]);out<<']';
            }
            out<<"],\"Z_R_fixed\":";vector_json(out,Z);
            out<<",\"maximum_form_identity_error\":"<<maximum_form_identity_error<<'}';
        }
    } else {
        auto all=target_blocks(opts,b,p,q,signs,vacuum,transitions,seed_terms);
        int count=0;
        for(auto &[l,blocks]:all) {
            std::array<Scalar,2> Z{};
            for(int spin=0;spin<2;spin++) {
                for(int a=0;a<2;a++) {
                    Scalar coupling=Scalar(a?-1:1)*C_a[a]*C_a[a]*propagation;
                    Z[spin]+=coupling*blocks.fixed[a][spin]
                             *conjugate(blocks.fixed[a][spin]);
                }
            }
            if(count++)out<<',';out<<"{\"level\":"<<l<<",\"F\":[";
            vector_json(out,blocks.F[0]);out<<',';vector_json(out,blocks.F[1]);
            out<<"],\"F_fixed\":[";
            vector_json(out,blocks.fixed[0]);out<<',';
            vector_json(out,blocks.fixed[1]);
            out<<"],\"Z_NS_fixed\":";vector_json(out,Z);
            out<<'}';
        }
    }
    out<<"],\"maximum_ward_residual\":"<<ward<<",\"maximum_sector_residual\":"<<sector
       <<",\"ns_transitions\":"<<transitions<<",\"ns_seed_terms\":"<<seed_terms
       <<",\"seconds\":"<<seconds()-started<<"}\n";
    atomic(fs::path(opts.output)/input.filename(),out.str());
}
void integrate(const Options& opts,const Json& config) {
    auto collect=[&](const std::string& dir,const std::string& expected) {
        auto run=Json::read((fs::path(dir)/"run.json").string());
        require(run.at("complete").kind==Json::Bool && run.at("complete").text=="true","run did not complete");
        require(run.at("config").scalar()==opts.config && run.at("channel").scalar()==expected,"run/configuration mismatch");
        auto files=node_paths(dir);std::map<int,std::array<Scalar,2>> sums;int N=-1;
        std::set<int> seen;
        for(const auto &path:files) {
            auto d=Json::read(path.string());
            require(d.at("channel").scalar()==expected && d.at("normalization_factor").integer()==1,"wrong channel/normalization");
            require(d.at("conventions").scalar()=="ordered_yutai_bpz_sewing_2026-09-25","wrong conventions");
            require(d.has("sewing_convention") && d.at("sewing_convention").scalar()==nc::sewing_convention,
                    "obsolete nonchiral sewing: recompute both fixed-spin channels");
            require(d.has("coefficient_convention") && d.at("coefficient_convention").scalar()==nc::coefficient_convention,
                    "obsolete or mixed structure-constant normalization; recompute with the identity-normalized coefficients");
            require(d.at("dps").integer()==run.at("dps").integer() && d.at("dps").integer()<=opts.dps,"mixed or insufficient reduction precision");
            int lower=run.at("minimum_level").integer(),upper=run.at("level").integer();
            require(d.at("minimum_level").integer()==lower && d.at("level").integer()==upper,"mixed cutoffs");
            require(int(d.at("rows").array.size())==upper-lower+1,"missing level rows");
            if(N<0)N=d.at("N").integer();require(N==d.at("N").integer(),"mixed grids");
            require(seen.insert(d.at("index").integer()).second,"duplicate node");
            for(auto &row:d.at("rows").array) {
                int l=row.at("level").integer();require(l==lower++,"missing or duplicate level row");
                auto values=numbers<2>(row.at(expected=="target"?"Z_NS_fixed":"Z_R_fixed"));
                for(int i=0;i<2;i++)sums[l][i]+=values[i];
            }
        }
        require(int(files.size())==N*N*N,"incomplete tensor grid");
        for(int j=0;j<N*N*N;j++)require(seen.count(j),"missing node index");
        return std::make_pair(N,sums);
    };
    auto src=collect(opts.source,"source"),tar=collect(opts.target,"target");
    Scalar b=number(config.at("b")),Q=b+Scalar(1)/b,kappa=Scalar(1)+Scalar(2)*Q*Q;
    Scalar frame=exp_value(kappa*log_value(number(config.at("point").at("source_free"))/number(config.at("point").at("target_free"))));
    constexpr int reported_spins=2;
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\"ordered-fixed-spin-comparison-v5\","
       <<"\"coefficient_convention\":"<<json_quote(nc::coefficient_convention)<<','
       <<"\"sewing_convention\":"<<json_quote(nc::sewing_convention)<<','
       <<"\"implementation\":\"C++17\",\"normalization_factor\":1,\"source_N\":"<<src.first<<",\"target_N\":"<<tar.first
       <<",\"free_frame_power\":"<<json_number(frame)<<",\"physical_inputs\":\"frozen momenta, measures, SCFT constants, geometry and free-frame factors; no saved chiral blocks used\","
       <<"\"spin_transport_status\":\"Two individually fixed marked spins, each transported to its target marked spin.\","
       <<"\"comparison_status\":\"algorithm blocks evaluated at one fixed tube-sign assignment per spin; no block conversion in this comparison\",\"comparisons\":[";
    int count=0;
    for(auto &[sl,sz]:src.second)for(auto &[tl,tz]:tar.second) {
        for(int spin=0;spin<reported_spins;spin++) {
            require(magnitude(tz[spin])>0,"zero target denominator");
            require(magnitude(sz[spin]-real_number(sz[spin]))<1e-30*magnitude(sz[spin]),"nonreal source contribution");
            require(magnitude(tz[spin]-real_number(tz[spin]))<1e-30*magnitude(tz[spin]),"nonreal target contribution");
        }
        if(count++)out<<',';
        out<<"{\"source_level\":"<<sl<<",\"target_level\":"<<tl<<",\"source_R_fixed\":";vector_json(out,sz);
        out<<",\"target_NS_fixed\":";vector_json(out,tz);
        out<<",\"fixed_spin_ratios\":[";
        for(int spin=0;spin<reported_spins;spin++) {
            if(spin)out<<',';out<<json_number(sz[spin]/tz[spin]/frame);
        }
        out<<"]}";
    }
    out<<"],\"source_eta_e\":";
    auto put_signs=[&](const char* key) {
        auto signs=fixed_signs(config,key);out<<'[';
        for(int spin=0;spin<2;spin++) {
            if(spin)out<<',';out<<'[';
            for(int edge=0;edge<3;edge++){if(edge)out<<',';out<<signs[spin][edge];}
            out<<']';
        }
        out<<']';
    };
    put_signs("source_eta_e");out<<",\"target_eta_e\":";put_signs("target_eta_e");out<<"}\n";
    atomic(opts.output,out.str());
}
int main(int argc,char**argv) {
    try {
        Options opts;
        for(int i=1;i<argc;i++) {
            std::string key=argv[i];
            if(key=="--help") {
                std::cout<<"partition --config frozen-config.json --node-directory DIR --channel source|target\n"
                         <<"  --level N --minimum-level M --dps 40 --workers 4 --output DIR [--index I]\n"
                         <<"partition --config frozen-config.json --reduce-source DIR --reduce-target DIR --output FILE\n"
                         <<"Input node fields: index, N, momenta (paper edge order), measure, C_a or C_f_eta.\n"
                         <<"Both channels are calculated natively at total descendant cutoff. No factor-four option.\n";return 0;
            }
            require(i+1<argc,"missing option value");std::string value=argv[++i];
            if(key=="--config")opts.config=value;
            else if(key=="--node-directory")opts.node_directory=value;
            else if(key=="--channel")opts.channel=value;
            else if(key=="--level")opts.level=std::stoi(value);
            else if(key=="--minimum-level")opts.minimum=std::stoi(value);
            else if(key=="--dps")opts.dps=std::stoi(value);
            else if(key=="--workers")opts.workers=std::stoi(value);
            else if(key=="--index")opts.index=std::stoi(value);
            else if(key=="--output")opts.output=value;
            else if(key=="--reduce-source")opts.source=value;
            else if(key=="--reduce-target")opts.target=value;
            else throw std::runtime_error("unknown option: "+key);
        }
        require(!opts.config.empty() && !opts.output.empty(),"config and output required");
        require(opts.dps>=30 && opts.dps<=10000,"partition driver needs 30 to 10000 decimal digits");
        MP::precision(opts.dps);auto config=Json::read(opts.config);validate_config(config);
        if(!opts.source.empty() || !opts.target.empty()) {
            require(!opts.source.empty() && !opts.target.empty(),"both reduction directories required");
            integrate(opts,config);return 0;
        }
        require(opts.channel=="source" || opts.channel=="target","select source or target");
        require(opts.level>=0 && opts.level<=20 && opts.workers>=1,"invalid run controls");
        if(opts.minimum<0)opts.minimum=opts.level;
        require(opts.minimum>=0 && opts.minimum<=opts.level,"invalid minimum level");
        fs::create_directories(opts.output);auto paths=node_paths(opts.node_directory);
        if(opts.index>=0) {
            std::vector<fs::path> chosen;
            for(auto &p:paths)if(Json::read(p.string()).at("index").integer()==opts.index)chosen.push_back(p);
            require(chosen.size()==1,"selected node missing or duplicated");paths=chosen;
        }
        for(auto &p:paths)require(!fs::exists(fs::path(opts.output)/p.filename()),"output already exists; use a fresh run directory");
        double started=seconds();scblocks::QPoly vacuum;
        if(opts.channel=="target") {
            scblocks::Setup setup("theta_ns");vacuum=scblocks::ns_schottky(setup.graph,setup.domain(opts.level));
        }
        double seed_seconds=seconds()-started;
        std::vector<pid_t> children;
        int workers=std::min(opts.workers,int(paths.size()));
        for(int w=0;w<workers;w++) {
            pid_t pid=fork();require(pid>=0,"fork failed");
            if(!pid) {
                std::ofstream log(fs::path(opts.output)/("worker-"+std::to_string(w)+".log"));std::cerr.rdbuf(log.rdbuf());
                try {
                    for(size_t j=w;j<paths.size();j+=workers) {
                        calculate_node(opts,config,paths[j],vacuum);
                        std::cerr<<"completed "<<paths[j].filename()<<'\n';
                    }
                    log.flush();_exit(0);
                } catch(const std::exception &e) {std::cerr<<"FAILED: "<<e.what()<<'\n';log.flush();_exit(1);}
            }
            children.push_back(pid);
        }
        bool ok=true;
        for(pid_t pid:children) {
            int status=0;pid_t result;
            do {result=waitpid(pid,&status,0);}while(result<0 && errno==EINTR);
            require(result==pid,"waitpid failed");ok=ok && WIFEXITED(status) && WEXITSTATUS(status)==0;
        }
        std::ostringstream report;report<<std::setprecision(17)<<"{\"complete\":"<<(ok?"true":"false")<<",\"channel\":"<<json_quote(opts.channel)
            <<",\"config\":"<<json_quote(opts.config)<<",\"node_directory\":"<<json_quote(opts.node_directory)<<",\"nodes\":"<<paths.size()
            <<",\"workers\":"<<workers<<",\"level\":"<<opts.level<<",\"minimum_level\":"<<opts.minimum<<",\"dps\":"<<opts.dps
            <<",\"schottky_seed_seconds\":"<<seed_seconds<<",\"wall_seconds\":"<<seconds()-started<<"}\n";
        atomic(fs::path(opts.output)/"run.json",report.str());std::cout<<report.str();
        require(ok,"worker failed; inspect worker logs");return 0;
    } catch(const std::exception &e) {std::cerr<<"partition: "<<e.what()<<'\n';return 1;}
}
