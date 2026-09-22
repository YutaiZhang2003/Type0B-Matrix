#include "ramond/resummed_pipeline.hpp"
#include <filesystem>
#include <fstream>
using namespace ramond;
int main(int argc, char **argv) {
    try {
        ResummedSettings s;
        s.branching.dps = 40;
        std::string output;
        auto integer = [](const std::string &v) {
            size_t end; int n = std::stoi(v, &end);
            require(end == v.size(), "invalid integer"); return n;
        };
        auto real = [](const std::string &v) {
            size_t end; double x = std::stod(v, &end);
            require(end == v.size() && std::isfinite(x), "invalid real number"); return x;
        };
        for (int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(key=="--help") {
                std::cout << "ramond_resummed --mode ordinary|inserted --branch-level N --recursion-order R\n"
                    "  --branch-truncation total|per-edge --global-tolerance 1e-13 --global-max-shell 64\n"
                    "  --q1 real,imag --q2 real,imag --q3 real,imag (human/tensor slots: NS,R1,R0)\n"
                    "  --b 1.4 --P1 0,p_NS --P2 0,p_R1 --P3 0,p_R0 --p 0 --f 0 --eta 1\n"
                    "  --dps 40 --json output.json\n"
                    "  Global ceilings cause failure if unconverged; they do not set residue order.\n";
                return 0;
            }
            require(i+1<argc,"missing option value");
            std::string v=argv[++i];
            if(key=="--mode") { require(v=="ordinary" || v=="inserted","invalid mode"); s.branching.inserted=v=="inserted"; }
            else if(key=="--branch-level") s.branching.level=integer(v);
            else if(key=="--recursion-order") s.recursion_order=integer(v);
            else if(key=="--branch-truncation") { require(v=="total" || v=="per-edge","invalid branching truncation"); s.branching.box=v=="per-edge"; }
            else if(key=="--global-tolerance") s.global.tolerance=real(v);
            else if(key=="--global-max-shell") s.global.maximum_shell=integer(v);
            else if(key=="--auxiliary-tolerance") s.auxiliary_tolerance=real(v);
            else if(key=="--auxiliary-max-level") s.auxiliary_max_level=integer(v);
            else if(key=="--vacuum-tolerance") s.vacuum_tolerance=real(v);
            else if(key=="--vacuum-max-arc-length") s.vacuum_max_arc_length=integer(v);
            else if(key=="--vacuum-max-mode") s.vacuum_max_mode=integer(v);
            else if(key=="--b") s.branching.b=v;
            else if(key=="--p") s.branching.p=integer(v);
            else if(key=="--f") s.branching.f=integer(v);
            else if(key=="--eta") s.branching.eta=integer(v);
            else if(key=="--dps") s.branching.dps=integer(v);
            else if(key=="--P1") s.branching.momenta[0]=v;
            else if(key=="--P2") s.branching.momenta[1]=v;
            else if(key=="--P3") s.branching.momenta[2]=v;
            else if(key=="--q1") s.plumbing[0]=v;
            else if(key=="--q2") s.plumbing[1]=v;
            else if(key=="--q3") s.plumbing[2]=v;
            else if(key=="--json") output=v;
            else throw std::runtime_error("unknown option: "+key);
        }
        require(!output.empty(),"--json is required");
        require(s.branching.dps==0 || s.branching.dps>=30,"precision must be zero or at least 30 digits");
        std::filesystem::path path(output);
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        auto write=[&](const auto &r) {
            std::ofstream stream(output+".tmp");
            require(bool(stream),"cannot open output");
            encode_resummed(stream,s,r); stream.close();
            require(bool(stream),"output write failed");
            std::filesystem::rename(output+".tmp",output);
            std::cerr << "resummed block: " << r.seconds << " s\n";
        };
        if(s.branching.dps) { MP::precision(s.branching.dps); write(resummed_pipeline<MP>(s)); }
        else write(resummed_pipeline<Machine>(s));
        return 0;
    } catch(const std::exception &e) { std::cerr << "ramond_resummed: " << e.what() << '\n'; return 1; }
}
