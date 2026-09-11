#include "ramond/pipeline.hpp"
#include <filesystem>
#include <fstream>
using namespace ramond;
static int integer(const std::string &s) {
    size_t end = 0;
    int value = std::stoi(s, &end);
    require(end == s.size(), "invalid integer: " + s);
    return value;
}
static void help() {
    std::cout
        << "Usage: ramond --mode ordinary|inserted --level N --dps 0|DIGITS --json FILE\n"
           "  --b 7/5 --P1 11/23 --P2 13/29 --P3 17/31\n"
           "  --p 0|1 --f 0|1 --eta -1|1 --sector-policy error|record\n"
           "  --dps 0 uses machine complex arithmetic; higher precision requires >=30.\n"
           "  Complex momenta use real,imaginary (each part may be rational).\n"
           "  Both modes return the physical superconformal block; inserted uses Theta v_1/2.\n";
}
int main(int argc, char **argv) {
    try {
        Settings s;
        std::string output;
        for (int i = 1; i < argc; i++) {
            std::string key = argv[i];
            if (key == "--help") {
                help();
                return 0;
            }
            require(i + 1 < argc, "missing value for " + key);
            std::string value = argv[++i];
            if (key == "--mode") {
                require(value == "ordinary" || value == "inserted", "invalid mode");
                s.inserted = value == "inserted";
            } else if (key == "--level")
                s.level = integer(value);
            else if (key == "--dps")
                s.dps = integer(value);
            else if (key == "--p")
                s.p = integer(value);
            else if (key == "--f")
                s.f = integer(value);
            else if (key == "--eta")
                s.eta = integer(value);
            else if (key == "--b")
                s.b = value;
            else if (key == "--P1")
                s.momenta[0] = value;
            else if (key == "--P2")
                s.momenta[1] = value;
            else if (key == "--P3")
                s.momenta[2] = value;
            else if (key == "--json")
                output = value;
            else if (key == "--sector-policy") {
                require(value == "error" || value == "record", "invalid sector policy");
                s.record_sector = value == "record";
            } else
                throw std::invalid_argument("unknown argument: " + key);
        }
        require(!output.empty(), "--json is required");
        require(s.level >= 0 && s.level <= 1000, "level must be between zero and 1000");
        require((s.p == 0 || s.p == 1) && (s.f == 0 || s.f == 1) && std::abs(s.eta) == 1,
                "invalid parity or vertex sign");
        require(s.dps == 0 || s.dps >= 30, "dps must be zero or at least 30");
        std::filesystem::path path(output);
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        auto write = [&](const auto &result) {
            auto temporary = output + ".tmp";
            std::ofstream out(temporary);
            require(bool(out), "cannot open output file");
            encode_result(out, s, result);
            out.close();
            require(bool(out), "output write failed");
            std::filesystem::rename(temporary, output);
            std::cerr << "total computation: " << result.timing.total << " s; saved " << output
                      << '\n';
        };
        if (s.dps) {
            MP::precision(s.dps);
            write(pipeline<MP>(s));
        } else
            write(pipeline<Machine>(s));
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ramond: " << e.what() << '\n';
        return 1;
    }
}
