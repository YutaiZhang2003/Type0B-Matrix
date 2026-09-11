#include "ramond/anchors.hpp"
#include <iostream>
using namespace ramond;
template <class S> void run() {
    S b = parse<S>("7/5");
    std::array<S, 3> momenta{parse<S>("11/23"), parse<S>("13/29"), parse<S>("17/31")};
    LowAnchors<S> anchors(b, momenta, 0);
    std::cout << '[';
    bool comma = false;
    for (int n : {-2, 0, 2})
        for (int r : {-3, -1, 1, 3})
            for (int t : {-3, -1, 1, 3})
                for (int a = 0; a < 2; a++)
                    for (int g = 0; g < 2; g++) {
                        if (comma)
                            std::cout << ',';
                        comma = true;
                        std::cout << "{\"labels\":[" << n << ',' << r << ',' << t << "],\"a\":" << a
                                  << ",\"g\":" << g
                                  << ",\"value\":" << json_number(anchors.raw({n, r, t}, a, g, 1))
                                  << '}';
                    }
    std::cout << "]\n";
}
int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::stoi(argv[1])) {
            MP::precision(std::stoi(argv[1]));
            run<MP>();
        } else
            run<Machine>();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
