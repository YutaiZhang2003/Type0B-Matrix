#include "ramond/schottky.hpp"
#include <iostream>
using namespace ramond;
int main() {
    try {
        auto q = schottky_vacuum(6);
        std::cout << "{\"vacuum\":[";
        bool comma = false;
        for (const auto &[k, v] : q) {
            if (comma)
                std::cout << ',';
            comma = true;
            std::cout << "{\"levels\":[" << k[0] << ',' << k[1] << ',' << k[2] << "],\"value\":\""
                      << v.get_str() << "\"}";
        }
        std::cout << "],\"fermion\":[";
        comma = false;
        auto f = fermion_series(3, true);
        for (const auto &[k, v] : f) {
            if (comma)
                std::cout << ',';
            comma = true;
            std::cout << "{\"levels\":[" << k[0] << ',' << k[1] << ',' << k[2] << ',' << k[3]
                      << "],\"values\":[";
            for (int i = 0; i < 8; i++) {
                if (i)
                    std::cout << ',';
                std::cout << '\"' << v[i].get_str() << '\"';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
