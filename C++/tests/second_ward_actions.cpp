#include "ramond/free_field.hpp"
#include <iostream>
using namespace ramond;
template <class S> void run() {
    S b = S(7) / S(5), p = S(13) / S(29);
    for (int reflected : {0, 1}) {
        FreeField<S> ns(false, b, reflected ? -p : p);
        for (int n4 : {2, 4, 6}) {
            Fit fit;
            auto terms = ns_minus_action(ns, n4, fit);
            std::cout << "NS reflected=" << reflected << " n4=" << n4
                      << " residual=" << fit.relative_residual << std::endl;
        }
        FreeField<S> r(true, b, reflected ? -p : p);
        for (int n4 : {3, 5, 7})
            for (int mode : {0, 1}) {
                Fit fit;
                auto terms = ramond_nonnegative_action(r, n4, mode, fit);
                std::cout << "R reflected=" << reflected << " n4=" << n4 << " mode=" << mode
                          << " residual=" << fit.relative_residual << std::endl;
            }
    }
}
int main() {
    try {
        MP::precision(40);
        run<MP>();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
