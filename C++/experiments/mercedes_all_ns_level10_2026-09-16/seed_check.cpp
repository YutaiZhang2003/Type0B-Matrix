#include "dv_engine.hpp"
#include "ns_schottky.hpp"
int main(){Graph g(6,{{0,1,2},{0,3,5},{1,4,3},{2,5,4}},{{0,3},{1,6},{2,9},{4,8},{7,11},{10,5}});auto x=ns_mercedes_schottky(g,10);return 0;}
