CXX=clang++
CPPFLAGS=-I../../include -I/opt/homebrew/include
CXXFLAGS=-O2 -std=c++17 -Wall -Wextra
LDFLAGS=-L/opt/homebrew/lib
LDLIBS=-lmpc -lmpfr -lgmpxx -lgmp -framework Accelerate
compare: compare.cpp ns_recursion.hpp ns_schottky.hpp dv_engine.hpp graph_ccy.hpp pole_fragment.inc ns_local.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
