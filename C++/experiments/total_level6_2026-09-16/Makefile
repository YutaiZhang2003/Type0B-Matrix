CXX = clang++
CPPFLAGS += -I../../include -I/opt/homebrew/include
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS += -L/opt/homebrew/lib
LDLIBS += -lmpc -lmpfr -lgmpxx -lgmp -framework Accelerate
graph_validate: graph_validate.cpp graph_ccy.hpp ns_local.hpp pole_fragment.inc $(wildcard ../../include/ramond/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: all
all: graph_validate
	$(MAKE) -f theta_makefile theta_validate
