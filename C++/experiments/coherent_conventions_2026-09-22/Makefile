CXX = clang++
CPPFLAGS += -I../../include -I/opt/homebrew/include
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS += -L/opt/homebrew/lib
LDLIBS += -lmpc -lmpfr -lgmpxx -lgmp -framework Accelerate
HEADERS = $(wildcard ../../include/ramond/*.hpp)
check: check.cpp unified_ward.hpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) check.cpp $(LDFLAGS) $(LDLIBS) -o $@
blocks: blocks.cpp unified_ward.hpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) blocks.cpp $(LDFLAGS) $(LDLIBS) -o $@
