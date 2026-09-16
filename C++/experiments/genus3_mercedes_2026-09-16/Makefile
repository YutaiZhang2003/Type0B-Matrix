CXX = clang++
CPPFLAGS += -I../../include -I/opt/homebrew/include
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS += -L/opt/homebrew/lib
LDLIBS += -lmpc -lmpfr -lgmpxx -lgmp -framework Accelerate
mercedes: mercedes.cpp central_data.hpp $(wildcard ../../include/ramond/*.hpp)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
central_data.hpp: export_ns.py
	python3 export_ns.py
