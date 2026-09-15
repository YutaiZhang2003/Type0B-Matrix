CXX = clang++
CPPFLAGS += -Iinclude -I../../include/ramond -I../../include -I/opt/homebrew/include
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS += -L/opt/homebrew/lib
LDLIBS += -lmpc -lmpfr -lgmpxx -lgmp -lm
ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
else
LDLIBS += -llapack -lblas
endif
HEADERS = $(wildcard include/ramond/*.hpp ../../include/ramond/*.hpp)
all: ramond_unphased literal_pbw vertex_probe
ramond_unphased: main.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
literal_pbw: literal_pbw.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
vertex_probe: vertex_probe.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: all
