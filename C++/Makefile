CXX = clang++
PYTHON ?= python3
CPPFLAGS += -Iinclude -I/opt/homebrew/include
CXXFLAGS ?= -O3 -g -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS += -L/opt/homebrew/lib
LDLIBS += -lmpc -lmpfr -lgmpxx -lgmp -lm
ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
else
LDLIBS += -llapack -lblas
endif
HEADERS = $(wildcard include/ramond/*.hpp)
GRAPH_HEADERS = $(wildcard include/scblocks/*.hpp include/scblocks/*.inc)
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:src/%.cpp=build/%.o)
all: bin/ramond bin/pbw bin/ramond_resummed
section6: all bin/graph_blocks bin/partition
partition: bin/partition
bin/partition_sewing: tests/partition_sewing.cpp $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/partition_blocks: tests/partition_blocks.cpp $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/partition_bpz: tests/partition_bpz.cpp $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
check-partition: bin/partition_sewing bin/partition_blocks bin/partition_bpz
	bin/partition_sewing
	bin/partition_blocks
	bin/partition_bpz
bin/partition: drivers/partition_main.cpp $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/graph_blocks: drivers/graph_main.cpp drivers/graph_pbw_check.inc drivers/graph_dv.inc $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/graph_ns_forward: tests/graph_ns_forward.cpp $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
check-ns-forward: bin/graph_ns_forward
	bin/graph_ns_forward
bin/graph_schottky: tests/graph_schottky.cpp $(GRAPH_HEADERS) $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
check-graph-schottky: bin/graph_schottky
	bin/graph_schottky
bin/pbw: drivers/pbw_main.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/ramond_resummed: tools/resummed_main.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/resummed_global: tests/resummed_global.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/resummed_recovery: tests/resummed_recovery.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
check-resummed: bin/ramond_resummed bin/resummed_global
	bin/resummed_global
bin/ramond: $(OBJECTS)
	@mkdir -p bin
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
build/%.o: src/%.cpp $(HEADERS)
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
clean:
	rm -rf build bin
check: check-partition check-branching check-ns-forward check-graph-schottky
check-saved: bin/ramond
	$(PYTHON) tests/verify_saved.py --run --level 3
check-pbw: bin/ramond
	$(PYTHON) tests/verify_fresh_pbw.py --below-order 5 --all-parities --results results/fresh_pbw_smoke
bin/outer_ward_driver: tests/outer_ward_driver.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/second_ward_actions: tests/second_ward_actions.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
bin/per_edge_cutoff: tests/per_edge_cutoff.cpp $(HEADERS)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
check-branching: bin/outer_ward_driver bin/second_ward_actions
	bin/second_ward_actions
	bin/outer_ward_driver 5 40 0
	bin/outer_ward_driver 5 40 0 1
	bin/outer_ward_driver 5 0 0 1
.PHONY: all section6 partition check-partition clean check check-saved check-pbw check-branching check-resummed check-ns-forward check-graph-schottky
