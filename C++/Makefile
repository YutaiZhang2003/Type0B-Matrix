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
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:src/%.cpp=build/%.o)
all: bin/ramond
bin/ramond: $(OBJECTS)
	@mkdir -p bin
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
build/%.o: src/%.cpp $(HEADERS)
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
clean:
	rm -rf build bin
check: bin/ramond
	$(PYTHON) tests/verify_saved.py --run --level 3
.PHONY: all clean check
