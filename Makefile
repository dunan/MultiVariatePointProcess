print-%: ; @echo $*=$($*)

dir_guard = @mkdir -pv $(@D)

CXXFLAGS := -Wall -O3 -m64 -std=c++11
CXXFLAGS += -I./include -I./3rd-party
PROGRAM := execute

CXX := g++

shell := /bin/sh
cpp_files := $(shell find src -name "*.cc")
cxx_obj_files := $(subst .cc,.o,$(cpp_files))

obj_build_root := build

objs := $(addprefix $(obj_build_root)/,$(cxx_obj_files))
DEPS := ${objs:.o=.d}

all : $(objs)
	$(CXX) -o $(obj_build_root)/$(PROGRAM) $(CXXFLAGS) $(objs)

$(obj_build_root)/src/%.o : ./src/%.cc
	$(dir_guard)
	$(CXX) -MMD -c $(CXXFLAGS) -o $@ $<

clean:
	rm -rf build

-include $(DEPS)