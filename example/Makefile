print-%: ; @echo $*=$($*)

dir_guard = @mkdir -pv $(@D)

PTPACK_HOME = ..

include_dirs = $(PTPACK_HOME)/include $(PTPACK_HOME)/3rd-party

lib_dir = $(PTPACK_HOME)/build/lib
ptpack_lib = $(lib_dir)/libptpack.a

CXXFLAGS := -Wall -O3 -m64 -std=c++11
CXXFLAGS += $(addprefix -I,$(include_dirs))

CXX := g++

all : build/learning_network_dynamics_exp_kernel build/learning_network_structure_exp_kernel build/learning_network_dynamics_general_kernel build/learning_network_structure_general_kernel build/learning_sparse_standard_hawkes build/learning_standard_hawkes build/learning_standard_hawkes_lowrank_constraint build/learning_lowrank_hawkes build/learning_standard_hawkes build/learning_standard_hawkes_learning_triggering_kernel build/influence_maximization build/learning_standard_self_inhibiting_process build/learning_sparse_self_inhibiting_process build/fast_simulating_standard_hawkes build/learning_hawkes_with_customized_kernel build/learning_sparse_hawkes_with_customized_kernel build/basic_residual_analysis

build/learning_network_dynamics_exp_kernel : learning_network_dynamics_exp_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_network_structure_exp_kernel : learning_network_structure_exp_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_network_dynamics_general_kernel : learning_network_dynamics_general_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_network_structure_general_kernel : learning_network_structure_general_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_sparse_standard_hawkes : learning_sparse_standard_hawkes.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_standard_hawkes : learning_standard_hawkes.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/fast_simulating_standard_hawkes : fast_simulating_standard_hawkes.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_standard_hawkes_lowrank_constraint : learning_standard_hawkes_lowrank_constraint.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_lowrank_hawkes : learning_lowrank_hawkes.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_hawkes_with_customized_kernel : learning_hawkes_with_customized_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_sparse_hawkes_with_customized_kernel : learning_sparse_hawkes_with_customized_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_standard_hawkes_learning_triggering_kernel : learning_standard_hawkes_learning_triggering_kernel.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/influence_maximization : influence_maximization.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_standard_self_inhibiting_process : learning_standard_self_inhibiting_process.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/learning_sparse_self_inhibiting_process : learning_sparse_self_inhibiting_process.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

build/basic_residual_analysis : basic_residual_analysis.cc $(ptpack_lib)
	$(dir_guard)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(lib_dir) -lptpack

clean:
	rm -rf build

-include $(DEPS)