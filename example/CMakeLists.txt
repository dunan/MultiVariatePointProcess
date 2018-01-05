set(ptpack_example_targets
    learning_network_dynamics_exp_kernel
    learning_network_structure_exp_kernel
    learning_network_dynamics_general_kernel
    learning_network_structure_general_kernel
    learning_sparse_standard_hawkes
    learning_standard_hawkes
    fast_simulating_standard_hawkes
    learning_standard_hawkes_lowrank_constraint
    learning_lowrank_hawkes
    learning_hawkes_with_customized_kernel
    learning_sparse_hawkes_with_customized_kernel
    learning_standard_hawkes_learning_triggering_kernel
    influence_maximization
    learning_standard_self_inhibiting_process
    learning_sparse_self_inhibiting_process
    basic_residual_analysis
	)

foreach(ptpack_example_target ${ptpack_example_targets})
    add_executable(${ptpack_example_target} ${ptpack_example_target}.cc)
    target_link_libraries(${ptpack_example_target} ptpack)
endforeach()