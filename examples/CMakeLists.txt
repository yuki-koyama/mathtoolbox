add_subdirectory(bfgs)
add_subdirectory(classical-mds)
add_subdirectory(rbf-interpolation)

if(MATHTOOLBOX_NLOPT_BASED_MODULES)
	add_subdirectory(gaussian-process-regression)
endif()
