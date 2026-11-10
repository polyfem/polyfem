set(SOURCES
	BCLagrangianForm.cpp
	BCLagrangianForm.hpp
	AugmentedLagrangianForm.hpp
	MatrixLagrangianForm.hpp
	MatrixLagrangianForm.cpp
	PeriodicBoundaryLagrangianForm.hpp
	PeriodicBoundaryLagrangianForm.cpp
	MacroStrainLagrangianForm.cpp
	MacroStrainLagrangianForm.hpp
	StackedAugmentedLagrangianForm.cpp
	StackedAugmentedLagrangianForm.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
