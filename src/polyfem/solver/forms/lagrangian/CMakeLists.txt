set(SOURCES
	BCLagrangianForm.cpp
	BCLagrangianForm.hpp
	PeriodicLagrangianForm.cpp
	PeriodicLagrangianForm.hpp
	AugmentedLagrangianForm.hpp
	MatrixLagrangianForm.hpp
	MatrixLagrangianForm.cpp
	MacroStrainLagrangianForm.cpp
	MacroStrainLagrangianForm.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
