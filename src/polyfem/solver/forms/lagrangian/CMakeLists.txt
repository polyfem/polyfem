set(SOURCES
	BCLagrangianForm.cpp
	BCLagrangianForm.hpp
	BCPenaltyForm.cpp
	BCPenaltyForm.hpp
	CMakeLists.txt
	LagrangianForm.hpp
	LagrangianPenaltyForm.hpp
	MacroStrainLagrangianForm.cpp
	MacroStrainLagrangianForm.hpp
	MacroStrainALForm.cpp
	MacroStrainALForm.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
