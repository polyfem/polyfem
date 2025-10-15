set(SOURCES
	Assembler.cpp
	Assembler.hpp
	AssemblerData.hpp
	AssemblerUtils.cpp
	AssemblerUtils.hpp
	AssemblyValsCache.cpp
	AssemblyValsCache.hpp
	AssemblyValues.hpp
	ElementAssemblyValues.cpp
	ElementAssemblyValues.hpp
	GenericElastic.cpp
	GenericElastic.hpp
	GenericFiber.cpp
	GenericFiber.hpp
	GenericProblem.cpp
	GenericProblem.hpp
	MacroStrain.hpp
	MatParams.cpp
	MatParams.hpp
	Problem.cpp
	Problem.hpp
	RhsAssembler.cpp
	RhsAssembler.hpp
	PeriodicBoundary.cpp
	PeriodicBoundary.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
add_subdirectory(local)
