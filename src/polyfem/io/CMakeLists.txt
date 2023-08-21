set(SOURCES
	MatrixIO.cpp
	MatrixIO.hpp
	MshReader.cpp
	MshReader.hpp
	MshWriter.cpp
	MshWriter.hpp
	OBJReader.cpp
	OBJReader.hpp
	OBJWriter.cpp
	OBJWriter.hpp
	Evaluator.cpp
	OutData.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
