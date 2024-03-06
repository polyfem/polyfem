set(SOURCES
	Evaluator.cpp
	Evaluator.hpp
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
	OutData.cpp
	OutData.hpp
	YamlToJson.cpp
	YamlToJson.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
