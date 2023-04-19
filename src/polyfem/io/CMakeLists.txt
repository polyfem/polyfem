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

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
