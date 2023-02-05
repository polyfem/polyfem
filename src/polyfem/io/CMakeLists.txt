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

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
