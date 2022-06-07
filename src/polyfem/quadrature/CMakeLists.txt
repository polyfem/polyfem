set(SOURCES
	HexQuadrature.cpp
	HexQuadrature.hpp
	LineQuadrature.cpp
	LineQuadrature.hpp
	PolygonQuadrature.cpp
	PolygonQuadrature.hpp
	PolyhedronQuadrature.cpp
	PolyhedronQuadrature.hpp
	QuadQuadrature.cpp
	QuadQuadrature.hpp
	Quadrature.hpp
	TetQuadrature.cpp
	TetQuadrature.hpp
	TriQuadrature.cpp
	TriQuadrature.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
