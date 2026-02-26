set(QUAD_SOURCES
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
	PrismQuadrature.cpp
	PrismQuadrature.hpp
	PyramidQuadrature.cpp
	PyramidQuadrature.hpp
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${QUAD_SOURCES})

add_library(polyfem_quadrature ${QUAD_SOURCES})
target_include_directories(polyfem_quadrature PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(polyfem_quadrature PRIVATE Eigen3::Eigen igl::core igl::predicates)
target_compile_features(polyfem_quadrature PRIVATE cxx_std_17)

target_link_libraries(polyfem PRIVATE polyfem_quadrature)
