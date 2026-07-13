set(SOURCES
	Basis.cpp
	Basis.hpp
	ElementBases.cpp
	ElementBases.hpp
	EvalBasis.cpp
	EvalBasis.hpp
	EvalLagrangeBasis.cpp
	EvalLagrangeBasis.hpp
	EvalRationalBasis.cpp
	EvalRationalBasis.hpp
	LagrangeBasis2d.cpp
	LagrangeBasis2d.hpp
	LagrangeBasis3d.cpp
	LagrangeBasis3d.hpp
	function/QuadraticBSpline.cpp
	function/QuadraticBSpline.hpp
	function/QuadraticBSpline2d.cpp
	function/QuadraticBSpline2d.hpp
	function/QuadraticBSpline3d.cpp
	function/QuadraticBSpline3d.hpp
	function/RBFWithLinear.cpp
	function/RBFWithLinear.hpp
	function/RBFWithQuadratic.cpp
	function/RBFWithQuadratic.hpp
	function/RBFWithQuadraticLagrange.cpp
	function/RBFWithQuadraticLagrange.hpp
	InterfaceData.hpp
	PolygonalBasis2d.cpp
	PolygonalBasis2d.hpp
	PolygonalBasis3d.cpp
	PolygonalBasis3d.hpp
	SplineBasis2d.cpp
	SplineBasis2d.hpp
	SplineBasis3d.cpp
	SplineBasis3d.hpp
	barycentric/BarycentricBasis2d.cpp
	barycentric/BarycentricBasis2d.hpp
	barycentric/MVPolygonalBasis2d.cpp
	barycentric/MVPolygonalBasis2d.hpp
	barycentric/WSPolygonalBasis2d.cpp
	barycentric/WSPolygonalBasis2d.hpp
  BasisStore.hpp
  BasisStore.cpp
)

if(POLYFEM_WITH_CUDA)
	set_source_files_properties(
		EvalBasis.cpp
		EvalLagrangeBasis.cpp
		EvalRationalBasis.cpp
		TARGET_DIRECTORY polyfem
		PROPERTIES LANGUAGE CUDA)
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
