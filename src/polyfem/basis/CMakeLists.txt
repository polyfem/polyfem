set(SOURCES
	Basis.cpp
	Basis.hpp
	ElementBases.cpp
	ElementBases.hpp
	FEBasis2d.cpp
	FEBasis2d.hpp
	FEBasis3d.cpp
	FEBasis3d.hpp
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
	MVPolygonalBasis2d.cpp
	MVPolygonalBasis2d.hpp
	PolygonalBasis2d.cpp
	PolygonalBasis2d.hpp
	PolygonalBasis3d.cpp
	PolygonalBasis3d.hpp
	# SpectralBasis2d.cpp
	# SpectralBasis2d.hpp
	SplineBasis2d.cpp
	SplineBasis2d.hpp
	SplineBasis3d.cpp
	SplineBasis3d.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
