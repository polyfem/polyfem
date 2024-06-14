set(SOURCES
	AdjointForm.cpp
	AdjointForm.hpp
	ParametrizationForm.hpp
	ParametrizationForm.cpp
	SpatialIntegralForms.cpp
	SpatialIntegralForms.hpp
	VariableToSimulation.cpp
	VariableToSimulation.hpp
	CompositeForm.cpp
	CompositeForm.hpp
	CompositeForms.cpp
	CompositeForms.hpp
	SumCompositeForm.hpp
	AMIPSForm.hpp
	WeightedVolumeForm.cpp
	WeightedVolumeForm.hpp
	ParametrizedProductForm.hpp
	TransientForm.hpp
	TransientForm.cpp
	BarrierForms.hpp
	BarrierForms.cpp
	SmoothingForms.hpp
	SmoothingForms.cpp
	SurfaceTractionForms.hpp
	SurfaceTractionForms.cpp
	TargetForms.hpp
	TargetForms.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
