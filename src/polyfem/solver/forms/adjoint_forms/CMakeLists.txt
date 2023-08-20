set(SOURCES
	AdjointForm.cpp
	AdjointForm.hpp
	ParametrizationForm.hpp
	ParametrizationForm.cpp
	SpatialIntegralForms.cpp
	SpatialIntegralForms.hpp
	SDFVariableToSimulation.cpp
	VariableToSimulation.cpp
	VariableToSimulation.hpp
	CompositeForm.hpp
	CompositeForms.cpp
	CompositeForms.hpp
	SumCompositeForm.hpp
	AMIPSForm.hpp
	WeightedVolumeForm.hpp
	ParametrizedProductForm.hpp
	TransientForm.hpp
	TransientForm.cpp
	BarrierForms.hpp
	SmoothingForms.hpp
	SmoothingForms.cpp
	TargetForms.hpp
	TargetForms.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
