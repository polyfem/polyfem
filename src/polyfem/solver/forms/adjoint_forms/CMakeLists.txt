set(SOURCES
	AdjointForm.cpp
	AdjointForm.hpp
	ParametrizationForm.hpp
	SpatialIntegralForms.cpp
	SpatialIntegralForms.hpp
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
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
