set(SOURCES
	Form.hpp
	ParametrizationForm.hpp
	ALForm.hpp
	ALForm.cpp
	CompositeForm.hpp
	SumCompositeForm.hpp
	BodyForm.hpp
	BodyForm.cpp
	ElasticForm.hpp
	ElasticForm.cpp
	InertiaForm.hpp
	InertiaForm.cpp
	InertiaForm.hpp
	InertiaForm.cpp
	LaggedRegForm.cpp
	LaggedRegForm.hpp
	FrictionForm.cpp
	FrictionForm.hpp
	ContactForm.cpp
	ContactForm.hpp
	RayleighDampingForm.cpp
	RayleighDampingForm.hpp
)

prepend_current_path(SOURCES)
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(parametrization)
add_subdirectory(adjoint_forms)
