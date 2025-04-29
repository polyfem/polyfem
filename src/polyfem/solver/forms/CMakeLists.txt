set(SOURCES
	Form.hpp
	BodyForm.hpp
	BodyForm.cpp
	ElasticForm.hpp
	ElasticForm.cpp
	InertiaForm.hpp
	InertiaForm.cpp
	InertiaForm.hpp
	InertiaForm.cpp
	InversionBarrierForm.cpp
	InversionBarrierForm.hpp
	L2ProjectionForm.cpp
	L2ProjectionForm.hpp
	LaggedRegForm.cpp
	LaggedRegForm.hpp
	FrictionForm.cpp
	FrictionForm.hpp
	ContactForm.cpp
	ContactForm.hpp
	BarrierContactForm.cpp
	BarrierContactForm.hpp
	SmoothContactForm.cpp
	SmoothContactForm.hpp
	PeriodicContactForm.cpp
	PeriodicContactForm.hpp
	RayleighDampingForm.cpp
	RayleighDampingForm.hpp
	PressureForm.cpp
	PressureForm.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(parametrization)
add_subdirectory(adjoint_forms)
add_subdirectory(lagrangian)
