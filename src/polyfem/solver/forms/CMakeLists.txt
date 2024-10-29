set(SOURCES
	Form.hpp
	BCLagrangianForm.hpp
	BCLagrangianForm.cpp
	BCPenaltyForm.hpp
	BCPenaltyForm.cpp
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
	NormalAdhesionForm.cpp
	NormalAdhesionForm.hpp
	TangentialAdhesionForm.cpp
	TangentialAdhesionForm.hpp
	PeriodicContactForm.cpp
	PeriodicContactForm.hpp
	MacroStrainLagrangianForm.cpp
	MacroStrainLagrangianForm.hpp
	MacroStrainALForm.cpp
	MacroStrainALForm.hpp
	RayleighDampingForm.cpp
	RayleighDampingForm.hpp
	PressureForm.cpp
	PressureForm.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

add_subdirectory(parametrization)
add_subdirectory(adjoint_forms)
