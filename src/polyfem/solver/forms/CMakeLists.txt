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
	LinearForm.cpp
	LinearForm.hpp
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

