set(SOURCES
	Assembler.cpp
	Assembler.hpp
	Bilaplacian.cpp
	Bilaplacian.hpp
	AssemblyValues.hpp
	AssemblyValsCache.cpp
	AssemblyValsCache.hpp
	ElementAssemblyValues.cpp
	ElementAssemblyValues.hpp
	Helmholtz.cpp
	Helmholtz.hpp
	HookeLinearElasticity.cpp
	HookeLinearElasticity.hpp
	IncompressibleLinElast.cpp
	IncompressibleLinElast.hpp
	Laplacian.cpp
	Laplacian.hpp
	LinearElasticity.cpp
	LinearElasticity.hpp
	MassMatrixAssembler.cpp
	MassMatrixAssembler.hpp
	MultiModel.cpp
	MultiModel.hpp
	NeoHookeanElasticity.cpp
	NeoHookeanElasticity.hpp
	# OgdenElasticity.cpp
	# OgdenElasticity.hpp
	RhsAssembler.cpp
	RhsAssembler.hpp
	SaintVenantElasticity.cpp
	SaintVenantElasticity.hpp
	Stokes.cpp
	Stokes.hpp
	NavierStokes.cpp
	NavierStokes.hpp
	AssemblerUtils.cpp
	AssemblerUtils.hpp
)

prepend_current_path(SOURCES)
polyfem_copy_headers(${SOURCES})
polyfem_set_source_group(${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
