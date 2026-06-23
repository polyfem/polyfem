set(SOURCES
    VarForm.cpp
    VarFormFactory.hpp
    VarFormFactory.cpp
    VarFormUtils.hpp
    FESpace.hpp
    ElasticVarForm.hpp
    ElasticVarForm.cpp
    LinearElasticVarForm.hpp
    LinearElasticVarForm.cpp
    ScalarVarForm.hpp
    ScalarVarForm.cpp
    NonlinearElasticVarForm.hpp
    NonlinearElasticVarForm.cpp
    FluidVarForm.hpp
    FluidVarForm.cpp
    IncompressibleElasticVarForm.hpp
    IncompressibleElasticVarForm.cpp
    BilaplacianVarForm.hpp
    BilaplacianVarForm.cpp
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "VarForms" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
