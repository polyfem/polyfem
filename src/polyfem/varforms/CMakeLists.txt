set(SOURCES
    VarForm.cpp
    VarFormFactory.hpp
    VarFormFactory.cpp
    ElasticVarForm.hpp
    ElasticVarForm.cpp
    LinearElasticVarForm.hpp
    LinearElasticVarForm.cpp
    NonlinearElasticTransientVarForm.hpp
    NonlinearElasticTransientVarForm.cpp
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "VarForms" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
