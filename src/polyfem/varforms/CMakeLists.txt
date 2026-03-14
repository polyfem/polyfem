set(SOURCES
    VarForm.cpp
    NonlinearElasticTransientVarForm.hpp
    NonlinearElasticTransientVarForm.cpp
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "VarForms" FILES ${SOURCES})

target_sources(polyfem PRIVATE ${SOURCES})


