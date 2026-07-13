set(SOURCES
    MaterialExprRegistryImpl.hpp
    MaterialExprRegistry.hpp
    LameParameter.hpp
    LameParameter.cpp
    Dummy.hpp
    Density.hpp
    Density.cpp
    NeoHookean.hpp
    NeoHookean.cpp
    BuildMaterialExprFromJson.hpp
    BuildMaterialExprFromJson.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
