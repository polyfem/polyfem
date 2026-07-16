set(SOURCES
    ActiveSelectionUtils.cpp
    ActiveSelectionUtils.hpp
    VariableToSimulation.hpp
    VariableToSimulationGroup.cpp
    VariableToSimulationGroup.hpp
    DampingVariableToSimulation.cpp
    DampingVariableToSimulation.hpp
    DirichletBoundaryVariableToSimulation.cpp
    DirichletBoundaryVariableToSimulation.hpp
    DirichletNodesVariableToSimulation.cpp
    DirichletNodesVariableToSimulation.hpp
    ElasticVariableToSimulation.cpp
    ElasticVariableToSimulation.hpp
    FrictionVariableToSimulation.cpp
    FrictionVariableToSimulation.hpp
    InitialConditionVariableToSimulation.cpp
    InitialConditionVariableToSimulation.hpp
    PeriodicShapeVariableToSimulation.cpp
    PeriodicShapeVariableToSimulation.hpp
    PressureBoundaryVariableToSimulation.cpp
    PressureBoundaryVariableToSimulation.hpp
    ShapeVariableToSimulation.cpp
    ShapeVariableToSimulation.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
