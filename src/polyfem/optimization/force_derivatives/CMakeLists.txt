set(SOURCES
    ElasticForceDerivative.hpp
    ElasticForceDerivative.cpp
    BodyForceDerivative.hpp
    BodyForceDerivative.cpp
    PressureForceDerivative.hpp
    PressureForceDerivative.cpp
    InertiaForceDerivative.hpp
    InertiaForceDerivative.cpp
    BarrierContactForceDerivative.hpp
    BarrierContactForceDerivative.cpp
    SmoothContactForceDerivative.hpp
    SmoothContactForceDerivative.cpp
    NormalAdhesionForceDerivative.hpp
    NormalAdhesionForceDerivative.cpp
    PeriodicContactForceDerivative.hpp
    PeriodicContactForceDerivative.cpp
    FrictionForceDerivative.hpp
    FrictionForceDerivative.cpp
    TangentialAdhesionForceDerivative.hpp
    TangentialAdhesionForceDerivative.cpp)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
