set(SOURCES
	CL_m0.cpp
	LB_3p6_1p3.cpp
	subdiv_0_3p1_1p1.cpp
	subdiv_1_3p1_1p1.cpp
	subdiv_0_3p6_1p3.cpp
	subdiv_1_3p6_1p3.cpp
	P3TetCGV.cpp
)

target_sources(miso PRIVATE ${SOURCES})
