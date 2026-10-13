set(SOURCES
	CL_m0.cpp
	LB_1p3.cpp
	subdiv_0_1p3.cpp
	subdiv_1_1p3.cpp
	subdiv_0_3p1_1p1.cpp
	subdiv_1_3p1_1p1.cpp
	P1TetCGV.cpp
)

target_sources(miso PRIVATE ${SOURCES})
