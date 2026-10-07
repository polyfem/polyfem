set(SOURCES
	CL_m0.cpp
	LB_1p2.cpp
	subdiv_0_1p2.cpp
	subdiv_1_1p2.cpp
	subdiv_0_2p1_1p1.cpp
	subdiv_1_2p1_1p1.cpp
	P1TriCGV.cpp
)

target_sources(miso PRIVATE ${SOURCES})
