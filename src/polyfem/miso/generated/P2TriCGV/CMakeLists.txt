set(SOURCES
	CL_m0.cpp
	LB_2p2_1p2.cpp
	subdiv_0_2p1_1p1.cpp
	subdiv_1_2p1_1p1.cpp
	subdiv_0_2p2_1p2.cpp
	subdiv_1_2p2_1p2.cpp
	P2TriCGV.cpp
)

target_sources(miso PRIVATE ${SOURCES})
