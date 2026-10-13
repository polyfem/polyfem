set(SOURCES
	CL_m0.cpp
	LB_2p6_1p2.cpp
	subdiv_0_2p1_1p1.cpp
	subdiv_1_2p1_1p1.cpp
	subdiv_0_2p6_1p2.cpp
	subdiv_1_2p6_1p2.cpp
	P4TriCGV.cpp
)

target_sources(miso PRIVATE ${SOURCES})
