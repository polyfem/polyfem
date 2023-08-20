////////////////////////////////////////////////////////////////////////////////
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/State.hpp>

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::mesh;

TEST_CASE("append_2d", "[mesh_test]")
{
	// Used to init geogram
	State state;

	const std::string path = POLYFEM_DATA_DIR;
	auto m1 = Mesh::create(POLYFEM_DATA_DIR + std::string("/contact/meshes/2D/arch/largeArch.01.obj"));
	const auto m2 = Mesh::create(POLYFEM_DATA_DIR + std::string("/contact/meshes/2D/arch/largeArch.02.obj"));

	m1->append(m2);
}
