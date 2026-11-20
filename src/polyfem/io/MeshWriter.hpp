#pragma once
#include <string>

namespace h5pp
{
	class File;
}
namespace polyfem::mesh
{
	class MeshData;
}

namespace polyfem::io
{
	/// Canonical mesh encoding shared by ordinary input bundles and checkpoints.
	void write_mesh(h5pp::File &file, const std::string &group, const mesh::MeshData &data);
} // namespace polyfem::io
