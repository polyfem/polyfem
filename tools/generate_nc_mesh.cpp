#include <polyfem/io/InputLoader.hpp>
#include <polyfem/io/MeshWriter.hpp>
#include <polyfem/mesh/GeometryLoader.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/NCMesh3D.hpp>
#include <polyfem/utils/GeogramUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/embedded_spec/polyfem.hpp>
#include <jse/jse.h>
#include <CLI/CLI.hpp>
#include <h5pp/h5pp.h>
#include <polysolve/linear/Solver.hpp>
#include <filesystem>
#include <iostream>
#include <iterator>

using namespace polyfem;
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
	CLI::App app("Generate a locally refined NC mesh and package a simulation input into HDF5.");
	std::string input_path, output_path, mesh_override;
	int base_refs = 0, local_refs = 1, final_refs = 0;
	double cut = 0;
	app.add_option("-j,--json", input_path, "Simulation JSON input")->required()->check(CLI::ExistingFile);
	app.add_option("-o,--output", output_path, "Output HDF5 input bundle")->required();
	app.add_option("--mesh", mesh_override, "Override the input's base mesh path")->check(CLI::ExistingFile);
	app.add_option("--base-refs", base_refs, "Uniform refinements before local refinement")->check(CLI::NonNegativeNumber);
	app.add_option("--local-refs", local_refs, "Local refinement passes")->check(CLI::NonNegativeNumber);
	app.add_option("--final-refs", final_refs, "Uniform refinements after local refinement")->check(CLI::NonNegativeNumber);
	auto *cut_option = app.add_option("--x-cut", cut, "Refine left of this x coordinate (default: 0.5 in 2D, 0.25 in 3D)");
	CLI11_PARSE(app, argc, argv);
	try
	{
		utils::GeogramUtils::instance().initialize();
		auto input = io::load_json_input(input_path);
		json config = input.config;
		auto resources = utils::apply_common_params(config, *input.resources);
		const io::ResourceIO &source_resources = resources ? *resources : *input.resources;
		const json source_config = config;
		auto geometry = utils::json_as_array(config.at("geometry"));
		if (geometry.size() != 1 || geometry[0].value("type", "mesh") != "mesh" || geometry[0].value("is_obstacle", false))
			log_and_throw_error("NC generation expects one FEM mesh geometry entry.");
		if (!mesh_override.empty())
			geometry[0]["mesh"] = fs::absolute(mesh_override).string();
		config["geometry"] = geometry;

		jse::JSE schema;
		auto rules = jse::embed::polyfem_spec::polyfem::spec();
		polysolve::linear::Solver::apply_default_solver(rules, "/solver/linear");
		polysolve::linear::Solver::apply_default_solver(rules, "/solver/adjoint_linear");
		if (!schema.verify_json(config, rules))
			log_and_throw_error("Invalid generation input: {}", schema.log2str());
		const json args = schema.inject_defaults(config, rules);
		Units units;
		units.init(args["units"]);
		mesh::GeometryLoader loader(units, source_resources);
		auto mesh = loader.load_fem(args["geometry"])->to_nonconforming();
		mesh->refine(base_refs, .5);
		mesh->prepare_mesh();
		if (cut_option->count() == 0)
			cut = mesh->dimension() == 2 ? .5 : .25;
		for (int pass = 0; pass < local_refs; ++pass)
		{
			std::vector<int> selected;
			for (int e = 0; e < mesh->n_elements(); ++e)
			{
				bool refine = mesh->dimension() == 2 && mesh->face_barycenter(e)(0) < cut;
				if (mesh->dimension() == 3)
					for (int j = 0; j < mesh->n_cell_vertices(e); ++j)
						refine |= mesh->point(mesh->cell_vertex(e, j))(0) < cut;
				if (refine)
					selected.push_back(e);
			}
			if (auto *nc = dynamic_cast<mesh::NCMesh2D *>(mesh.get()))
				nc->refine_elements(selected);
			else
				dynamic_cast<mesh::NCMesh3D &>(*mesh).refine_elements(selected);
			mesh->prepare_mesh();
		}
		mesh->refine(final_refs, .5);
		mesh->prepare_mesh();
		const auto data = mesh->to_mesh_data();
		data.validate();
		// Coordinates, selections, and refinements have already been applied once.
		config["geometry"] = json::array({{{"mesh", "/meshes/nc"}, {"type", "mesh"}, {"n_refs", 0}, {"advanced", {{"normalize_mesh", false}}}}});
		config.erase("root_path");
		config.erase("common");
		const fs::path output = fs::absolute(output_path);
		fs::create_directories(output.parent_path());
		if (fs::exists(output))
			log_and_throw_error("Output bundle already exists: {}", output.string());
		{
			h5pp::File file(output.string(), h5pp::FileAccess::REPLACE);
			file.writeDataset(config.dump(), "/config");
			io::write_mesh(file, "/meshes/nc", data);
			file.writeDataset(source_config.dump(), "/generation/source_config");
			auto source_stream = source_resources.open(geometry[0]["mesh"].get<std::string>(), true);
			const auto source_bytes = std::vector<unsigned char>(std::istreambuf_iterator<char>(*source_stream), std::istreambuf_iterator<char>());
			file.writeDataset(source_bytes, "/generation/source_mesh");
			file.writeDataset(json{{"base_refs", base_refs}, {"local_refs", local_refs}, {"final_refs", final_refs}, {"x_cut", cut}}.dump(), "/generation/refinement");
		}
		int hanging = 0;
		for (int v = 0; v < mesh->n_vertices(); ++v)
		{
			if (auto *nc = dynamic_cast<const mesh::NCMesh2D *>(mesh.get()))
				hanging += nc->leader_edge_of_vertex(v) >= 0;
			else
			{
				const auto &nc3d = dynamic_cast<const mesh::NCMesh3D &>(*mesh);
				hanging += nc3d.leader_edge_of_vertex(v) >= 0 || nc3d.leader_face_of_vertex(v) >= 0;
			}
		}
		std::cout << "Wrote " << output << ": " << mesh->n_elements() << " active elements, " << mesh->n_vertices() << " active vertices, " << hanging << " hanging vertices.\n";
		return 0;
	}
	catch (const std::exception &error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
