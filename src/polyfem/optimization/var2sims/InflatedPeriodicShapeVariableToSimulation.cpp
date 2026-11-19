#include "InflatedPeriodicShapeVariableToSimulation.hpp"

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/VarFormDiff.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace polyfem::solver
{
	InflatedPeriodicShapeVariableToSimulation::InflatedPeriodicShapeVariableToSimulation(
		std::vector<std::shared_ptr<varform::DifferentiableVarForm>> varforms,
		std::vector<std::shared_ptr<DiffCache>> diff_caches,
		CompositeParametrization parametrization, const json &settings, const io::ResourceIO &resources)
		: varforms_(std::move(varforms)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrization)),
		  dimension_(varforms_[0]->get_mesh().dimension()),
		  meshing_(settings.at("meshing"))
	{
		for (const auto &varform : varforms_)
		{
			if (varform->get_mesh().dimension() != dimension_ || !varform->is_homogenization())
			{
				log_and_throw_adjoint_error("Inflated periodic shape requires homogenization states of the same dimension.");
			}

			const Eigen::MatrixXd offsets = varform->periodic_tile_offsets();
			if (offsets.cols() != dimension_ || Eigen::FullPivLU<Eigen::MatrixXd>(offsets).rank() != dimension_)
			{
				log_and_throw_adjoint_error("Inflated periodic shape requires periodicity in every direction.");
			}
		}

		const auto parameters = settings.at("initial").get<std::vector<double>>();
		initial_ = Eigen::Map<const Eigen::VectorXd>(parameters.data(), parameters.size());

		// The 3D inflator uses reflection to obtain matching periodic boundary meshes.
		const std::string symmetry = settings.at("symmetry").get<std::string>();
		if (dimension_ == 2 && (symmetry == "auto" || symmetry == "doubly_periodic"))
		{
			options_.cell_type = inflator::CellType::Periodic2D;
		}
		else if (dimension_ == 2 && symmetry == "orthotropic")
		{
			options_.cell_type = inflator::CellType::Orthotropic2D;
		}
		else if (dimension_ == 3 && (symmetry == "auto" || symmetry == "orthotropic"))
		{
			options_.cell_type = inflator::CellType::Orthotropic3D;
		}
		else
		{
			log_and_throw_adjoint_error("Unsupported inflator symmetry {} in {}D.", symmetry, dimension_);
		}
		options_.tiles = settings.at("tiles").get<std::size_t>();
		options_.graph_radius = settings.at("graph_radius").get<std::size_t>();
		options_.contour_resolution = meshing_.at("marchingSquaresGridSize").get<std::size_t>();
		options_.force_contour_resolution = meshing_.at("forceMSGridSize").get<bool>();
		options_.max_triangle_area = meshing_.at("maxArea").get<double>();
		options_.facet_size = meshing_.at("facetSize").get<double>();
		options_.facet_distance = meshing_.at("facetDistance").get<double>();
		options_.cell_size = meshing_.at("cellSize").get<double>();
		options_.edge_size = meshing_.at("edgeSize").get<double>();
		wire_ = resources.materialize(settings.at("wire").get<std::string>());
		work_directory_ = varforms_[0]->output_file_path(settings.at("work_directory"));

		std::filesystem::create_directories(work_directory_);
	}

	bool InflatedPeriodicShapeVariableToSimulation::affects_varform(const varform::DifferentiableVarForm &target) const
	{
		return std::any_of(varforms_.begin(), varforms_.end(), [&](const auto &varform) { return varform.get() == &target; });
	}

	void InflatedPeriodicShapeVariableToSimulation::update(const Eigen::VectorXd &variables)
	{
		// Layout: [cell width, graph position DOFs..., vertex radii..., joint blending...].
		const Eigen::VectorXd parameters = parametrization_.eval(variables);
		if (parameters.size() != initial_.size() || !parameters.allFinite() || parameters(0) <= 0)
		{
			log_and_throw_adjoint_error("Invalid inflated periodic shape parameters.");
		}
		if (parameters.size() == last_parameters_.size() && parameters == last_parameters_)
		{
			return;
		}

		const auto directory = std::filesystem::path(work_directory_) / std::to_string(generation_++);
		if (!std::filesystem::create_directory(directory))
		{
			log_and_throw_adjoint_error("Inflator generation directory already exists: {}", directory.string());
		}

		// Debugging code
		// const std::string options_path = (directory / "meshing.json").string();
		// std::ofstream(options_path) << meshing_.dump(4);
		// std::ofstream(directory / "parameters.json") << json(std::vector<double>(parameters.data(), parameters.data() + parameters.size())).dump(4);

		// Microstructure is represented as a graph composed of joints, each with its own radius.
		// Due to historical reason, we use an external inflator program to convert between graph
		// and periodic mesh. This program also computes the chain rule term for first derivative.
		const inflator::Mesh mesh = inflator::inflate(wire_, options_, parameters.data() + 1, parameters.size() - 1);

		// Layouts: vertices[vertex * dim + axis], elements[cell * (dim + 1) + corner].
		// Elements are triangles in 2D and tetrahedra in 3D. Scale each length-2 inflator cell
		// to physical side lengths [width, 1, ...]; tiling multiplies these lengths by tiles.
		Eigen::MatrixXd vertices(mesh.vertices.size() / dimension_, dimension_);
		Eigen::MatrixXi cells(mesh.elements.size() / (dimension_ + 1), dimension_ + 1);
		for (int vertex = 0; vertex < vertices.rows(); ++vertex)
		{
			for (int axis = 0; axis < dimension_; ++axis)
			{
				const double cell_length = axis == 0 ? parameters(0) : 1.0;
				vertices(vertex, axis) = mesh.vertices[vertex * dimension_ + axis] * cell_length / 2;
			}
		}
		for (int cell = 0; cell < cells.rows(); ++cell)
		{
			for (int corner = 0; corner < cells.cols(); ++corner)
			{
				cells(cell, corner) = mesh.elements[cell * (dimension_ + 1) + corner];
			}
		}

		// Jacobian rows are [width, graph parameters...]; columns are interleaved vertex coordinates.
		shape_jacobian_.setZero(parameters.size(), vertices.size());
		for (int vertex = 0; vertex < vertices.rows(); ++vertex)
		{
			shape_jacobian_(0, dimension_ * vertex) = vertices(vertex, 0) / parameters(0);
		}

		// Velocities use [graph parameter][vertex][axis] order: normal speed times unit normal,
		// with zero interior entries.
		for (int parameter = 1; parameter < parameters.size(); ++parameter)
		{
			const std::size_t parameter_offset = (parameter - 1) * mesh.vertices.size();
			for (int vertex = 0; vertex < vertices.rows(); ++vertex)
			{
				for (int axis = 0; axis < dimension_; ++axis)
				{
					const Eigen::Index coordinate = dimension_ * vertex + axis;
					const double velocity = mesh.boundary_shape_velocities[parameter_offset + coordinate];
					const double cell_length = axis == 0 ? parameters(0) : 1.0;
					shape_jacobian_(parameter, coordinate) = velocity * cell_length / 2;
				}
			}
		}
		if (!vertices.allFinite() || !shape_jacobian_.allFinite())
		{
			log_and_throw_adjoint_error("Non-finite inflator geometry or shape velocities.");
		}

		// Regeneration can change vertex count and connectivity, requiring new periodic pairings.
		periodic_map_ = std::make_unique<PeriodicMeshToMesh>(vertices);
		periodic_parameters_ = periodic_map_->inverse_eval(utils::flatten(vertices));
		periodic_jacobian_.setZero(parameters.size(), periodic_parameters_.size());
		// Layout: [unique normalized vertex coordinates, cell matrix entries].
		// Divide physical velocities by tiled cell lengths and assign each shared coordinate once.
		std::vector<bool> visited(periodic_map_->n_periodic_dof(), false);
		for (int vertex = 0; vertex < vertices.rows(); ++vertex)
		{
			const int representative = periodic_map_->full_to_periodic(vertex);
			if (visited[representative])
			{
				continue;
			}

			visited[representative] = true;
			for (int axis = 0; axis < dimension_; ++axis)
			{
				const double tiled_length = options_.tiles * (axis == 0 ? parameters(0) : 1.0);
				const auto physical_derivatives = shape_jacobian_.block(1, dimension_ * vertex + axis, parameters.size() - 1, 1);
				auto periodic_derivatives = periodic_jacobian_.block(1, dimension_ * representative + axis, parameters.size() - 1, 1);
				periodic_derivatives = physical_derivatives / tiled_length;
			}
		}
		// Width changes only the cell matrix in periodic coordinates: cell_xx = tiles * width.
		const Eigen::Index cell_matrix_offset = periodic_parameters_.size() - dimension_ * dimension_;
		periodic_jacobian_(0, cell_matrix_offset) = options_.tiles;

		// File-based replace_mesh requires this MSH round-trip of the in-memory library output.
		const std::string mesh_path = (directory / "mesh.msh").string();
		const std::vector<int> body_ids(cells.rows(), 0);
		const bool is_volume = dimension_ == 3;
		io::MshWriter::write(mesh_path, vertices, cells, body_ids, is_volume, /*binary=*/true);
		for (int state = 0; state < varforms_.size(); ++state)
		{
			varforms_[state]->replace_mesh(mesh_path);
			// Clear mesh-dependent solutions and derivatives in place; objectives retain shared references to this cache.
			*diff_caches_[state] = DiffCache();
		}

		last_parameters_ = parameters;
		adjoint_logger().info("Inflated mesh generation {}: {} vertices, {} elements", generation_ - 1, vertices.rows(), cells.rows());
	}

	void InflatedPeriodicShapeVariableToSimulation::update_state_variables(const Eigen::VectorXd &variables, Eigen::VectorXd &state_variables) const
	{
		log_and_throw_adjoint_error("Generated meshes do not support fixed-connectivity SLIM updates.");
	}

	Eigen::VectorXd InflatedPeriodicShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &variables) const
	{
		Eigen::VectorXd result = Eigen::VectorXd::Zero(initial_.size());
		for (int state = 0; state < varforms_.size(); ++state)
		{
			Eigen::VectorXd term;
			AdjointTools::dJ_periodic_shape_adjoint_term(*varforms_[state],
														 *diff_caches_[state],
														 *periodic_map_,
														 periodic_parameters_,
														 diff_caches_[state]->u(0),
														 get_adjoint_mat(*varforms_[state], *diff_caches_[state], 0),
														 term);
			// Chain the FEM adjoint through the inflator's geometric shape velocities.
			result += periodic_jacobian_ * term;
		}

		return parametrization_.apply_jacobian(result, variables);
	}

	int InflatedPeriodicShapeVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(initial_.size());
	}

	Eigen::VectorXd InflatedPeriodicShapeVariableToSimulation::inverse_eval() const
	{
		return parametrization_.inverse_eval(initial_);
	}

	Eigen::VectorXd InflatedPeriodicShapeVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &variables) const
	{
		return parametrization_.apply_jacobian(shape_jacobian_ * term, variables);
	}
} // namespace polyfem::solver
