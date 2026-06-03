#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/assembler/Problem.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/io/OutStatsData.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

namespace polyfem::varform
{
	class VarForm;
} // namespace polyfem::varform

namespace polyfem::assembler
{
	class Assembler;
	class Mass;
	class MixedAssembler;
} // namespace polyfem::assembler

namespace polyfem::mesh
{
	class Obstacle;
} // namespace polyfem::mesh

namespace ipc
{
	class CollisionMesh;
} // namespace ipc

namespace polyfem::io
{
	struct OutputFieldOptions
	{
		std::vector<std::string> fields;

		bool export_field(const std::string &field) const;
	};

	struct OutputSample
	{
		Eigen::MatrixXd points;
		Eigen::MatrixXd local_points;
		Eigen::VectorXi element_ids;
		Eigen::VectorXi primitive_ids;
		Eigen::VectorXi node_ids;
		Eigen::MatrixXd normals;
		double time = 0;
		double dt = 0;
	};

	struct OutputField
	{
		enum class Association
		{
			Point,
			Cell
		};

		std::string name;
		Eigen::MatrixXd values;
		Association association = Association::Point;
	};

	struct OutputSpace
	{
		const mesh::Mesh *mesh = nullptr;
		const std::vector<basis::ElementBases> *geometry_bases = nullptr;
		Eigen::VectorXi output_orders;
		const std::map<int, Eigen::MatrixXd> *polys = nullptr;
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> *polys_3d = nullptr;
		const std::vector<mesh::LocalBoundary> *total_local_boundary = nullptr;
		const mesh::Obstacle *obstacle = nullptr;
		const ipc::CollisionMesh *collision_mesh = nullptr;
	};

	struct OutputState
	{
		const json &args;
		const mesh::Mesh *mesh;
		const assembler::Problem *problem;
		const assembler::Assembler *assembler;
		const assembler::Mass *mass_matrix_assembler;
		const assembler::MixedAssembler *mixed_assembler;
		const solver::SolveData &solve_data;

		const std::vector<basis::ElementBases> &bases;
		const std::vector<basis::ElementBases> &pressure_bases;
		const std::vector<basis::ElementBases> &geom_bases_;
		const int n_bases;
		const int n_pressure_bases;
		const Eigen::VectorXi &disc_orders;
		const Eigen::VectorXi &disc_ordersq;
		const Eigen::VectorXi &in_node_to_node;
		const Eigen::MatrixXd &rhs;
		const std::map<int, Eigen::MatrixXd> &polys;
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d;
		const mesh::Obstacle &obstacle;
		const ipc::CollisionMesh &collision_mesh;
		const std::vector<mesh::LocalBoundary> &total_local_boundary;
		const std::vector<int> &dirichlet_nodes;
		const std::vector<RowVectorNd> &dirichlet_nodes_position;

		const bool iso_parametric;
		const std::string formulation;
		const std::string primary_field_name;
		const std::string root_path;
		const std::string output_path;
		const QuadratureOrders boundary_samples;
		const OutStatsData &stats;
		const OutRuntimeData &timings;
		const double starting_min_edge_length;
		const varform::VarForm *var_form = nullptr;

		const std::vector<basis::ElementBases> &geom_bases() const
		{
			return iso_parametric ? bases : geom_bases_;
		}

		QuadratureOrders n_boundary_samples() const
		{
			return boundary_samples;
		}

		bool is_adhesion_enabled() const
		{
			return args["contact"]["adhesion"]["adhesion_enabled"];
		}
	};
} // namespace polyfem::io
