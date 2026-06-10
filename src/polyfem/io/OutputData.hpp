#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <string>
#include <vector>

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
		enum class Domain
		{
			Unknown,
			Volume,
			Surface,
			Contact,
			Wire,
			Points,
			Grid
		};

		Eigen::MatrixXd points;
		Eigen::MatrixXd local_points;
		Eigen::VectorXi element_ids;
		Eigen::VectorXi primitive_ids;
		Eigen::VectorXi node_ids;
		Eigen::MatrixXd normals;
		std::vector<std::string> requested_fields;
		Domain domain = Domain::Unknown;
		int cell_count = 0;
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

	using OutputFieldFunction = std::function<std::vector<OutputField>(const OutputSample &)>;

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
		const std::vector<int> *dirichlet_nodes = nullptr;
		const std::vector<RowVectorNd> *dirichlet_nodes_position = nullptr;
	};
} // namespace polyfem::io
