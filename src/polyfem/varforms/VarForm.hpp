#pragma once

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/io/OutputData.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

namespace polyfem
{
	class State;

	namespace varform
	{
		class VarForm
		{
		public:
			void set_args(const json &args) { this->args = args; }
			virtual ~VarForm() = default;

			virtual void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path);
			virtual void load_mesh(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) = 0;
			virtual void assemble_rhs(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void solve(Eigen::MatrixXd &sol) = 0;
			virtual void build_stiffness_mat(StiffnessMatrix &stiffness);
			virtual const StiffnessMatrix *mass_matrix() const { return nullptr; }

			virtual std::string name() const = 0;
			io::OutStatsData compute_errors(const Eigen::MatrixXd &solution);
			virtual io::OutputState output_state() const = 0;
			virtual std::vector<io::OutputField> output_fields(
				const io::OutputSample &sample,
				const Eigen::MatrixXd &solution,
				const io::OutputFieldOptions &options) const
			{
				return {};
			}

		protected:
			std::string resolve_input_path(const std::string &path, const bool only_if_exists = false) const;
			std::string resolve_output_path(const std::string &path) const;

			/// current problem, it contains rhs and bc
			std::shared_ptr<assembler::Problem> problem;
			Units units;
			json args;

			std::vector<std::shared_ptr<solver::Form>> forms;

			virtual void reset()
			{
				stats.reset();
			}

			bool iso_parametric;

			void assign_discr_orders(const json &discr_order, const mesh::Mesh &mesh, Eigen::VectorXi &disc_orders);

			io::OutStatsData stats;

			/// runtime statistics
			io::OutRuntimeData timings;

			std::string root_path;
			std::string output_path;

			const mesh::Mesh *mesh_ = nullptr;

			/// Geometric mapping bases, if the elements are isoparametric, this list is empty
			std::vector<basis::ElementBases> geom_bases_;
			/// number of geometric bases
			int n_geom_bases = 0;

			/// polyhedra/polygons, used since poly elements have no geometry mapping
			std::map<int, Eigen::MatrixXd> polys;
			std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

			/// Mapping from input nodes to geometric mapping nodes
			std::shared_ptr<polyfem::mesh::MeshNodes> geom_mesh_nodes;

			/// Input nodes (including high-order) to polyfem nodes, only for isoparametric
			Eigen::VectorXi in_node_to_node;
			/// maps input vertices/edges/faces/cells to polyfem vertices/edges/faces/cells
			Eigen::VectorXi in_primitive_to_primitive;

			/// list of boundary nodes
			std::vector<int> boundary_nodes;
			/// mapping from elements to nodes for all mesh
			std::vector<mesh::LocalBoundary> total_local_boundary;
			/// mapping from elements to nodes for dirichlet boundary conditions
			std::vector<mesh::LocalBoundary> local_boundary;
			/// mapping from elements to nodes for neumann boundary conditions
			std::vector<mesh::LocalBoundary> local_neumann_boundary;
			/// mapping from elements to nodes for pressure boundary conditions
			std::vector<mesh::LocalBoundary> local_pressure_boundary;
			/// mapping from elements to nodes for pressure boundary conditions
			std::unordered_map<int, std::vector<mesh::LocalBoundary>> local_pressure_cavity;
			/// per node dirichlet
			std::vector<int> dirichlet_nodes;
			std::vector<RowVectorNd> dirichlet_nodes_position;
			/// per node neumann
			std::vector<int> neumann_nodes;
			std::vector<RowVectorNd> neumann_nodes_position;
		};
	} // namespace varform
} // namespace polyfem
