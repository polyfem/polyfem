#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/Problem.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/solver/SolveData.hpp>
#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/io/OutputData.hpp>
#include <polyfem/io/OutData.hpp>
#include <polyfem/io/OutStatsData.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

#include <iosfwd>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

namespace polyfem
{
	namespace varform
	{
		struct VarFormDebugData
		{
			const mesh::Mesh *mesh = nullptr;
			const assembler::Assembler *assembler = nullptr;
			const std::vector<basis::ElementBases> *bases = nullptr;
			const std::vector<basis::ElementBases> *geometry_bases = nullptr;
			const std::vector<mesh::LocalBoundary> *total_local_boundary = nullptr;
			int n_bases = 0;
			int n_obstacle_vertices = 0;
			std::string root_path;
		};

		class VarFormTestAccess
		{
		public:
			virtual ~VarFormTestAccess() = default;
			virtual VarFormDebugData debug_data() const = 0;
		};

		class VarFormMatrixTestAccess : public VarFormTestAccess
		{
		public:
			virtual ~VarFormMatrixTestAccess() = default;
			virtual void build_stiffness_mat_debug(StiffnessMatrix &stiffness) = 0;
			virtual const StiffnessMatrix *mass_matrix_debug() const = 0;
		};

		class VarForm
		{
		public:
			virtual ~VarForm() = default;

			/// @brief Get the name of the variational formulation
			/// @return Name of the variational formulation
			virtual std::string name() const = 0;

			/// @brief Reset the internal state of the variational formulation, e.g. when loading a new mesh
			/// @param args json input arguments, used to determine which data to reset
			void set_args(const json &args) { this->args = args; }

			/// @brief Initialize the variational formulation with the given parameters
			/// @param formulation name of the variational formulation
			/// @param units unit system to use for the formulation
			/// @param args json input arguments, used to initialize the formulation
			/// @param out_path output path for the formulation, used to save intermediate data
			virtual void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path);

			/// @brief Set the mesh for the variational formulation
			/// @param mesh unique pointer to the mesh to use for the formulation
			void set_mesh(std::unique_ptr<mesh::Mesh> mesh, const double loading_mesh_time = 0);

			/// @brief Solve the variational formulation and store the solution in the given matrix
			/// @param sol matrix to store the solution
			void solve(Eigen::MatrixXd &sol);

			/// @brief Get the problem dimension of the variational formulation, for output purposes
			/// @return Problem dimension
			int problem_dimension() const;
			/// @brief Check if contact is enabled for the variational formulation, for output purposes
			/// @return True if contact is enabled, false otherwise
			virtual bool is_contact_enabled() const { return false; }

			/// @brief Get the output space of the variational formulation, for output purposes
			/// @return Output space
			virtual io::OutputSpace output_space() const;
			/// @brief Get the output fields of the variational formulation, for output purposes
			/// @param sample Output sample
			/// @param solution Solution matrix
			/// @param options Output field options
			/// @return Output fields
			virtual std::vector<io::OutputField> output_fields(
				const io::OutputSample &sample,
				const Eigen::MatrixXd &solution,
				const io::OutputFieldOptions &options) const = 0;

			/// @brief Get the runtime timings of the variational formulation, for output purposes
			/// @return Runtime timings
			const io::OutRuntimeData &output_timings() const { return timings; }
			/// @brief Get the error statistics of the variational formulation, for output purposes
			/// @return Error statistics
			virtual io::OutStatsData compute_errors(const Eigen::MatrixXd &solution);

			/// @brief Save the solution to a JSON file, for output purposes
			/// @param solution Solution matrix to save
			/// @param out Output stream to save the solution
			virtual void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const = 0;
			/// @brief 	Save the solution to a JSON file, for output purposes
			/// @param solution
			void save_json(const Eigen::MatrixXd &solution) const;
			void export_data(const Eigen::MatrixXd &solution) const;

			QuadratureOrders n_boundary_samples() const;
			void prepare();

		protected:
			std::string resolve_output_path(const std::string &path) const;
			std::string resolve_input_path(const std::string &path, const bool only_if_exists = false) const;

			virtual void set_materials(assembler::Assembler &assembler) const;
			virtual void reset();

			virtual void load_mesh(const mesh::Mesh &mesh, const json &args);
			virtual void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args);
			void assign_discr_orders(const json &discr_order, const mesh::Mesh &mesh, Eigen::VectorXi &disc_orders);
			virtual void assemble_rhs(const mesh::Mesh &mesh, const json &args);
			virtual void assemble_mass_mat(const mesh::Mesh &mesh, const json &args);
			virtual void solve_problem(Eigen::MatrixXd &sol) = 0;

			/// @brief Get a constant reference to the geometry mapping bases.
			/// @return A constant reference to the geometry mapping bases.
			const std::vector<basis::ElementBases> &geom_bases() const
			{
				return iso_parametric ? bases : geom_bases_;
			}

			void build_polygonal_basis(const mesh::Mesh &mesh);
			void build_node_mapping(const mesh::Mesh &mesh, const json &args);
			std::vector<int> primitive_to_node() const;
			std::vector<int> node_to_primitive() const;

			virtual void build_rhs_assembler();

			void initial_solution(Eigen::MatrixXd &solution) const;

			virtual void save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol) const;

			void ensure_output_sampler() const;
			std::vector<io::OutputField> common_output_fields(
				const io::OutputSample &sample,
				const Eigen::MatrixXd &solution,
				const io::OutputFieldOptions &options) const;
			void save_json_stats(
				const Eigen::MatrixXd &solution,
				const int n_auxiliary_bases,
				std::ostream &out) const;

			void save_restart_json(const double t0, const double dt, const int t) const;
			void save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &solution) const;
			void save_subsolve(const int i, const int t, const Eigen::MatrixXd &solution) const;
			int output_file_index(const int t) const;

			io::OutGeometryData::ExportOptions export_options(const io::OutputSpace &space) const;
			io::OutputFieldFunction output_field_function(const Eigen::MatrixXd &solution, const io::OutGeometryData::ExportOptions &opts) const;

			/// current problem, it contains rhs and bc
			std::shared_ptr<assembler::Problem> problem;
			Units units;
			json args;

			bool iso_parametric;

			io::OutStatsData stats;

			/// runtime statistics
			io::OutRuntimeData timings;

			std::string root_path;
			std::string output_path;

			std::unique_ptr<mesh::Mesh> mesh_;

			/// assembler corresponding to governing physical equations
			std::shared_ptr<assembler::Assembler> assembler = nullptr;
			std::shared_ptr<assembler::Mass> mass_matrix_assembler = nullptr;
			std::shared_ptr<assembler::HRZMass> pure_mass_matrix_assembler = nullptr;

			/// FE bases, the size is #elements
			std::vector<basis::ElementBases> bases;

			/// number of bases
			int n_bases = 0;

			/// vector of discretization orders, used when not all elements have the same degree, one per element
			Eigen::VectorXi disc_orders, disc_ordersq;

			/// Geometric mapping bases, if the elements are isoparametric, this list is empty
			std::vector<basis::ElementBases> geom_bases_;
			/// number of geometric bases
			int n_geom_bases = 0;

			/// polyhedra/polygons, used since poly elements have no geometry mapping
			std::map<int, Eigen::MatrixXd> polys;
			std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

			/// nodes on the boundary of polygonal elements, used for harmonic bases
			std::map<int, basis::InterfaceData> poly_edge_to_data;

			/// Mapping from input nodes to FE nodes
			std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes;

			/// Mapping from input nodes to geometric mapping nodes
			std::shared_ptr<polyfem::mesh::MeshNodes> geom_mesh_nodes;

			/// Input nodes (including high-order) to polyfem nodes, only for isoparametric
			Eigen::VectorXi in_node_to_node;
			/// maps input vertices/edges/faces/cells to polyfem vertices/edges/faces/cells
			Eigen::VectorXi in_primitive_to_primitive;

			/// used to store assembly values for small problems
			assembler::AssemblyValsCache ass_vals_cache;
			assembler::AssemblyValsCache mass_ass_vals_cache;
			assembler::AssemblyValsCache pure_mass_ass_vals_cache;

			std::shared_ptr<assembler::RhsAssembler> rhs_assembler;

			/// Mass matrix, it is computed only for time dependent problems
			StiffnessMatrix mass;
			StiffnessMatrix pure_mass;
			/// average system mass, used for contact with IPC
			double avg_mass = 0;
			Eigen::MatrixXd rhs;

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

			double t0 = 0;
			int time_steps = 0;
			double dt = 0;

			mutable io::OutGeometryData output_geometry_;
			mutable bool output_sampler_initialized_ = false;

			static bool read_initial_x_from_file(
				const std::string &state_path,
				const std::string &x_name,
				const bool reorder,
				const Eigen::VectorXi &in_node_to_node,
				const int dim,
				Eigen::MatrixXd &x);

			static void rebuild_node_positions(
				const std::vector<basis::ElementBases> &bases,
				const std::vector<int> &node_ids,
				std::vector<RowVectorNd> &positions);
		};
	} // namespace varform
} // namespace polyfem
