#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/Problem.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/io/OutputData.hpp>
#include <polyfem/io/OutData.hpp>
#include <polyfem/io/OutStatsData.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/varforms/FESpace.hpp>

#include <Eigen/Dense>

#include <functional>
#include <iosfwd>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

namespace polyfem
{
	namespace time_integrator
	{
		class ImplicitTimeIntegrator;
	}

	namespace test
	{
		class VarFormTestAccess;
	}

	namespace varform
	{
		class VarForm
		{
			friend class polyfem::test::VarFormTestAccess;

		public:
			VarForm(const Units &units, const json &args, const std::string &out_path);

			virtual ~VarForm() = default;

			/// @brief Get the name of the variational formulation
			/// @return Name of the variational formulation
			virtual std::string name() const = 0;

			/// @brief Update the args and mark as unprepared, e.g. when loading a new mesh
			/// @param args json input arguments
			void set_args(const json &args)
			{
				this->args = args;
				prepared_ = false;
				output_sampler_initialized_ = false;
			}

			/// @brief Set the mesh for the variational formulation
			/// @param mesh unique pointer to the mesh to use for the formulation
			void set_mesh(std::unique_ptr<mesh::Mesh> mesh, const double loading_mesh_time = 0);

			/// @brief Solve the variational formulation and store the solution in the given matrix
			/// @param sol matrix to store the solution
			void solve(Eigen::MatrixXd &sol);
			void set_time_callback(const std::function<void(int, int, double, double)> &callback) { time_callback = callback; }

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

		protected:
			std::string resolve_output_path(const std::string &path) const;
			std::string resolve_input_path(const std::string &path, const bool only_if_exists = false) const;

			virtual void set_materials(assembler::Assembler &assembler) const;

			virtual void load_mesh(const mesh::Mesh &mesh, const json &args);
			virtual void build_basis(mesh::Mesh &mesh, const json &args);
			virtual void build_assembler_cache(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void build_boundary_condition(mesh::Mesh &mesh, const json &args) = 0;
			virtual void assemble_rhs(const mesh::Mesh &mesh, const json &args);
			virtual void assemble_mass_mat(const mesh::Mesh &mesh, const json &args);
			virtual void solve_problem(Eigen::MatrixXd &sol) = 0;
			void prepare();

			/// @brief Get a constant reference to the geometry mapping bases.
			/// @return A constant reference to the geometry mapping bases.
			const std::vector<basis::ElementBases> &geom_bases() const
			{
				return *primary_geometry()->bases;
			}

			// primary_xxx method is for lagacy compatibility only!!!!!
			// NEVER EVER USE IT IN NEW CODE.
			virtual FESpace &primary_space() = 0;
			virtual const FESpace &primary_space() const = 0;
			virtual std::shared_ptr<GeometryMapping> &primary_geometry() = 0;
			virtual const std::shared_ptr<GeometryMapping> &primary_geometry() const = 0;
			virtual AssemblyCaches &primary_caches() = 0;
			virtual const AssemblyCaches &primary_caches() const = 0;
			virtual VarFormBoundaryState &boundary_state() = 0;
			virtual const VarFormBoundaryState &boundary_state() const = 0;

			void build_polygonal_basis(const mesh::Mesh &mesh);
			void build_node_mapping(const mesh::Mesh &mesh, const json &args);
			std::vector<int> primitive_to_node() const;
			std::vector<int> node_to_primitive() const;

			virtual void build_rhs_assembler();

			void initial_solution(Eigen::MatrixXd &solution) const;

			void save_step_state(
				const double t0,
				const double dt,
				const int t,
				const time_integrator::ImplicitTimeIntegrator *time_integrator) const;

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
			void notify_time_step(const int t) const;

			io::OutGeometryData::ExportOptions export_options(const io::OutputSpace &space) const;
			io::OutputFieldFunction output_field_function(const Eigen::MatrixXd &solution, const io::OutGeometryData::ExportOptions &opts) const;

			/// current problem, it contains rhs and bc
			std::shared_ptr<assembler::Problem> problem;
			Units units;
			json args;

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

			/// nodes on the boundary of polygonal elements, used for harmonic bases
			std::map<int, basis::InterfaceData> poly_edge_to_data;

			/// Input nodes (including high-order) to polyfem nodes, only for isoparametric
			Eigen::VectorXi in_node_to_node;
			/// maps input vertices/edges/faces/cells to polyfem vertices/edges/faces/cells
			Eigen::VectorXi in_primitive_to_primitive;

			std::shared_ptr<assembler::RhsAssembler> rhs_assembler;

			/// Mass matrix, it is computed only for time dependent problems
			StiffnessMatrix mass;
			StiffnessMatrix pure_mass;
			/// average system mass, used for contact with IPC
			double avg_mass = 0;
			Eigen::MatrixXd rhs;

			double t0 = 0;
			int time_steps = 0;
			double dt = 0;
			std::function<void(int, int, double, double)> time_callback;

			mutable io::OutGeometryData output_geometry_;
			mutable bool output_sampler_initialized_ = false;
			bool prepared_ = false;

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
