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
		class BDF;
		class ImplicitTimeIntegrator;
	} // namespace time_integrator

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
			virtual ~VarForm() = default;

			/// @brief Get the name of the variational formulation
			/// @return Name of the variational formulation
			virtual std::string name() const = 0;

			/// @brief Reset the internal state of the variational formulation, e.g. when loading a new mesh
			/// @param args json input arguments, used to determine which data to reset
			void set_args(const json &args)
			{
				this->args = args;
				prepared_ = false;
				output_sampler_initialized_ = false;
			}

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
			void set_time_callback(const std::function<void(int, int, double, double)> &callback) { time_callback = callback; }

			/// @brief Get the problem dimension of the variational formulation, for output purposes
			/// @return Problem dimension
			int problem_dimension() const;
			/// @brief Check if contact is enabled for the variational formulation, for output purposes
			/// @return True if contact is enabled, false otherwise
			virtual bool is_contact_enabled() const { return false; }

			/// @brief Get the output space of the variational formulation, for output purposes
			/// @return Output space
			virtual io::OutputSpace output_space() const = 0;
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
			virtual io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) = 0;

			/// @brief Save the solution to a JSON file, for output purposes
			/// @param solution Solution matrix to save
			/// @param out Output stream to save the solution
			virtual void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const = 0;
			/// @brief 	Save the solution to a JSON file, for output purposes
			/// @param solution
			void save_json(const Eigen::MatrixXd &solution) const;
			virtual void export_data(const Eigen::MatrixXd &solution) const = 0;

		protected:
			std::string resolve_output_path(const std::string &path) const;
			std::string resolve_input_path(const std::string &path, const bool only_if_exists = false) const;

			void set_materials(assembler::Assembler &assembler, const int size) const;
			virtual void reset() = 0;

			virtual void load_mesh(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) = 0;
			void assign_discr_orders(const json &discr_order, const mesh::Mesh &mesh, Eigen::VectorXi &disc_orders);
			virtual void assemble_rhs(const mesh::Mesh &mesh) = 0;
			virtual void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void solve_problem(Eigen::MatrixXd &sol) = 0;
			void prepare();
			QuadratureOrders n_boundary_samples(const int discr_order, const int gdiscr_order) const;

			void build_fe_space(
				mesh::Mesh &mesh,
				const bool iso_parametric,
				const Eigen::VectorXi &disc_orders,
				const std::string &basis_type,
				const std::string &poly_basis_type,
				const assembler::Assembler &space_assembler,
				const int value_dim,
				const int quadrature_order,
				const int mass_quadrature_order,
				const bool use_corner_quadrature,
				const int n_harmonic_samples,
				const int integral_constraints,
				FESpace &space,
				VarFormBoundaryState &boundary,
				std::shared_ptr<GeometryMapping> geometry = nullptr);

		private:
			void build_polygonal_basis(
				const mesh::Mesh &mesh,
				const std::string &poly_basis_type,
				const assembler::Assembler &space_assembler,
				bool iso_parametric,
				const int quadrature_order,
				const int mass_quadrature_order,
				const int n_harmonic_samples,
				const int integral_constraints,
				FESpace &space,
				VarFormBoundaryState &boundary);

			void build_node_mapping(
				const mesh::Mesh &mesh,
				const std::string &basis_type,
				const FESpace &space,
				Eigen::VectorXi &space_in_node_to_node,
				Eigen::VectorXi &space_in_primitive_to_primitive) const;

		protected:
			virtual void build_rhs_assembler() = 0;

			std::shared_ptr<time_integrator::BDF> make_bdf_time_integrator() const;

			void save_step_state(
				const double t0,
				const double dt,
				const int t,
				const time_integrator::ImplicitTimeIntegrator *time_integrator,
				const bool rest_mesh_written = false) const;

			void ensure_output_sampler() const;
			void save_restart_json(const double t0, const double dt, const int t, const bool rest_mesh_written) const;
			void save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &solution) const;
			void save_subsolve(const int i, const int t, const Eigen::MatrixXd &solution) const;
			int output_file_index(const int t) const;
			void notify_time_step(const int t, const int time_steps, const double t0, const double dt) const;

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
