#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Problem.hpp>

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/io/OutData.hpp>

#include <Eigen/Dense>

#include <memory>
#include <map>
#include <vector>
#include <string>

namespace polyfem
{
	namespace varform
	{
		class VarForm
		{
		public:
			virtual ~VarForm() = default;

			virtual void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path);
			virtual void load_mesh(const mesh::Mesh &mesh, const json &args) = 0;
			virtual void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) = 0;
			virtual void solve(Eigen::MatrixXd &sol) = 0;

		protected:
			std::string resolve_output_path(const std::string &path) const;

			/// current problem, it contains rhs and bc
			std::shared_ptr<assembler::Problem> problem;
			Units units;

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
		};
	} // namespace varform
} // namespace polyfem
