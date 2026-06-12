#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <memory>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::varform
{
	class ScalarVarForm : public VarForm
	{
		friend class polyfem::test::VarFormTestAccess;

	public:
		std::string name() const override { return "Scalar"; }

		ScalarVarForm(const std::string &formulation, const Units &units, const json &args, const std::string &out_path);
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		FESpace &legacy_primary_space_dont_use() override { return scalar_space; }
		const FESpace &legacy_primary_space_dont_use() const override { return scalar_space; }
		std::shared_ptr<GeometryMapping> &legacy_primary_geometry_dont_use() override { return scalar_space.geometry; }
		const std::shared_ptr<GeometryMapping> &legacy_primary_geometry_dont_use() const override { return scalar_space.geometry; }
		AssemblyCaches &legacy_primary_caches_dont_use() override { return scalar_caches; }
		const AssemblyCaches &legacy_primary_caches_dont_use() const override { return scalar_caches; }
		VarFormBoundaryState &boundary_state() override { return boundary; }
		const VarFormBoundaryState &boundary_state() const override { return boundary; }

	protected:
		void build_fe_space(mesh::Mesh &mesh, const json &args) override;
		void build_assembler_cache(const mesh::Mesh &mesh, const json &args) override;
		void build_boundary_condition(mesh::Mesh &mesh, const json &args) override;
		// No-op.
		void build_solution_layout() override {};

	private:
		void build_stiffness_mat(StiffnessMatrix &stiffness);

		void solve_problem(Eigen::MatrixXd &sol) override;
		void solve_linear_system(
			const std::unique_ptr<polysolve::linear::Solver> &solver,
			StiffnessMatrix &A,
			Eigen::VectorXd &b,
			const bool compute_spectrum,
			Eigen::MatrixXd &sol);
		void solve_static(Eigen::MatrixXd &sol);
		void solve_transient(Eigen::MatrixXd &sol);

		std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator;

	protected:
		FESpace scalar_space;
		AssemblyCaches scalar_caches;
		VarFormBoundaryState boundary;
	};
} // namespace polyfem::varform
