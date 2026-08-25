#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

namespace polyfem::varform
{
	class OperatorSplittingVarForm : public VarForm
	{
	public:
		std::string name() const override { return "OperatorSplitting"; }

		void init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path) override;
		void save_json(const Eigen::MatrixXd &solution, std::ostream &out) const override;
		void export_data(const Eigen::MatrixXd &solution) const override;
		io::OutputSpace output_space() const override;
		io::OutStatsData compute_errors(const Eigen::MatrixXd &solution) override;

		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		void reset() override;
		void load_mesh(const mesh::Mesh &mesh, const json &args) override;
		void build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args) override;
		void assemble_rhs(const mesh::Mesh &mesh) override;
		void assemble_mass_mat(const mesh::Mesh &mesh, const json &args) override;
		void solve_problem(Eigen::MatrixXd &sol) override;
		void build_rhs_assembler() override;

	private:
		int primary_ndof() const;
		int pressure_block_size() const;
		int stacked_ndof() const;

		void prepare_initial_solution(Eigen::MatrixXd &sol) const;
		void split_solution(const Eigen::MatrixXd &stacked, Eigen::MatrixXd &primary, Eigen::MatrixXd &pressure) const;
		void stack_solution(const Eigen::MatrixXd &primary, const Eigen::MatrixXd &pressure, Eigen::MatrixXd &stacked) const;

		FESpace space_;
		FESpace pressure_space_;

		VarFormBoundaryState boundary_;
		VarFormBoundaryState pressure_boundary_;

		assembler::AssemblyValsCache ass_vals_cache_;
		assembler::AssemblyValsCache pressure_ass_vals_cache_;
		assembler::AssemblyValsCache mass_ass_vals_cache_;

		std::shared_ptr<assembler::RhsAssembler> rhs_assembler_;

		StiffnessMatrix mass_;
		Eigen::MatrixXd rhs_;

		std::shared_ptr<assembler::Assembler> primary_assembler_ = nullptr;
		std::shared_ptr<assembler::Mass> mass_assembler_ = nullptr;
		std::shared_ptr<assembler::MixedAssembler> mixed_assembler_ = nullptr;
		std::shared_ptr<assembler::Assembler> pressure_assembler_ = nullptr;

		bool use_avg_pressure = true;
		double avg_mass_ = 0;
		double t0 = 0;
		int time_steps = 0;
		double dt = 0;
	};
} // namespace polyfem::varform
