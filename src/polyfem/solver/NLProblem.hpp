#pragma once

#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/assembler/PeriodicBoundary.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>

namespace polysolve::linear
{
	class Solver;
}

namespace polyfem::solver
{
	class NLProblem : public FullNLProblem
	{
	public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

	protected:
		NLProblem(
			const int full_size,
			const std::vector<std::shared_ptr<Form>> &forms,
			const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
			const std::shared_ptr<polysolve::linear::Solver> &solver);

	public:
		NLProblem(const int full_size,
				  const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc,
				  const double t,
				  const std::vector<std::shared_ptr<Form>> &forms,
				  const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
				  const std::shared_ptr<polysolve::linear::Solver> &solver);
		virtual ~NLProblem() = default;

		virtual double value(const TVector &x) override;
		virtual void gradient(const TVector &x, TVector &gradv) override;
		virtual void hessian(const TVector &x, THessian &hessian) override;

		virtual bool is_step_valid(const TVector &x0, const TVector &x1) override;
		virtual bool is_step_collision_free(const TVector &x0, const TVector &x1) override;
		virtual double max_step_size(const TVector &x0, const TVector &x1) override;
		void line_search_begin(const TVector &x0, const TVector &x1) override;
		virtual void post_step(const polysolve::nonlinear::PostStepData &data) override;

		void solution_changed(const TVector &new_x) override;

		void init_lagging(const TVector &x) override;
		void update_lagging(const TVector &x, const int iter_num) override;

		// --------------------------------------------------------------------

		virtual void update_quantities(const double t, const TVector &x);

		int full_size() const { return full_size_; }
		int reduced_size() const { return reduced_size_; }

		void use_full_size() { current_size_ = CurrentSize::FULL_SIZE; }
		void use_reduced_size() { current_size_ = CurrentSize::REDUCED_SIZE; }

		TVector full_to_reduced(const TVector &full) const;
		virtual TVector full_to_reduced_grad(const TVector &full) const;
		TVector reduced_to_full(const TVector &reduced) const;

		void full_hessian_to_reduced_hessian(StiffnessMatrix &hessian) const;

		double normalize_forms() override;

		void update_constraint_values();

	protected:
		const int full_size_; ///< Size of the full problem
		int reduced_size_;    ///< Size of the reduced problem

		enum class CurrentSize
		{
			FULL_SIZE,
			REDUCED_SIZE
		};
		CurrentSize current_size_; ///< Current size of the problem (either full or reduced size)
		int current_size() const
		{
			return current_size_ == CurrentSize::FULL_SIZE ? full_size() : reduced_size();
		}

		double t_;

	protected:
		std::vector<std::shared_ptr<AugmentedLagrangianForm>> penalty_forms_;
		// The decomposion comes from sec 1.3 of https://www.cs.cornell.edu/courses/cs6241/2021sp/meetings/nb-2021-03-11.pdf
		StiffnessMatrix Q1_;                                         ///< Q1 block of the QR decomposition of the constraints matrix
		StiffnessMatrix Q2_;                                         ///< Q2 block of the QR decomposition of the constraints matrix
		StiffnessMatrix Q2t_;                                        ///< Q2 transpose
		StiffnessMatrix R1_;                                         ///< R1 block of the QR decomposition of the constraints matrix
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P_; ///< Permutation matrix of the QR decomposition of the constraints matrix
		TVector Q1R1iTb_;                                            ///< Q1_ * (R1_.transpose().triangularView<Eigen::Upper>().solve(constraint_values_))
		std::shared_ptr<polysolve::linear::Solver> solver_;

		std::shared_ptr<FullNLProblem> penalty_problem_;
		int num_penalty_constraints_;

		void setup_constraints();

	};
} // namespace polyfem::solver
