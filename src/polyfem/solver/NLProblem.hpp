#pragma once

#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/assembler/PeriodicBoundary.hpp>
#include <polyfem/solver/forms/lagrangian/AugmentedLagrangianForm.hpp>

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
			const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms);

	public:
		NLProblem(const int full_size,
				  const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc,
				  const double t,
				  const std::vector<std::shared_ptr<Form>> &forms,
				  const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms);
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

		virtual TVector full_to_reduced(const TVector &full) const;
		virtual TVector full_to_reduced_grad(const TVector &full) const;
		virtual void full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const;
		virtual TVector reduced_to_full(const TVector &reduced) const;

	protected:
		virtual Eigen::MatrixXd constraint_values(const TVector &reduced) const;

		std::vector<int> constraint_nodes_;

		const int full_size_; ///< Size of the full problem
		int reduced_size_;    ///< Size of the reduced problem

		std::shared_ptr<utils::PeriodicBoundary> periodic_bc_;

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

	private:
		std::vector<std::shared_ptr<AugmentedLagrangianForm>> penalty_forms_;

		void setup_constrain_nodes();

		template <class FullMat, class ReducedMat>
		void full_to_reduced_aux(const std::vector<int> &constraint_nodes, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced) const;

		template <class ReducedMat, class FullMat>
		void reduced_to_full_aux(const std::vector<int> &constraint_nodes, const int full_size, const int reduced_size, const ReducedMat &reduced, const Eigen::MatrixXd &rhs, FullMat &full) const;

		template <class FullMat, class ReducedMat>
		void full_to_reduced_aux_grad(const std::vector<int> &constraint_nodes, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced) const;
	};
} // namespace polyfem::solver
