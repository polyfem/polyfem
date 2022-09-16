#pragma once

#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/State.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>

namespace polyfem::solver
{
	class NLProblem : public FullNLProblem
	{
	public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

		NLProblem(const State &state, const assembler::RhsAssembler &rhs_assembler, const double t, std::vector<std::shared_ptr<Form>> &forms);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;
		void hessian(const TVector &x, THessian &hessian) override;

		bool is_step_valid(const TVector &x0, const TVector &x1) const override;
		bool is_step_collision_free(const TVector &x0, const TVector &x1) const override;
		double max_step_size(const TVector &x0, const TVector &x1) const override;

		void line_search_begin(const TVector &x0, const TVector &x1) override;
		void post_step(const int iter_num, const TVector &x) override;

		void solution_changed(const TVector &new_x) override;

		void init_lagging(const TVector &x) override;
		bool update_lagging(const TVector &x, const int iter_num) override;

		// --------------------------------------------------------------------

		void update_quantities(const double t, const TVector &x);

		int full_size() const { return full_size_; }
		int reduced_size() const { return reduced_size_; }

		void use_full_size() { current_size_ = FULL_SIZE; }
		void use_reduced_size() { current_size_ = REDUCED_SIZE; }

		TVector full_to_reduced(const TVector &full) const;
		TVector reduced_to_full(const TVector &reduced) const;

		void set_apply_DBC(const TVector &x, const bool val);

	private:
		const State &state_;
		const assembler::RhsAssembler &rhs_assembler_;
		double t_;

		const int full_size_;    ///< Size of the full problem
		const int reduced_size_; ///< Size of the reduced problem

		enum CurrentSize
		{
			FULL_SIZE,
			REDUCED_SIZE
		};
		CurrentSize current_size_; ///< Current size of the problem (either full or reduced size)
		int current_size() const
		{
			return current_size_ == FULL_SIZE ? full_size() : reduced_size();
		}

		template <class FullMat, class ReducedMat>
		static void full_to_reduced_aux(const State &state, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced);

		template <class ReducedMat, class FullMat>
		static void reduced_to_full_aux(const State &state, const int full_size, const int reduced_size, const ReducedMat &reduced, const Eigen::MatrixXd &rhs, FullMat &full);
	};
} // namespace polyfem::solver
