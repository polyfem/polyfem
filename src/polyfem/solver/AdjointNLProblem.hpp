#pragma once

#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/solver/AdjointForm.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	class AdjointNLProblem : public FullNLProblem
	{
	public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

		AdjointNLProblem(std::vector<std::shared_ptr<Form>> &forms);

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
		void update_lagging(const TVector &x, const int iter_num) override;
	};
} // namespace polyfem::solver
