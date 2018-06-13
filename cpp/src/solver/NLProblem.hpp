#pragma once


#include "AssemblerUtils.hpp"
#include "RhsAssembler.hpp"
#include "State.hpp"

#include <cppoptlib/problem.h>
#include <Eigen/Sparse>



namespace poly_fem
{
	class NLProblem : public cppoptlib::Problem<double> {
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;
		typedef Eigen::SparseMatrix<double> THessian;

		NLProblem(const RhsAssembler &rhs_assembler, const double t);
		NLProblem(const RhsAssembler &rhs_assembler, const double t, const int full_size, const int reduced_size);

		TVector initial_guess();

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
		void hessian(const TVector &x, THessian &hessian);
#pragma clang diagnostic pop


		template<class FullMat, class ReducedMat>
		static void full_to_reduced_aux(const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced)
		{
			using namespace poly_fem;

			assert(full.size() == full_size);
			assert(full.cols() == 1);
			reduced.resize(reduced_size, 1);

			long j = 0;
			size_t k = 0;
			for(int i = 0; i < full.size(); ++i)
			{
				if(k < State::state().boundary_nodes.size() && State::state().boundary_nodes[k] == i)
				{
					++k;
					continue;
				}

				reduced(j++) = full(i);
			}
			assert(j == reduced.size());
		}

		template<class ReducedMat, class FullMat>
		static void reduced_to_full_aux(const int full_size, const int reduced_size, const ReducedMat &reduced, Eigen::MatrixXd &rhs, FullMat &full)
		{
			using namespace poly_fem;

			assert(reduced.size() == reduced_size);
			assert(reduced.cols() == 1);
			full.resize(full_size, 1);

			long j = 0;
			size_t k = 0;
			for(int i = 0; i < full.size(); ++i)
			{
				if(k < State::state().boundary_nodes.size() && State::state().boundary_nodes[k] == i)
				{
					++k;
					full(i) = rhs(i);
					continue;
				}

				full(i) = reduced(j++);
			}

			assert(j == reduced.size());
		}

	private:
		AssemblerUtils &assembler;
		const RhsAssembler &rhs_assembler;
		Eigen::MatrixXd current_rhs;

		const int full_size, reduced_size;
		const double t;
		bool rhs_computed;

		void full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced) const;
		void reduced_to_full(const TVector &reduced, Eigen::MatrixXd &full);
	};
}
