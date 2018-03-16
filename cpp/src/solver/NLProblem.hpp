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

		NLProblem(const RhsAssembler &rhs_assembler);
		NLProblem(const RhsAssembler &rhs_assembler, const int full_size, const int reduced_size);

		TVector initial_guess();

		double value(const TVector &x);
		void gradient(const TVector &x, TVector &gradv);
		void hessian(const TVector &x, THessian &hessian);


		template<class FullMat, class ReducedMat>
		static void full_to_reduced_aux(const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced)
		{
			using namespace poly_fem;

			assert(full.size() == full_size);
			assert(full.cols() == 1);
			reduced.resize(reduced_size, 1);

			long j = 0;
			size_t k = 0;
			for(long i = 0; i < full.size(); ++i)
			{
				if(State::state().boundary_nodes[k] == i)
				{
					++k;
					continue;
				}

				reduced(j++) = full(i);
			}
		}

		template<class ReducedMat, class FullMat>
		static void reduced_to_full_aux(const int full_size, const int reduced_size, const ReducedMat &reduced, const bool set_zero, FullMat &full)
		{
			using namespace poly_fem;

			assert(reduced.size() == reduced_size);
			assert(reduced.cols() == 1);
			full.resize(full_size, 1);

			long j = 0;
			size_t k = 0;
			for(long i = 0; i < full.size(); ++i)
			{
				if(State::state().boundary_nodes[k] == i)
				{
					++k;
					full(i) = set_zero ? 0 : State::state().rhs(i);
					continue;
				}

				full(i) = reduced(j++);
			}
		}

	private:
		const AssemblerUtils &assembler;
		const RhsAssembler &rhs_assembler;

		const int full_size, reduced_size;

		void full_to_reduced(const Eigen::MatrixXd &full, TVector &reduced);
		void reduced_to_full(const TVector &reduced, const bool set_zero, Eigen::MatrixXd &full);
	};
}
