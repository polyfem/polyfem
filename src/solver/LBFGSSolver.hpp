// L-BFGS solver
//
// Based LBFGSpp under MIT License:
//
// Copyright (c) 1990 Jorge Nocedal
// Copyright (c) 2007-2010 Naoaki Okazaki
// Copyright (c) 2016-2021 Yixuan Qiu
// Copyright (c) 2018-2021 Dirk Toewe
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/NonlinearSolver.hpp>
#include <polysolve/LinearSolver.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <LBFGSpp/BFGSMat.h>

namespace cppoptlib
{
	template <typename ProblemType>
	class LBFGSSolver : public NonlinearSolver<ProblemType>
	{
	public:
		using Superclass = NonlinearSolver<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		LBFGSSolver(const json &solver_params)
			: Superclass(solver_params)
		{
		}

		std::string name() const override { return "LBFGSSolver"; }

	protected:
		LBFGSpp::BFGSMat<Scalar> m_bfgs; // Approximation to the Hessian matrix

		/// The number of corrections to approximate the inverse Hessian matrix.
		/// The L-BFGS routine stores the computation results of previous \ref m
		/// iterations to approximate the inverse Hessian matrix of the current
		/// iteration. This parameter controls the size of the limited memories
		/// (corrections). The default value is \c 6. Values less than \c 3 are
		/// not recommended. Large values will result in excessive computing time.
		int m_history_size = 6;

		TVector m_prev_x;    // Previous x
		TVector m_prev_grad; // Previous gradient

		void reset(ProblemType &objFunc, TVector &x) override
		{
			Superclass::reset(objFunc, x);

			m_bfgs.reset(x.size(), m_history_size);
			m_prev_x.resize(x.size());
			m_prev_grad.resize(x.size());
		}

		virtual void compute_search_direction(ProblemType &objFunc, const TVector &x, const TVector &grad, TVector &direction)
		{
			if (this->m_current.iterations == 0 || this->line_search_failed)
			{
				// Use gradient descent in the first iteration or if the previous iteration failed
				direction = -grad;
			}
			else
			{
				// Update s and y
				// s_{i+1} = x_{i+1} - x_i
				// y_{i+1} = g_{i+1} - g_i
				m_bfgs.add_correction(x - m_prev_x, grad - m_prev_grad);

				// Recursive formula to compute d = -H * g
				m_bfgs.apply_Hv(grad, -Scalar(1), direction);
			}

			m_prev_x = x;
			m_prev_grad = grad;
		}
	};
} // namespace cppoptlib
