#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	template <typename FiberModel>
	class GenericFiber : public GenericElastic<FiberModel>
	{
	public:
		GenericFiber();
		virtual ~GenericFiber() = default;

		// sets material params
		virtual void add_multimaterial(const int index, const json &params, const Units &units) override;
		virtual void set_size(const int size) override;

	protected:
		template <typename T>
		T I4Bar(const RowVectorNd &p,
				const double t,
				const int el_id,
				const DefGradMatrix<T> &def_grad) const
		{
			const T J = polyfem::utils::determinant(def_grad);
			const auto Cbar = (def_grad.transpose() * def_grad / pow(J, 2.0 / 3.0)).eval();

			const auto a_tmp = fiber_direction_(p, p, t, el_id);
			assert(a_tmp.rows() == this->size() && a_tmp.cols() == 1);
			Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> a(a_tmp.rows(), a_tmp.cols());
			for (int i = 0; i < a.size(); ++i)
				a(i) = T(a_tmp(i));

			const Eigen::Matrix<T, 1, 1> tmp = a.transpose() * Cbar * a;
			assert(tmp.rows() == 1 && tmp.cols() == 1);
			return tmp(0, 0);
		}

		FiberDirection fiber_direction_;
	};
} // namespace polyfem::assembler