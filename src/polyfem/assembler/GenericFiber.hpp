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
		virtual void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;
		virtual void set_size(const int size) override;

		std::map<std::string, Assembler::ParamFunc> parameters() const override;

	protected:
		template <typename T>
		T I4Bar(const RowVectorNd &p,
				const double t,
				const int el_id,
				const DefGradMatrix<T> &def_grad) const
		{
			return I4Bar_generic(p, t, el_id, def_grad, false, true);
		}
		template <typename T>
		T I4Bar_with_norm(const RowVectorNd &p,
						  const double t,
						  const int el_id,
						  const DefGradMatrix<T> &def_grad) const
		{
			return I4Bar_generic(p, t, el_id, def_grad, true, true);
		}

		template <typename T>
		T I4Bar_generic(const RowVectorNd &p,
						const double t,
						const int el_id,
						const DefGradMatrix<T> &def_grad,
						const bool normalize,
						const bool isocoric) const
		{
			const T J = polyfem::utils::determinant(def_grad);
			const T powJ = isocoric ? (this->size() == 2 ? J : pow(J, 2.0 / 3.0)) : T(1);
			const auto Cbar = (def_grad.transpose() * def_grad / powJ).eval();

			auto a_tmp = fiber_direction_(p, p, t, el_id);
			assert((a_tmp.rows() == this->size() && a_tmp.cols() == 1) || (a_tmp.rows() == this->size() && a_tmp.cols() == this->size()));
			const bool is_a_vector = a_tmp.cols() == 1;

			if (normalize)
			{
				if (is_a_vector)
					a_tmp.normalize();
				else
					a_tmp /= a_tmp.trace();
			}

			const int d = a_tmp.rows();
			if (is_a_vector)
			{
				// compute a^T * Cbar * a
				T res = T(0);
				for (int i = 0; i < d; ++i)
					for (int j = 0; j < d; ++j)
						res += T(a_tmp(i)) * Cbar(i, j) * T(a_tmp(j));

				return res;
			}
			else
			{
				// compute trace(Cbar * a)
				T res = T(0);
				for (int i = 0; i < d; ++i)
					for (int j = 0; j < d; ++j)
						res += Cbar(i, j) * T(a_tmp(j, i));
				return res;
			}

			// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> a(a_tmp.rows(), a_tmp.cols());
			// for (int i = 0; i < a.rows(); ++i)
			// 	for (int j = 0; j < a.cols(); ++j)
			// 		a(i, j) = T(a_tmp(i, j));

			// const T tmp = is_a_vector ? (a.transpose() * Cbar * a)(0, 0) : (Cbar * a).trace();
			// // assert(tmp.rows() == 1 && tmp.cols() == 1);
			// return tmp;
		}

		FiberDirection fiber_direction_;
	};
} // namespace polyfem::assembler