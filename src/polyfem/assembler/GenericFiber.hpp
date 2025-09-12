#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	class GenericFiber : public GenericElastic<GenericFiber>
	{
	public:
		GenericFiber();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;
		void set_size(const int size) override;

	protected:
		template <typename T>
		DefGradMatrix<T> I4Bar(RowVectorNd &p,
							   const double t,
							   const int el_id,
							   const DefGradMatrix<T> &def_grad)
		{
			const T J = polyfem::utils::determinant(def_grad);
			const auto Cbar = (def_grad.transpose() * def_grad / pow(J, 2.0 / 3.0)).eval();

			return fiber_direction_(p, p, t, el_id).transpose() * Cbar * fiber_direction_(p, p, t, el_id);
		}

		FiberDirection fiber_direction_;
	};
} // namespace polyfem::assembler