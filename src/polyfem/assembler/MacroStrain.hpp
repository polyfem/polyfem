#pragma once

#include <polyfem/assembler/Problem.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/JSONUtils.hpp>

namespace polyfem
{
	namespace assembler
	{
		class MacroStrainValue
		{
			bool _is_active = false;
			int _dim;
			Eigen::VectorXi fixed_entry;
			std::array<utils::ExpressionValue, 9> value;

		public:
			MacroStrainValue() = default;

			bool is_active() const { return _is_active; }
			int dim() const { return _dim; }
			const Eigen::VectorXi &get_fixed_entry() const { return fixed_entry; }

			void init(const int dim, const json &param)
			{
				_is_active = true;
				_dim = dim;
				fixed_entry = param["fixed_macro_strain"];
				if (utils::is_param_valid(param, "linear_displacement_offset"))
				{
					json arg = param["linear_displacement_offset"];
					assert(arg.is_array());
#ifndef NDEBUG
					for (const auto &a : arg)
						assert(a.is_array());
#endif
					for (size_t i = 0; i < arg.size(); ++i)
					{
						for (size_t j = 0; j < arg[i].size(); ++j)
						{
							value[i * 3 + j].init(arg[i][j]);
							value[i * 3 + j].set_unit_type("");
						}
					}
				}
			}

			Eigen::MatrixXd eval(const double t) const
			{
				Eigen::MatrixXd strain(_dim, _dim);
				for (int i = 0; i < _dim; i++)
					for (int j = 0; j < _dim; j++)
						strain(i, j) = eval(i, j, t);

				return strain;
			}

			double eval(const int i, const int j, const double t) const
			{
				return value[i * 3 + j](0, 0, 0, t);
			}
		};
	} // namespace assembler
} // namespace polyfem