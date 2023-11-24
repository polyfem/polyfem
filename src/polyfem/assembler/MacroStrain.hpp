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
			std::array<utils::ExpressionValue, 9> value;

        public:
            MacroStrainValue() = default;

            void init(const json& param)
            {
                if (utils::is_param_valid(param, "linear_displacement_offset"))
                {
                    json arg = param["linear_displacement_offset"];
                    assert(arg.is_array() && arg.size() > 0 && arg[0].is_array());
                    
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

            Eigen::MatrixXd eval(const int dim, const double t) const
            {
                Eigen::MatrixXd strain(dim, dim);
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        strain(i, j) = eval(i, j, t);
                
                return strain;
            }

			double eval(const int i, const int j, const double t) const
            {
                return value[i * 3 + j](0,0,0,t);
            }
		};
    }
}