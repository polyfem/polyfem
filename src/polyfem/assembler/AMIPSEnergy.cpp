#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{

	AMIPSEnergy::AMIPSEnergy()
	{
		canonical_transformation_.resize(0);
	}

	std::map<std::string, Assembler::ParamFunc> AMIPSEnergy::parameters() const
	{
		return std::map<std::string, ParamFunc>();
	}

	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &)
	{
		if (params.contains("canonical_transformation"))
		{
			canonical_transformation_.reserve(params["canonical_transformation"].size());
			for (int i = 0; i < params["canonical_transformation"].size(); ++i)
			{
				Eigen::MatrixXd transform_matrix(size(), size());
				for (int j = 0; j < size(); ++j)
					for (int k = 0; k < size(); ++k)
						transform_matrix(j, k) = params["canonical_transformation"][i][j][k];
				canonical_transformation_.push_back(transform_matrix);
			}
		}
	}

} // namespace polyfem::assembler