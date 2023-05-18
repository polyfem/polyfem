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

	template <typename T>
	T AMIPSEnergy::elastic_energy_T(
		const RowVectorNd &p,
		const int el_id,
		const DefGradMatrix<T> &def_grad) const
		{
			using std::pow;

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> J(size(), size());
			for (int i = 0; i < size(); ++i)
				for (int j = 0; j < size(); ++j)
				{
					J(i, j) = T(0);
					for (int k = 0; k < size(); ++k)
						J(i, j) += def_grad(i, k) * canonical_transformation_[el_id](k, j);
				}

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> JtJ;
			JtJ = J.transpose() * J;

			const T val = JtJ.diagonal().sum() / pow(polyfem::utils::determinant(J), 2. / size());

			return val;
		}

	void AMIPSEnergy::add_multimaterial(const int index, const json &params)
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