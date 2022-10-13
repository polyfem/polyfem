#include "MooneyRivlinElasticity.hpp"

#include <polyfem/basis/Basis.hpp>

namespace polyfem::assembler
{
	void MooneyRivlinElasticity::add_multimaterial(const int index, const json &params)
	{
		for (int i = c1_.size(); i <= index; ++i)
		{
			c1_.emplace_back();
			c2_.emplace_back();
			k_.emplace_back();
		}

		if (params.count("c1"))
			c1_[index].init(params["c1"]);
		if (params.count("c2"))
			c2_[index].init(params["c2"]);
		if (params.count("k"))
			k_[index].init(params["k"]);
	}

	void MooneyRivlinElasticity::stress_from_disp_grad(
		const int size,
		const RowVectorNd &p,
		const int el_id,
		const Eigen::MatrixXd &displacement_grad,
		Eigen::MatrixXd &stress_tensor) const
	{
		assert(c1_.size() == 1 || el_id < c1_.size());

		const double x = p(0);
		const double y = p(1);
		const double z = size == 3 ? p(2) : 0;
		const double t = 0; // TODO

		const auto &tmp_c1 = c1_.size() == 1 ? c1_[0] : c1_[el_id];
		const auto &tmp_c2 = c2_.size() == 1 ? c2_[0] : c2_[el_id];
		const auto &tmp_k = k_.size() == 1 ? k_[0] : k_[el_id];

		const double c1 = tmp_c1(x, y, z, t, el_id);
		const double c2 = tmp_c2(x, y, z, t, el_id);
		const double k = tmp_k(x, y, z, t, el_id);

		const Eigen::MatrixXd FmT = displacement_grad.inverse().transpose();

		// stress = 2*c1*F + 4*c2*FF^{T}F + k*lnJ*F^{-T}
		stress_tensor = 2 * c1 * displacement_grad + 4 * c2 * displacement_grad * displacement_grad.transpose() * displacement_grad + k * std::log(displacement_grad.determinant()) * FmT;
	}
} // namespace polyfem::assembler