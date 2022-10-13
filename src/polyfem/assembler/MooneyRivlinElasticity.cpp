#include "MooneyRivlinElasticity.hpp"

#include <polyfem/basis/Basis.hpp>

namespace polyfem::assembler
{
	MooneyRivlinElasticity::MooneyRivlinElasticity()
		: c1_("c1"), c2_("c2"), k_("k")
	{
	}

	void MooneyRivlinElasticity::add_multimaterial(const int index, const json &params)
	{
		c1_.add_multimaterial(index, params);
		c2_.add_multimaterial(index, params);
		k_.add_multimaterial(index, params);
	}

	void MooneyRivlinElasticity::stress_from_disp_grad(
		const int size,
		const RowVectorNd &p,
		const int el_id,
		const Eigen::MatrixXd &displacement_grad,
		Eigen::MatrixXd &stress_tensor) const
	{

		const double t = 0; // TODO

		const double c1 = c1_(p, t, el_id);
		const double c2 = c2_(p, t, el_id);
		const double k = k_(p, t, el_id);

		auto def_grad = disp_grad;
		for (int d = 0; d < stress_tensor.rows(); ++d)
			def_grad(d, d) += 1;

		const Eigen::MatrixXd FmT = def_grad.inverse().transpose();

		// stress = 2*c1*F + 4*c2*FF^{T}F + k*lnJ*F^{-T}
		stress_tensor = 2 * c1 * def_grad + 4 * c2 * def_grad * def_grad.transpose() * def_grad + k * std::log(def_grad.determinant()) * FmT;
	}
} // namespace polyfem::assembler