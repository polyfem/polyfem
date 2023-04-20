#include "MultiModel.hpp"

// #include <polyfem/basis/Basis.hpp>
// #include <polyfem/autogen/auto_elasticity_rhs.hpp>

// #include <igl/Timer.h>

namespace polyfem::assembler
{
	void MultiModel::set_size(const int size)
	{
		Assembler::set_size(size);

		saint_venant_.set_size(size);
		neo_hookean_.set_size(size);
		linear_elasticity_.set_size(size);

		hooke_.set_size(size);
		mooney_rivlin_elasticity_.set_size(size);
		unconstrained_ogden_elasticity_.set_size(size);
		incompressible_ogden_elasticity_.set_size(size);
	}

	void MultiModel::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		saint_venant_.add_multimaterial(index, params, units);
		neo_hookean_.add_multimaterial(index, params, units);
		linear_elasticity_.add_multimaterial(index, params, units);

		hooke_.add_multimaterial(index, params, units);
		mooney_rivlin_elasticity_.add_multimaterial(index, params, units);
		unconstrained_ogden_elasticity_.add_multimaterial(index, params, units);
		incompressible_ogden_elasticity_.add_multimaterial(index, params, units);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	MultiModel::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;
		assert(false);

		return res;
	}

	Eigen::VectorXd
	MultiModel::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		const int el_id = data.vals.element_id;
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			return saint_venant_.assemble_gradient(data);
		else if (model == "NeoHookean")
			return neo_hookean_.assemble_gradient(data);
		else if (model == "LinearElasticity")
			return linear_elasticity_.assemble_gradient(data);
		else if (model == "HookeLinearElasticity")
			return hooke_.assemble_gradient(data);
		else if (model == "MooneyRivlin")
			return mooney_rivlin_elasticity_.assemble_gradient(data);
		else if (model == "UnconstrainedOgden")
			return unconstrained_ogden_elasticity_.assemble_gradient(data);
		else if (model == "IncompressibleOgden")
			return incompressible_ogden_elasticity_.assemble_gradient(data);
		else
		{
			assert(false);
			return Eigen::VectorXd(0, 0);
		}
	}

	Eigen::MatrixXd
	MultiModel::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const int el_id = data.vals.element_id;
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			return saint_venant_.assemble_hessian(data);
		else if (model == "NeoHookean")
			return neo_hookean_.assemble_hessian(data);
		else if (model == "LinearElasticity")
			return linear_elasticity_.assemble_hessian(data);
		else if (model == "HookeLinearElasticity")
			return hooke_.assemble_hessian(data);
		else if (model == "MooneyRivlin")
			return mooney_rivlin_elasticity_.assemble_hessian(data);
		else if (model == "UnconstrainedOgden")
			return unconstrained_ogden_elasticity_.assemble_hessian(data);
		else if (model == "IncompressibleOgden")
			return incompressible_ogden_elasticity_.assemble_hessian(data);
		else
		{
			assert(false);
			return Eigen::MatrixXd(0, 0);
		}
	}

	double MultiModel::compute_energy(const NonLinearAssemblerData &data) const
	{
		const int el_id = data.vals.element_id;
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			return saint_venant_.compute_energy(data);
		else if (model == "NeoHookean")
			return neo_hookean_.compute_energy(data);
		else if (model == "LinearElasticity")
			return linear_elasticity_.compute_energy(data);
		else if (model == "HookeLinearElasticity")
			return hooke_.compute_energy(data);
		else if (model == "MooneyRivlin")
			return mooney_rivlin_elasticity_.compute_energy(data);
		else if (model == "UnconstrainedOgden")
			return unconstrained_ogden_elasticity_.compute_energy(data);
		else if (model == "IncompressibleOgden")
			return incompressible_ogden_elasticity_.compute_energy(data);
		else
		{
			assert(false);
			return 0;
		}
	}

	void MultiModel::assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			saint_venant_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else if (model == "NeoHookean")
			neo_hookean_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else if (model == "LinearElasticity")
			linear_elasticity_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else if (model == "HookeLinearElasticity")
			return hooke_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else if (model == "MooneyRivlin")
			return mooney_rivlin_elasticity_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else if (model == "UnconstrainedOgden")
			return unconstrained_ogden_elasticity_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else if (model == "IncompressibleOgden")
			return incompressible_ogden_elasticity_.assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, all_size, type, all, fun);
		else
		{
			assert(false);
		}
	}

	std::map<std::string, Assembler::ParamFunc> MultiModel::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		// TODO

		return res;
	}
} // namespace polyfem::assembler
