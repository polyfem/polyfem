// #define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <polyfem/MultiModel.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/auto_elasticity_rhs.hpp>

#include <igl/Timer.h>

namespace polyfem
{

	void MultiModel::set_size(const int size)
	{
		size_ = size;
		// saint_venant_.set_size(size);
		neo_hookean_.set_size(size);
		linear_elasticity_.size() = size;
	}

	void MultiModel::init_multimaterial(const bool is_volume, const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus)
	{
		// saint_venant_.init_multimaterial(Es, nus);
		neo_hookean_.init_multimaterial(is_volume, Es, nus);
		linear_elasticity_.init_multimaterial(is_volume, Es, nus);
	}

	void MultiModel::set_parameters(const json &params)
	{
		set_size(params["size"]);

		// saint_venant_.set_parameters(params);
		neo_hookean_.set_parameters(params);
		linear_elasticity_.set_parameters(params);
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
	MultiModel::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		const int el_id = vals.element_id;
		const std::string model = multi_material_models_[el_id];

		// if (model == "SaintVenant")
		// 	return saint_venant_.assemble_grad(vals, displacement, da);
		// else
		if (model == "NeoHookean")
			return neo_hookean_.assemble_grad(vals, displacement, da);
		else if (model == "LinearElasticity")
			return linear_elasticity_.assemble_grad(vals, displacement, da);
		else
		{
			assert(false);
			return Eigen::VectorXd(0, 0);
		}
	}

	Eigen::MatrixXd
	MultiModel::assemble_hessian(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		const int el_id = vals.element_id;
		const std::string model = multi_material_models_[el_id];

		// if (model == "SaintVenant")
		// 	return saint_venant_.assemble_hessian(vals, displacement, da);
		// else
		if (model == "NeoHookean")
			return neo_hookean_.assemble_hessian(vals, displacement, da);
		else if (model == "LinearElasticity")
			return linear_elasticity_.assemble_hessian(vals, displacement, da);
		else
		{
			assert(false);
			return Eigen::MatrixXd(0, 0);
		}
	}

	double MultiModel::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		const int el_id = vals.element_id;
		const std::string model = multi_material_models_[el_id];

		// if (model == "SaintVenant")
		// 	return saint_venant_.compute_energy(vals, displacement, da);
		// else
		if (model == "NeoHookean")
			return neo_hookean_.compute_energy(vals, displacement, da);
		else if (model == "LinearElasticity")
			return linear_elasticity_.compute_energy(vals, displacement, da);
		else
		{
			assert(false);
			return 0;
		}
	}

	void MultiModel::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		const std::string model = multi_material_models_[el_id];

		// if (model == "SaintVenant")
		// 	saint_venant_.compute_stress_tensor(el_id, bs, gbs, local_pts, displacement, stresses);
		// else
		if (model == "NeoHookean")
			neo_hookean_.compute_stress_tensor(el_id, bs, gbs, local_pts, displacement, stresses);
		else if (model == "LinearElasticity")
			linear_elasticity_.compute_stress_tensor(el_id, bs, gbs, local_pts, displacement, stresses);
		else
		{
			assert(false);
			stresses = Eigen::MatrixXd(0, 0);
		}
	}

	void MultiModel::compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		const std::string model = multi_material_models_[el_id];

		// if (model == "SaintVenant")
		// 	saint_venant_.compute_von_mises_stresses(el_id, bs, gbs, local_pts, displacement, stresses);
		// else
		if (model == "NeoHookean")
			neo_hookean_.compute_von_mises_stresses(el_id, bs, gbs, local_pts, displacement, stresses);
		else if (model == "LinearElasticity")
			linear_elasticity_.compute_von_mises_stresses(el_id, bs, gbs, local_pts, displacement, stresses);
		else
		{
			assert(false);
			stresses = Eigen::MatrixXd(0, 0);
		}
	}
} // namespace polyfem
