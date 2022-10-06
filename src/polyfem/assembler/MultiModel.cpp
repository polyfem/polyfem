// #define EIGEN_STACK_ALLOCATION_LIMIT 0

#include "MultiModel.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <igl/Timer.h>

namespace polyfem::assembler
{
	using namespace basis;
	void MultiModel::set_size(const int size)
	{
		size_ = size;
		// saint_venant_.set_size(size);
		neo_hookean_.set_size(size);
		linear_elasticity_.set_size(size);
	}

	void MultiModel::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		// saint_venant_.add_multimaterial(index, params);
		neo_hookean_.add_multimaterial(index, params);
		linear_elasticity_.add_multimaterial(index, params);
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
	MultiModel::assemble_grad(const NonLinearAssemblerData &data) const
	{
		const int el_id = data.vals.element_id;
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			return saint_venant_.assemble_grad(data);
		else if (model == "NeoHookean")
			return neo_hookean_.assemble_grad(data);
		else if (model == "LinearElasticity")
			return linear_elasticity_.assemble_grad(data);
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
		else
		{
			assert(false);
			return 0;
		}
	}

	void MultiModel::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			saint_venant_.compute_stress_tensor(el_id, bs, gbs, local_pts, displacement, stresses);
		else if (model == "NeoHookean")
			neo_hookean_.compute_stress_tensor(el_id, bs, gbs, local_pts, displacement, stresses);
		else if (model == "LinearElasticity")
			linear_elasticity_.compute_stress_tensor(el_id, bs, gbs, local_pts, displacement, stresses);
		else
		{
			assert(false);
			stresses = Eigen::MatrixXd(0, 0);
		}
	}

	void MultiModel::compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		const std::string model = multi_material_models_[el_id];

		if (model == "SaintVenant")
			saint_venant_.compute_von_mises_stresses(el_id, bs, gbs, local_pts, displacement, stresses);
		else if (model == "NeoHookean")
			neo_hookean_.compute_von_mises_stresses(el_id, bs, gbs, local_pts, displacement, stresses);
		else if (model == "LinearElasticity")
			linear_elasticity_.compute_von_mises_stresses(el_id, bs, gbs, local_pts, displacement, stresses);
		else
		{
			assert(false);
			stresses = Eigen::MatrixXd(0, 0);
		}
	}
} // namespace polyfem::assembler
