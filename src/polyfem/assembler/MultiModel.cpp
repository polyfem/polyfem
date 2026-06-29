#include "MultiModel.hpp"

#include <cstddef>
#include <set>

// #include <polyfem/basis/Basis.hpp>
// #include <polyfem/autogen/auto_elasticity_rhs.hpp>

// #include <igl/Timer.h>

namespace polyfem::assembler
{
	void MultiModel::set_size(const int size)
	{
		Assembler::set_size(size);

		all_elastic_materials_.set_size(size);
	}

	void MultiModel::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		all_elastic_materials_.add_multimaterial(index, params, units, root_path);
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
		const auto assembler = all_elastic_materials_.get_assembler(model);
		return assembler->assemble_gradient(data);
	}

	Eigen::MatrixXd
	MultiModel::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const int el_id = data.vals.element_id;
		const std::string model = multi_material_models_[el_id];
		const auto assembler = all_elastic_materials_.get_assembler(model);
		return assembler->assemble_hessian(data);
	}

	double MultiModel::compute_energy(const NonLinearAssemblerData &data) const
	{
		const int el_id = data.vals.element_id;
		const std::string model = multi_material_models_[el_id];
		const auto assembler = all_elastic_materials_.get_assembler(model);
		return assembler->compute_energy(data);
	}

	void MultiModel::assign_stress_tensor(
		const OutputData &data,
		const int all_size,
		const ElasticityTensorType &type,
		Eigen::MatrixXd &all,
		const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		const std::string model = multi_material_models_[data.el_id];
		const auto assembler = all_elastic_materials_.get_assembler(model);
		auto elasticity_assembler = dynamic_cast<ElasticityNLAssembler *>(assembler.get());
		assert(elasticity_assembler && "Failed to cast assembler to ElasticityNLAssembler.");
		elasticity_assembler->assign_stress_tensor(data, all_size, type, all, fun);
	}

	std::map<std::string, Assembler::ParamFunc> MultiModel::parameters() const
	{
		std::map<std::string, Assembler::ParamFunc> params;
		const std::set<std::string> active_models(multi_material_models_.begin(), multi_material_models_.end());

		for (const std::string &model : active_models)
		{
			const auto assembler = all_elastic_materials_.get_assembler(model);
			for (const auto &entry : assembler->parameters())
			{
				const std::string name = entry.first;
				const ParamFunc func = entry.second;
				params[model + "/" + name] = [this, model, func](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
					if (e < 0 || std::size_t(e) >= multi_material_models_.size() || multi_material_models_[e] != model)
						return 0.0;
					return func(uv, p, t, e);
				};
			}
		}

		return params;
	}
} // namespace polyfem::assembler
