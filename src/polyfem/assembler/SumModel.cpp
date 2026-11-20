#include "SumModel.hpp"
#include "MatParams.hpp"

#include <jse/jse.h>

// #include <polyfem/basis/Basis.hpp>
// #include <polyfem/autogen/auto_elasticity_rhs.hpp>

// #include <igl/Timer.h>

namespace polyfem::assembler
{
	void SumModel::add_multimaterial(const int index, const json &params, const Units &units, const io::ResourceIO &resources)
	{
		assert(size() == 2 || size() == 3);
		if (params.count("models") == 0)
			return;

		auto models = params["models"];

		// add_multimaterial may be invoked once per mesh element (see Assembler::set_materials),
		// so the child assemblers must be created once and then reused across calls, not appended every time.
		if (assemblers_.empty())
		{
			for (const auto &model : models)
			{
				const std::string model_name = model["type"];

				const auto assembler = AssemblerUtils::make_assembler(model_name);
				// cast assembler to elasticity assembler
				assemblers_.emplace_back(std::dynamic_pointer_cast<NLAssembler>(assembler));
				assert(assemblers_.back() != nullptr);
				assemblers_.back()->set_size(size());
			}
		}

		assert(assemblers_.size() == models.size());
		for (size_t i = 0; i < assemblers_.size(); ++i)
		{
			json model = models[i];
			if (params.contains(MATERIAL_ELEMENT_INDEX))
				model[MATERIAL_ELEMENT_INDEX] = params[MATERIAL_ELEMENT_INDEX];
			assemblers_[i]->add_multimaterial(index, model, units, resources);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	SumModel::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;
		assert(false);

		return res;
	}

	Eigen::VectorXd
	SumModel::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		Eigen::VectorXd gradient = assemblers_.front()->assemble_gradient(data);
		for (size_t i = 1; i < assemblers_.size(); ++i)
		{
			const auto assembler = assemblers_[i];
			gradient += assembler->assemble_gradient(data);
		}
		return gradient;
	}

	void SumModel::assemble_hessian(const NonLinearAssemblerData &data, LocalHessian &hessian) const
	{
		assemblers_.front()->assemble_hessian(data, hessian);
		LocalHessian contribution;
		for (size_t i = 1; i < assemblers_.size(); ++i)
		{
			assemblers_[i]->assemble_hessian(data, contribution);
			hessian.visit([&](auto &H) {
				contribution.visit([&](const auto &term) { H += term; });
			});
		}
	}

	double SumModel::compute_energy(const NonLinearAssemblerData &data) const
	{
		double energy = assemblers_.front()->compute_energy(data);
		for (size_t i = 1; i < assemblers_.size(); ++i)
		{
			const auto assembler = assemblers_[i];
			energy += assembler->compute_energy(data);
		}
		return energy;
	}

	void SumModel::assign_stress_tensor(
		const OutputData &data,
		const int all_size,
		const ElasticityTensorType &type,
		Eigen::MatrixXd &all,
		const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		all.resize(data.local_pts.rows(), all_size);
		all.setZero();

		Eigen::MatrixXd tmp;

		if (type == ElasticityTensorType::F)
		{
			std::dynamic_pointer_cast<assembler::ElasticityAssembler>(assemblers_.front())
				->assign_stress_tensor(data, all_size, type, all, fun);
			return;
		}

		for (const auto &assembler : assemblers_)
		{
			std::dynamic_pointer_cast<assembler::ElasticityNLAssembler>(assembler)->assign_stress_tensor(data, all_size, type, tmp, fun);
			all += tmp;
		}
	}

	void SumModel::compute_output_values(const OutputData &data,
										 std::vector<NamedMatrix> *scalar, std::vector<NamedMatrix> *tensor) const
	{
		if (!scalar && !tensor)
			return;
		assert(!assemblers_.empty());
		std::vector<NamedMatrix> scalar_part, tensor_part;
		for (size_t i = 0; i < assemblers_.size(); ++i)
		{
			const auto material = std::dynamic_pointer_cast<ElasticityAssembler>(assemblers_[i]);
			material->compute_output_values(data, scalar ? &scalar_part : nullptr, tensor ? &tensor_part : nullptr);
			if (scalar)
			{
				if (i == 0)
					*scalar = std::move(scalar_part);
				else
					for (size_t k = 0; k < scalar_part.size(); ++k)
						(*scalar)[k].second += scalar_part[k].second;
			}
			if (tensor)
			{
				if (i == 0)
					*tensor = std::move(tensor_part);
				else
					for (size_t k = 0; k < tensor_part.size(); ++k)
						if (tensor_part[k].first != "F")
							(*tensor)[k].second += tensor_part[k].second;
			}
		}
	}

	std::map<std::string, Assembler::ParamFunc> SumModel::parameters() const
	{
		std::map<std::string, Assembler::ParamFunc> params;
		for (const auto &a : assemblers_)
		{
			auto p = a->parameters();
			for (auto &it : p)
			{
				params[a->name() + "/" + it.first] = it.second;
			}
		}
		return params;
	}
} // namespace polyfem::assembler
