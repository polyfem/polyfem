#include "SumModel.hpp"

#include <jse/jse.h>

// #include <polyfem/basis/Basis.hpp>
// #include <polyfem/autogen/auto_elasticity_rhs.hpp>

// #include <igl/Timer.h>

namespace polyfem::assembler
{
	void SumModel::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);
		if (params.count("models") == 0)
			return;

		auto models = params["models"];

		json rules;
		jse::JSE jse;
		jse.strict = true;
		const std::string mat_spec = POLYFEM_MATERIAL_INPUT_SPEC;
		std::ifstream file(mat_spec);

		if (file.is_open())
			file >> rules;
		else
			log_and_throw_error(fmt::format("unable to open {} rules", mat_spec));

		rules = jse.inject_include(rules);

		for (const auto model : models)
		{
			const bool valid_input = jse.verify_json(model, rules);
			if (!valid_input)
				log_and_throw_error(fmt::format("invalid material json:\n{}", jse.log2str()));
			jse.inject_defaults(model, rules);

			const std::string model_name = model["type"];

			const auto assembler = AssemblerUtils::make_assembler(model_name);
			// cast assembler to elasticity assembler
			assemblers_.emplace_back(std::dynamic_pointer_cast<NLAssembler>(assembler));
			assert(assemblers_.back() != nullptr);
			assemblers_.back()->set_size(size());
			assemblers_.back()->add_multimaterial(index, model, units);
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

	Eigen::MatrixXd
	SumModel::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		Eigen::MatrixXd hessian = assemblers_.front()->assemble_hessian(data);
		for (size_t i = 1; i < assemblers_.size(); ++i)
		{
			const auto assembler = assemblers_[i];
			hessian += assembler->assemble_hessian(data);
		}
		return hessian;
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
		for (const auto &assembler : assemblers_)
		{
			std::dynamic_pointer_cast<assembler::ElasticityAssembler>(assembler)->assign_stress_tensor(data, all_size, type, all, fun);
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
