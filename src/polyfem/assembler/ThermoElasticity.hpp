#pragma once

#include <polyfem/assembler/Assembler.hpp>

#include <memory>

namespace polyfem::assembler
{
	namespace detail
	{
		class ThermoElasticityModel;
	}

	class ThermoElasticity : public MixedNLAssembler
	{
	public:
		ThermoElasticity();
		~ThermoElasticity() override;

		std::string name() const override { return "ThermoElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void set_size(const int size) override;
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

	protected:
		int rows() const override { return size(); }
		int cols() const override { return 1; }

		double compute_energy(const MixedNonLinearAssemblerData &data) const override;
		Eigen::VectorXd compute_gradient(const MixedNonLinearAssemblerData &data) const override;
		Eigen::MatrixXd compute_hessian(const MixedNonLinearAssemblerData &data) const override;

	private:
		detail::ThermoElasticityModel &model();
		const detail::ThermoElasticityModel &model() const;

		std::unique_ptr<detail::ThermoElasticityModel> model_;
		std::string elastic_formulation_;
	};
} // namespace polyfem::assembler
