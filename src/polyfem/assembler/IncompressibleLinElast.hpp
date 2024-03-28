#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// local assembler for incompressible model, pressure is separate (see Stokes)
namespace polyfem::assembler
{
	// displacement assembler
	class IncompressibleLinearElasticityDispacement : public LinearAssembler, ElasticityAssembler
	{
	public:
		using LinearAssembler::assemble;
		// res is R^{dim²}
		FlatMatrixNd
		assemble(const LinearAssemblerData &data) const override;

		void add_multimaterial(const int index, const json &params, const Units &units) override;
		void set_params(const LameParameters &params) { params_ = params; }

		std::string name() const override { return "IncompressibleLinearElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

	protected:
		void assign_stress_tensor(const OutputData &data,
								  const int all_size,
								  const ElasticityTensorType &type,
								  Eigen::MatrixXd &all,
								  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		LameParameters params_;
	};

	// mixed, displacement and pressure
	class IncompressibleLinearElasticityMixed : public MixedAssembler
	{
	public:
		std::string name() const override { return "IncompressibleLinearElasticityMixed"; }

		// res is R^{dim}
		VectorNd
		assemble(const MixedAssemblerData &data) const override;

		inline int rows() const override { return codomain_size(); }
		inline int cols() const override { return 1; }
	};

	// pressure only part
	class IncompressibleLinearElasticityPressure : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		// res is R^{1}
		FlatMatrixNd
		assemble(const LinearAssemblerData &data) const override;

		void add_multimaterial(const int index, const json &params, const Units &units) override;
		void set_params(const LameParameters &params) { params_ = params; }

		std::string name() const override { return "IncompressibleLinearElasticityPressure"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		void set_sizes(const unsigned domain_size, const unsigned codomain_size) override
		{
			domain_size_ = domain_size;
			disp_size_ = codomain_size; // displacement
			codomain_size_ = 1;         // pressure
		}

	private:
		LameParameters params_;
		unsigned disp_size_ = 0;
	};
} // namespace polyfem::assembler
