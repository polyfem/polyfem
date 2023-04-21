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
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		void add_multimaterial(const int index, const json &params, const Units &units) override;
		void set_params(const LameParameters &params) { params_ = params; }

		std::string name() const override { return "IncompressibleLinearElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

	protected:
		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

	private:
		LameParameters params_;
	};

	// mixed, displacement and pressure
	class IncompressibleLinearElasticityMixed : public MixedAssembler
	{
	public:
		std::string name() const override { return "IncompressibleLinearElasticityMixed"; }

		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const override;

		inline int rows() const override { return size_; }
		inline int cols() const override { return 1; }
	};

	// pressure only part
	class IncompressibleLinearElasticityPressure : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		// res is R^{1}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		void add_multimaterial(const int index, const json &params, const Units &units) override;
		void set_params(const LameParameters &params) { params_ = params; }

		std::string name() const override { return "IncompressibleLinearElasticityPressure"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		void set_size(const int) override { size_ = 1; }

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler
