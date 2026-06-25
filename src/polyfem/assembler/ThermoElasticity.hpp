#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

namespace polyfem::assembler
{
	class ThermoElasticity : public MixedNLAssembler
	{
	public:
		std::string name() const override { return "ThermoElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

	protected:
		int rows() const override { return size(); }
		int cols() const override { return 1; }

		double compute_energy(const MixedNonLinearAssemblerData &data) const override;
		Eigen::VectorXd compute_gradient(const MixedNonLinearAssemblerData &data) const override;
		Eigen::MatrixXd compute_hessian(const MixedNonLinearAssemblerData &data) const override;

	private:
		template <typename T>
		T compute_energy_aux(const MixedNonLinearAssemblerData &data) const;

		template <typename T>
		T elastic_energy(
			const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &Fe,
			const double lambda,
			const double mu) const;

		double alpha(const RowVectorNd &uv, const RowVectorNd &p, const double t, const int element_id) const;
		double T0(const RowVectorNd &uv, const RowVectorNd &p, const double t, const int element_id) const;

		double lambda() const;
		double mu() const;

		static double eval_param(
			const std::vector<utils::ExpressionValue> &params,
			const double default_value,
			const RowVectorNd &p,
			const double t,
			const int element_id);

		std::vector<utils::ExpressionValue> alpha_;
		std::vector<utils::ExpressionValue> T0_;

		// Temporary constants while the coupled model is being brought up.
		double young_ = 20000.0;
		double nu_ = 0.3;
	};
} // namespace polyfem::assembler
