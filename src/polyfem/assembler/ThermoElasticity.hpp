#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>

namespace polyfem::assembler
{
	class ThermoElasticity : public MixedNLAssembler
	{
	public:
		ThermoElasticity();

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
		template <typename T>
		T compute_energy_aux(const MixedNonLinearAssemblerData &data) const;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const;

		double alpha(const RowVectorNd &uv, const RowVectorNd &p, const double t, const int element_id) const;
		double T0(const RowVectorNd &uv, const RowVectorNd &p, const double t, const int element_id) const;

		LameParameters elastic_params_;
		GenericMatParam alpha_;
		GenericMatParam T0_;
	};
} // namespace polyfem::assembler
