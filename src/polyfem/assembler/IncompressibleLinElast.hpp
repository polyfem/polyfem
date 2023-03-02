#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

// local assembler for incompressible model, pressure is separate (see Stokes)
namespace polyfem::assembler
{
	// displacement assembler
	class IncompressibleLinearElasticityDispacement : public TensorLinearAssembler
	{
	public:
		using TensorLinearAssembler::assemble;

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void add_multimaterial(const int index, const json &params);
		void set_params(const LameParameters &params) { params_ = params; }

		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const ElasticityTensorType &type, Eigen::MatrixXd &tensor) const;

	private:
		LameParameters params_;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};

	// mixed, displacement and pressure
	class IncompressibleLinearElasticityMixed : public TensorMixedAssembler
	{
	public:
		using TensorMixedAssembler::assemble;

		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const override;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		inline int rows() const override { return size_; }
		inline int cols() const override { return 1; }

		void add_multimaterial(const int index, const json &params) {}
		void set_params(const LameParameters &params) {}
	};

	// pressure only part
	class IncompressibleLinearElasticityPressure : public ScalarLinearAssembler
	{
	public:
		using ScalarLinearAssembler::assemble;

		// res is R^{1}
		Eigen::Matrix<double, 1, 1>
		assemble(const LinearAssemblerData &data) const override;

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		void add_multimaterial(const int index, const json &params);
		void set_params(const LameParameters &params) { params_ = params; }

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler
