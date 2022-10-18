#pragma once

#include "AssemblerData.hpp"
#include "MatParams.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <functional>

// local assembler for incompressible model, pressure is separate (see Stokes)
namespace polyfem::assembler
{
	// displacement assembler
	class IncompressibleLinearElasticityDispacement
	{
	public:
		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);
		inline int size() const { return size_; }

		void add_multimaterial(const int index, const json &params);
		void set_params(const LameParameters &params) { params_ = params; }

		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

	private:
		int size_ = -1;

		LameParameters params_;

		void assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};

	// mixed, displacement and pressure
	class IncompressibleLinearElasticityMixed
	{
	public:
		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);

		inline int rows() const { return size_; }
		inline int cols() const { return 1; }

		void add_multimaterial(const int index, const json &params) {}
		void set_params(const LameParameters &params) {}

	private:
		int size_ = -1;
	};

	// pressure only part
	class IncompressibleLinearElasticityPressure
	{
	public:
		// res is R^{1}
		Eigen::Matrix<double, 1, 1>
		assemble(const LinearAssemblerData &data) const;

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		inline int size() const { return 1; }
		void set_size(const int size) { size_ = size; }

		void add_multimaterial(const int index, const json &params);
		void set_params(const LameParameters &params) { params_ = params; }

	private:
		int size_ = -1;
		LameParameters params_;
	};
} // namespace polyfem::assembler
