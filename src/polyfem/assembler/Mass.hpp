#pragma once

#include "AssemblerData.hpp"
#include "MatParams.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <Eigen/Dense>

namespace polyfem::assembler
{
	class Mass
	{
	public:
		// computes local stiffness matrix (1x1) for bases i,j
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> assemble(const LinearAssemblerData &data) const;

		// uses autodiff to compute the rhs for a fabricated solution
		// in this case it just return pt.getHessian().trace()
		// pt is the evaluation of the solution at a point
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> compute_rhs(const AutodiffHessianPt &pt) const;

		// size of the problem, this is a tensor problem so the size is the size of the mesh
		void set_size(const int size) { size_ = size; }
		inline int size() const { return size_; }

		// inialize material parameter
		void add_multimaterial(const int index, const json &params);

		// class that stores and compute density per point
		const Density &density() const { return density_; }

	private:
		int size_ = -1;
		// class that stores and compute density per point
		Density density_;
	};
} // namespace polyfem::assembler
