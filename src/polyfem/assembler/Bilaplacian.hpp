#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <functional>

namespace polyfem
{
	namespace assembler
	{
		//local assembler for bilaplacian, see laplacian
		// 0 L
		// L M
		class BilaplacianMain
		{
		public:
			Eigen::Matrix<double, 1, 1>
			assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
			{
				return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
			}

			Eigen::Matrix<double, 1, 1>
			compute_rhs(const AutodiffHessianPt &pt) const;

			inline int size() const { return 1; }

			void add_multimaterial(const int index, const json &params) {}
		};

		class BilaplacianMixed
		{
		public:
			// res is R
			Eigen::Matrix<double, 1, 1>
			assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const;

			Eigen::Matrix<double, 1, 1>
			compute_rhs(const AutodiffHessianPt &pt) const
			{
				assert(false);
				return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
			}

			inline int rows() const { return 1; }
			inline int cols() const { return 1; }

			void add_multimaterial(const int index, const json &params) {}
		};

		class BilaplacianAux
		{
		public:
			// res is R
			Eigen::Matrix<double, 1, 1>
			assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;

			Eigen::Matrix<double, 1, 1>
			compute_rhs(const AutodiffHessianPt &pt) const
			{
				assert(false);
				return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
			}

			inline int size() const { return 1; }

			void add_multimaterial(const int index, const json &params) {}
		};
	} // namespace assembler
} // namespace polyfem
