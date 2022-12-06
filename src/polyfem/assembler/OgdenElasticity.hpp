#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/autogen/auto_eigs.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <array>

// attempt at Ogden, incomplete, not used, and not working
namespace polyfem::assembler
{
	class OgdenElasticity
	{
	public:
		OgdenElasticity();

		// sets material params
		void add_multimaterial(const int index, const json &params, const int size);

		// http://abaqus.software.polimi.it/v6.14/books/stm/default.htm?startat=ch04s06ath123.html Ogden form
		template <typename T>
		T elastic_energy(const int size,
						 const RowVectorNd &p,
						 const int el_id,
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &disp_grad) const
		{
			// Id + grad d
			auto def_grad = disp_grad;
			for (int d = 0; d < size; ++d)
				def_grad(d, d) += T(1);

			Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> eigs;

			if (size == 2)
				autogen::eigs_2d<T>(def_grad, eigs);
			else if (size == 3)
				autogen::eigs_3d<T>(def_grad, eigs);
			else
				assert(false);

			const T J = utils::determinant(def_grad);
			const T Jdenom = pow(J, -1. / size);

			for (long i = 0; i < eigs.size(); ++i)
				eigs(i) = eigs(i) * Jdenom;

			auto val = T(0);
			for (long N = 0; N < alphas_.size(); ++N)
			{
				auto tmp = T(-size);
				const double alpha = alphas_(N);
				const double mu = mus_(N);

				for (long i = 0; i < eigs.size(); ++i)
					tmp += pow(eigs(i), alpha);

				val += 2 * mu / (alpha * alpha) * tmp;
			}

			// std::cout<<val<<std::endl;

			for (long N = 0; N < Ds_.size(); ++N)
			{
				const double D = Ds_(N);

				val += 1. / D * pow(J - T(1), 2 * (N + 1));
			}

			return val;
		}

		const Eigen::VectorXd &alphas() const { return alphas_; }
		const Eigen::VectorXd &mus() const { return mus_; }
		const Eigen::VectorXd &Ds() const { return Ds_; }

	private:
		Eigen::VectorXd alphas_;
		Eigen::VectorXd mus_;
		Eigen::VectorXd Ds_;
	};
} // namespace polyfem::assembler
