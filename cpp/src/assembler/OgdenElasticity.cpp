// #define EIGEN_STACK_ALLOCATION_LIMIT 0

#include "OgdenElasticity.hpp"
#include "Basis.hpp"
#include "MatrixUtils.hpp"

#include "auto_eigs.hpp"
#include "auto_rhs.hpp"

#include <igl/Timer.h>


namespace poly_fem
{
	namespace
	{
		void fill_mat_from_vect(const std::vector<double> vals, Eigen::VectorXd &vec)
		{
			vec.resize(vals.size());

			for(size_t i = 0; i < vals.size(); ++i)
				vec(i) = vals[i];
		}
	}

	OgdenElasticity::OgdenElasticity()
	{
		set_size(size_);
		alphas_.resize(1);
		mus_.resize(1);
		Ds_.resize(1);

		alphas_.setOnes();
		mus_.setOnes();
		Ds_.setOnes();
	}

	void OgdenElasticity::set_parameters(const json &params)
	{
		set_size(params["size"]);

		if(params.count("alphas"))
		{
			const std::vector<double> tmp = params["alphas"];
			fill_mat_from_vect(tmp, alphas_);
		}
		if(params.count("mus"))
		{
			const std::vector<double> tmp = params["mus"];
			fill_mat_from_vect(tmp, mus_);
		}
		if(params.count("Ds"))
		{
			const std::vector<double> tmp = params["Ds"];
			fill_mat_from_vect(tmp, Ds_);
		}

		assert(alphas_.size() == mus_.size());
		assert(Ds_.size() == mus_.size());
	}

	void OgdenElasticity::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	OgdenElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;

		// if(size() == 2)
		// 	autogen::saint_venant_2d_function(pt, elasticity_tensor_, res);
		// else if(size() == 3)
		// 	autogen::saint_venant_3d_function(pt, elasticity_tensor_, res);
		// else
		// 	assert(false);

		return res;
	}

	Eigen::VectorXd
	OgdenElasticity::assemble(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		// igl::Timer time; time.start();

		const int n_bases = vals.basis_values.size();

		return poly_fem::gradient_from_energy(size(), n_bases, vals, displacement, da,
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 60, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 90, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 1000, 1>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(vals, displacement, da); }
		);
	}

	Eigen::MatrixXd
	OgdenElasticity::assemble_grad(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		// igl::Timer time; time.start();

		const int n_bases = vals.basis_values.size();
		return poly_fem::hessian_from_energy(size(), n_bases, vals, displacement, da,
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 90, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 90, 90>>>(vals, displacement, da); },
			[&](const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) { return compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(vals, displacement, da); }
		);
	}

	void OgdenElasticity::compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		// Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);
		ElementAssemblyValues vals;
		vals.compute(-1, size() == 3, local_pts, bs, bs);


		stresses.resize(local_pts.rows(), 1);


		Eigen::MatrixXd loc_displacement(1, size());
		for(long p = 0; p < local_pts.rows(); ++p)
		{
			loc_displacement.setZero();

			for(std::size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				const auto &loc_val = vals.basis_values[j];

				assert(bs.bases.size() == vals.basis_values.size());
				assert(loc_val.val.size() == local_pts.rows());

				for(int d = 0; d < size(); ++d)
				{
					for(std::size_t ii = 0; ii < b.global().size(); ++ii)
					{
						loc_displacement(d) += b.global()[ii].val * loc_val.val(p) * displacement(b.global()[ii].index*size() + d);
					}
				}
			}

			stresses(p) = loc_displacement.norm();
		}


		// for(long p = 0; p < local_pts.rows(); ++p)
		// {
		// 	displacement_grad.setZero();

		// 	for(std::size_t j = 0; j < bs.bases.size(); ++j)
		// 	{
		// 		const Basis &b = bs.bases[j];
		// 		const auto &loc_val = vals.basis_values[j];

		// 		assert(bs.bases.size() == vals.basis_values.size());
		// 		assert(loc_val.grad.rows() == local_pts.rows());
		// 		assert(loc_val.grad.cols() == size());

		// 		for(int d = 0; d < size(); ++d)
		// 		{
		// 			for(std::size_t ii = 0; ii < b.global().size(); ++ii)
		// 			{
		// 				displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index*size() + d);
		// 			}
		// 		}
		// 	}

		// 	displacement_grad = displacement_grad * vals.jac_it[p];

		// 	Eigen::MatrixXd strain = strain_from_disp_grad(displacement_grad);
		// 	Eigen::MatrixXd stress_tensor(size(), size());

		// 	if(size() == 2)
		// 	{
		// 		std::array<double, 3> eps;
		// 		eps[0] = strain(0,0);
		// 		eps[1] = strain(1,1);
		// 		eps[2] = 2*strain(0,1);


		// 		stress_tensor <<
		// 		stress(eps, 0), stress(eps, 2),
		// 		stress(eps, 2), stress(eps, 1);
		// 	}
		// 	else
		// 	{
		// 		std::array<double, 6> eps;
		// 		eps[0] = strain(0,0);
		// 		eps[1] = strain(1,1);
		// 		eps[2] = strain(2,2);
		// 		eps[3] = 2*strain(1,2);
		// 		eps[4] = 2*strain(0,2);
		// 		eps[5] = 2*strain(0,1);

		// 		stress_tensor <<
		// 		stress(eps, 0), stress(eps, 5), stress(eps, 4),
		// 		stress(eps, 5), stress(eps, 1), stress(eps, 3),
		// 		stress(eps, 4), stress(eps, 3), stress(eps, 2);
		// 	}

		// 	stress_tensor = (Eigen::MatrixXd::Identity(size(), size()) + displacement_grad) * stress_tensor;

		// 	stresses(p) = von_mises_stress_for_stress_tensor(stress_tensor);
		// }
	}

	double OgdenElasticity::compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		return compute_energy_aux<double>(vals, displacement, da);
	}

	//http://abaqus.software.polimi.it/v6.14/books/stm/default.htm?startat=ch04s06ath123.html Ogden form
	template<typename T>
	T OgdenElasticity::compute_energy_aux(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> 							AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> 		AutoDiffGradMat;

		assert(displacement.cols() == 1);

		const int n_pts = da.size();

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(vals.basis_values.size() * size(), 1);
		local_dispv.setZero();
		for(size_t i = 0; i < vals.basis_values.size(); ++i){
			const auto &bs = vals.basis_values[i];
			for(size_t ii = 0; ii < bs.global.size(); ++ii){
				for(int d = 0; d < size(); ++d){
					local_dispv(i*size() + d) += bs.global[ii].val * displacement(bs.global[ii].index*size() + d);
				}
			}
		}

		DiffScalarBase::setVariableCount(local_dispv.rows());
		AutoDiffVect local_disp(local_dispv.rows(), 1);
		T energy = T(0.0);

		const AutoDiffAllocator<T> allocate_auto_diff_scalar;

		for(long i = 0; i < local_dispv.rows(); ++i){
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv(i));
		}

		AutoDiffGradMat def_grad(size(), size());

		for(long p = 0; p < n_pts; ++p)
		{
			for(long k = 0; k < def_grad.size(); ++k)
				def_grad(k) = T(0);

			for(size_t i = 0; i < vals.basis_values.size(); ++i)
			{
				const auto &bs = vals.basis_values[i];
				const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
				assert(grad.size() == size());

				for(int d = 0; d < size(); ++d)
				{
					for(int c = 0; c < size(); ++c)
					{
						def_grad(d, c) += grad(c) * local_disp(i*size() + d);
					}
				}
			}

			AutoDiffGradMat jac_it(size(), size());
			for(long k = 0; k < jac_it.size(); ++k)
				jac_it(k) = T(vals.jac_it[p](k));

			def_grad = def_grad * jac_it;

			//Id + grad d
			for(int d = 0; d < size(); ++d)
				def_grad(d,d) += T(1);

			Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> eigs;

			if(size() == 2)
				autogen::eigs_2d<T>(def_grad, eigs);
			else if(size() == 3)
				autogen::eigs_3d<T>(def_grad, eigs);
			else
				assert(false);

			const T J = poly_fem::determinant(def_grad);
			const T Jdenom = pow(J, -1./size());

			for(long i = 0; i < eigs.size(); ++i)
				eigs(i) = eigs(i) * Jdenom;

			auto val = T(0);
			for(long N = 0; N < alphas_.size(); ++N)
			{
				auto tmp = T(-size());
				const double alpha = alphas_(N);
				const double mu = mus_(N);

				for(long i = 0; i < eigs.size(); ++i)
					tmp += pow(eigs(i), alpha);

				val += 2*mu/(alpha * alpha) * tmp;
			}

			// std::cout<<val<<std::endl;

			for(long N = 0; N < Ds_.size(); ++N)
			{
				const double D = Ds_(N);

				val += 1./D * pow(J - T(1), 2*(N+1));
			}

			energy += val * da(p);
		}
		// std::cout<<"\n\n------------\n\n\n"<<std::endl;
		return energy;
	}
}
