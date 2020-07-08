#include <polyfem/HookeLinearElasticity.hpp>

#include <polyfem/Basis.hpp>
#include <polyfem/ElementAssemblyValues.hpp>

#include <polyfem/auto_elasticity_rhs.hpp>

namespace polyfem
{

	namespace
	{
		template<class Matrix>
		Matrix strain_from_disp_grad(const Matrix &disp_grad)
		{
			Matrix mat =  (disp_grad + disp_grad.transpose());

			for(int i = 0; i < mat.size(); ++i)
				mat(i) *= 0.5;

			return mat;
		}

		template<int dim>
		Eigen::Matrix<double, dim, dim> strain(const Eigen::MatrixXd &grad, const Eigen::MatrixXd &jac_it, int k, int coo)
		{
			Eigen::Matrix<double, dim, dim> jac;
			jac.setZero();
			jac.row(coo) = grad.row(k);
			jac = jac*jac_it;

			return strain_from_disp_grad(jac);
		}
	}



	HookeLinearElasticity::HookeLinearElasticity()
	{
		set_size(size_);
	}

	void HookeLinearElasticity::set_parameters(const json &params)
	{
		set_size(params["size"]);

		if(params["elasticity_tensor"].empty())
		{
			if (params.count("young")) {
				if (params["young"].is_number() && params["nu"].is_number())
					elasticity_tensor_.set_from_young_poisson(params["young"], params["nu"]);
			} else if (params.count("E")) {
				if (params["E"].is_number() && params["nu"].is_number())
					elasticity_tensor_.set_from_young_poisson(params["E"], params["nu"]);
			} else {
				if (params["lambda"].is_number() && params["mu"].is_number())
					elasticity_tensor_.set_from_lambda_mu(params["lambda"], params["mu"]);
			}
		}
		else
		{
			std::vector<double> entries = params["elasticity_tensor"];
			elasticity_tensor_.set_from_entries(entries);
		}
	}

	void HookeLinearElasticity::set_size(const int size)
	{
		elasticity_tensor_.resize(size);
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
	HookeLinearElasticity::assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
	{
		const Eigen::MatrixXd &gradi = vals.basis_values[i].grad;
		const Eigen::MatrixXd &gradj = vals.basis_values[j].grad;


		// (C : gradi) : gradj
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size()*size());
		res.setZero();
		assert(gradi.cols() == size());
		assert(gradj.cols() == size());
		assert(size_t(gradi.rows()) ==  vals.jac_it.size());

		for(long k = 0; k < gradi.rows(); ++k)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res_k(size()*size());

			if(size_ == 2)
			{
				const Eigen::Matrix2d eps_x_i = strain<2>(gradi, vals.jac_it[k], k, 0);
				const Eigen::Matrix2d eps_y_i = strain<2>(gradi, vals.jac_it[k], k, 1);

				const Eigen::Matrix2d eps_x_j = strain<2>(gradj, vals.jac_it[k], k, 0);
				const Eigen::Matrix2d eps_y_j = strain<2>(gradj, vals.jac_it[k], k, 1);

				std::array<double, 3> e_x, e_y;
				e_x[0] = eps_x_i(0,0);
				e_x[1] = eps_x_i(1,1);
				e_x[2] = 2*eps_x_i(0,1);

				e_y[0] = eps_y_i(0,0);
				e_y[1] = eps_y_i(1,1);
				e_y[2] = 2*eps_y_i(0,1);

				Eigen::Matrix2d sigma_x; sigma_x <<
				elasticity_tensor_.compute_stress<3>(e_x, 0), elasticity_tensor_.compute_stress<3>(e_x, 2),
				elasticity_tensor_.compute_stress<3>(e_x, 2), elasticity_tensor_.compute_stress<3>(e_x, 1);

				Eigen::Matrix2d sigma_y; sigma_y <<
				elasticity_tensor_.compute_stress<3>(e_y, 0), elasticity_tensor_.compute_stress<3>(e_y, 2),
				elasticity_tensor_.compute_stress<3>(e_y, 2), elasticity_tensor_.compute_stress<3>(e_y, 1);


				res_k(0) = (sigma_x*eps_x_j).trace();
				res_k(2) = (sigma_x*eps_y_j).trace();

				res_k(1) = (sigma_y*eps_x_j).trace();
				res_k(3) = (sigma_y*eps_y_j).trace();
			}
			else
			{
				const Eigen::Matrix3d eps_x_i = strain<3>(gradi, vals.jac_it[k], k, 0);
				const Eigen::Matrix3d eps_y_i = strain<3>(gradi, vals.jac_it[k], k, 1);
				const Eigen::Matrix3d eps_z_i = strain<3>(gradi, vals.jac_it[k], k, 2);

				const Eigen::Matrix3d eps_x_j = strain<3>(gradj, vals.jac_it[k], k, 0);
				const Eigen::Matrix3d eps_y_j = strain<3>(gradj, vals.jac_it[k], k, 1);
				const Eigen::Matrix3d eps_z_j = strain<3>(gradj, vals.jac_it[k], k, 2);


				std::array<double, 6> e_x, e_y, e_z;
				e_x[0] = eps_x_i(0,0);
				e_x[1] = eps_x_i(1,1);
				e_x[2] = eps_x_i(2,2);
				e_x[3] = 2*eps_x_i(1,2);
				e_x[4] = 2*eps_x_i(0,2);
				e_x[5] = 2*eps_x_i(0,1);

				e_y[0] = eps_y_i(0,0);
				e_y[1] = eps_y_i(1,1);
				e_y[2] = eps_y_i(2,2);
				e_y[3] = 2*eps_y_i(1,2);
				e_y[4] = 2*eps_y_i(0,2);
				e_y[5] = 2*eps_y_i(0,1);

				e_z[0] = eps_z_i(0,0);
				e_z[1] = eps_z_i(1,1);
				e_z[2] = eps_z_i(2,2);
				e_z[3] = 2*eps_z_i(1,2);
				e_z[4] = 2*eps_z_i(0,2);
				e_z[5] = 2*eps_z_i(0,1);


				Eigen::Matrix3d sigma_x; sigma_x <<
				elasticity_tensor_.compute_stress<6>(e_x, 0), elasticity_tensor_.compute_stress<6>(e_x, 5), elasticity_tensor_.compute_stress<6>(e_x, 4),
				elasticity_tensor_.compute_stress<6>(e_x, 5), elasticity_tensor_.compute_stress<6>(e_x, 1), elasticity_tensor_.compute_stress<6>(e_x, 3),
				elasticity_tensor_.compute_stress<6>(e_x, 4), elasticity_tensor_.compute_stress<6>(e_x, 3), elasticity_tensor_.compute_stress<6>(e_x, 2);

				Eigen::Matrix3d sigma_y; sigma_y <<
				elasticity_tensor_.compute_stress<6>(e_y, 0), elasticity_tensor_.compute_stress<6>(e_y, 5), elasticity_tensor_.compute_stress<6>(e_y, 4),
				elasticity_tensor_.compute_stress<6>(e_y, 5), elasticity_tensor_.compute_stress<6>(e_y, 1), elasticity_tensor_.compute_stress<6>(e_y, 3),
				elasticity_tensor_.compute_stress<6>(e_y, 4), elasticity_tensor_.compute_stress<6>(e_y, 3), elasticity_tensor_.compute_stress<6>(e_y, 2);

				Eigen::Matrix3d sigma_z; sigma_z <<
				elasticity_tensor_.compute_stress<6>(e_z, 0), elasticity_tensor_.compute_stress<6>(e_z, 5), elasticity_tensor_.compute_stress<6>(e_z, 4),
				elasticity_tensor_.compute_stress<6>(e_z, 5), elasticity_tensor_.compute_stress<6>(e_z, 1), elasticity_tensor_.compute_stress<6>(e_z, 3),
				elasticity_tensor_.compute_stress<6>(e_z, 4), elasticity_tensor_.compute_stress<6>(e_z, 3), elasticity_tensor_.compute_stress<6>(e_z, 2);


				res_k(0) = (sigma_x*eps_x_j).trace();
				res_k(3) = (sigma_x*eps_y_j).trace();
				res_k(6) = (sigma_x*eps_z_j).trace();

				res_k(1) = (sigma_y*eps_x_j).trace();
				res_k(4) = (sigma_y*eps_y_j).trace();
				res_k(7) = (sigma_y*eps_z_j).trace();

				res_k(2) = (sigma_z*eps_x_j).trace();
				res_k(5) = (sigma_z*eps_y_j).trace();
				res_k(8) = (sigma_z*eps_z_j).trace();
			}

			res += res_k * da(k);
		}

		// std::cout<<"res\n"<<res<<"\n"<<std::endl;

		return res;
	}

	void HookeLinearElasticity::compute_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size()*size(), stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size()*size());
			return Eigen::MatrixXd(a);
		});
	}

	void HookeLinearElasticity::compute_von_mises_stresses(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress)
		{
			Eigen::Matrix<double, 1,1> res; res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void HookeLinearElasticity::assign_stress_tensor(const int el_id, const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);


		for(long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose())/2;
			Eigen::MatrixXd sigma(size(), size());

			if(size() == 2)
			{
				std::array<double, 3> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = 2*strain(0,1);

				sigma <<
				elasticity_tensor_.compute_stress<3>(eps, 0), elasticity_tensor_.compute_stress<3>(eps, 2),
				elasticity_tensor_.compute_stress<3>(eps, 2), elasticity_tensor_.compute_stress<3>(eps, 1);
			}
			else
			{
				std::array<double, 6> eps;
				eps[0] = strain(0,0);
				eps[1] = strain(1,1);
				eps[2] = strain(2,2);
				eps[3] = 2*strain(1,2);
				eps[4] = 2*strain(0,2);
				eps[5] = 2*strain(0,1);

				sigma <<
				elasticity_tensor_.compute_stress<6>(eps, 0), elasticity_tensor_.compute_stress<6>(eps, 5), elasticity_tensor_.compute_stress<6>(eps, 4),
				elasticity_tensor_.compute_stress<6>(eps, 5), elasticity_tensor_.compute_stress<6>(eps, 1), elasticity_tensor_.compute_stress<6>(eps, 3),
				elasticity_tensor_.compute_stress<6>(eps, 4), elasticity_tensor_.compute_stress<6>(eps, 3), elasticity_tensor_.compute_stress<6>(eps, 2);
			}

			all.row(p) = fun(sigma);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	HookeLinearElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
				assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;


		if(size() == 2)
			autogen::hooke_2d_function(pt, elasticity_tensor_, res);
		else if(size() == 3)
			autogen::hooke_3d_function(pt, elasticity_tensor_, res);
		else
			assert(false);

		return res;
	}

}
