#include "MultiscaleRB.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>

#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <unsupported/Eigen/MatrixFunctions>
#include <finitediff.hpp>

std::shared_ptr<polyfem::State> state;

namespace polyfem::assembler
{
	namespace {
		// uniform sampling on unit sphere
		void sample_on_sphere(Eigen::MatrixXd &directions, const int dim, const int n_samples) 
		{
			assert(dim == 2);
			directions.setZero(n_samples, dim);

			for (int i = 0; i < n_samples; i++)
			{
				const int theta = (double)i / n_samples * 2 * M_PI;
				directions(i, 0) = std::cos(theta);
				directions(i, 1) = std::sin(theta);
			}
		}

		void get_orthonomal_basis(std::vector<Eigen::MatrixXd> &basis, const int dim)
		{
			if (dim == 2)
			{
				basis.resize(2);
				basis[0] = std::pow(1./2, 1./2) * Eigen::MatrixXd{{1, 0}, {0, -1}};
				basis[1] = std::pow(1./2, 1./2) * Eigen::MatrixXd{{0, 1}, {1, 0}};
			}
			else if (dim == 3)
			{
				basis.resize(5);
				basis[0] = std::pow(1./6, 1./2) * Eigen::MatrixXd{{2,0,0},{0,-1,0},{0,0,-1}};
				basis[1] = std::pow(1./2, 1./2) * Eigen::MatrixXd{{0,0,0},{0,1,0},{0,0,-1}};
				basis[2] = std::pow(1./2, 1./2) * Eigen::MatrixXd{{0,1,0},{1,0,0},{0,0,0}};
				basis[3] = std::pow(1./2, 1./2) * Eigen::MatrixXd{{0,0,1},{0,0,0},{1,0,0}};
				basis[4] = std::pow(1./2, 1./2) * Eigen::MatrixXd{{0,0,0},{0,0,1},{0,1,0}};
			}
		}
	}

	MultiscaleRB::MultiscaleRB()
	{
	}

	MultiscaleRB::~MultiscaleRB()
	{
		
	}

	void MultiscaleRB::create_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads)
	{
		state = std::make_shared<polyfem::State>(utils::get_n_threads(), true);
		state->init(unit_cell_args, false, "", false);
		state->load_mesh(false);
		if (state->mesh == nullptr)
			log_and_throw_error("No microstructure mesh found!");
		state->stats.compute_mesh_stats(*state->mesh);
		state->build_basis();

		logger().info("Compute deformation gradient dataset, in total {} solves...", def_grads.size());

		Eigen::MatrixXd sols;
		sols.setZero(state->n_bases * size(), def_grads.size());
	
		utils::maybe_parallel_for(def_grads.size(), [&](int start, int end, int thread_id) {
			for (int idx = start; idx < end; idx++)
			// for (int idx = 0; idx < def_grads.size(); idx++)
			{
				// solve fluctuation field
				Eigen::MatrixXd grad = def_grads[idx] - Eigen::MatrixXd::Identity(size(), size());
				sols.col(idx) = state->solve_homogenized_field(grad) + state->generate_linear_field(grad);
			}
		});

		logger().info("Compute covarianece matrix...");

		// compute covariance matrix
		StiffnessMatrix laplacian;
		state->assembler.assemble_problem("Laplacian", state->mesh->is_volume(), state->n_bases, state->bases, state->geom_bases(), state->ass_vals_cache, laplacian);

		Eigen::MatrixXd covariance;
		// covariance = Eigen::MatrixXd::Identity(def_grads.size(), def_grads.size());
		covariance.setZero(def_grads.size(), def_grads.size());
		for (int i = 0; i < covariance.rows(); i++)
		{
			Eigen::MatrixXd sol_i = utils::unflatten(sols.col(i), size());
			for (int j = 0; j <= i; j++)
			{
				Eigen::MatrixXd sol_j = utils::unflatten(sols.col(j), size());
				for (int d = 0; d < size(); d++)
					covariance(i, j) += sol_i.col(d).transpose() * laplacian * sol_j.col(d);
			}
		}
		for (int i = 0; i < covariance.rows(); i++)
			for (int j = i + 1; j < covariance.cols(); j++)
				covariance(i, j) = covariance(j, i);

		// Schur Decomposition
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covariance);
		Eigen::MatrixXd ev = es.eigenvectors();

		// Setup reduced basis
		const int Ns = 5;
		reduced_basis = sols * ev.rightCols(Ns);
		for (int j = 0; j < Ns; j++)
			reduced_basis.col(j) /= sqrt(es.eigenvalues()(ev.cols() - Ns + j));

		for (int i = 0; i < es.eigenvalues().size() - 1; i++)
			assert(es.eigenvalues()(i) <= es.eigenvalues()(i+1));

		logger().info("Reduced basis created!");
	}

	void MultiscaleRB::projection(const Eigen::MatrixXd &F, Eigen::VectorXd &xi) const
	{
		const int ndof = reduced_basis.cols();

		xi.setZero(ndof);

		Eigen::MatrixXd x0 = state->generate_linear_field(F);
		Eigen::MatrixXd x = x0;

		Eigen::MatrixXd gradv(ndof, 1), grad;
		utils::SpareMatrixCache mat_cache_;
		StiffnessMatrix hessian;
		Eigen::MatrixXd hessianv(ndof, ndof);

		const double eps = 1e-8;
		const int max_iter = 100;

		state->assembler.assemble_energy_gradient(
			state->formulation(), size() == 3, state->n_bases, state->bases, state->geom_bases(),
			state->ass_vals_cache, 0, x, x, grad);
		gradv = reduced_basis.transpose() * grad;

		int iter = 0;
		while (gradv.norm() > eps && iter++ < max_iter) {
			// evaluate hessian
			state->assembler.assemble_energy_hessian(
				state->formulation(), size() == 3, state->n_bases, false, state->bases,
				state->geom_bases(), state->ass_vals_cache, 0, x, x, mat_cache_, hessian);
			hessianv = reduced_basis.transpose() * hessian * reduced_basis;

			// Eigen::MatrixXd fhess;
			// fd::finite_jacobian(
			// 	xi,
			// 	[this, &x0](const Eigen::VectorXd &xi_) -> Eigen::VectorXd {
			// 		Eigen::MatrixXd grad_;
			// 		Eigen::MatrixXd tmp = x0 + this->reduced_basis * xi_;
			// 		state->assembler.assemble_energy_gradient(
			// 			state->formulation(), this->size() == 3, state->n_bases, state->bases, state->geom_bases(),
			// 			state->ass_vals_cache, 0, tmp, tmp, grad_);
			// 		return this->reduced_basis.transpose() * grad_;
			// 	},
			// 	fhess);

			// if (!fd::compare_hessian(hessianv, fhess))
			// {
			// 	logger().error("RB hessian doesn't match with FD!");
			// }

			// newton step
			xi = xi - hessianv.ldlt().solve(gradv);
			x = x0 + reduced_basis * xi;

			state->assembler.assemble_energy_gradient(
				state->formulation(), size() == 3, state->n_bases, state->bases, state->geom_bases(),
				state->ass_vals_cache, 0, x, x, grad);
			gradv = reduced_basis.transpose() * grad;
		}

		if (gradv.norm() > eps)
			logger().error("Newton failed to converge on finding RB coefficients!");
	}

	void MultiscaleRB::homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &stiffness) const
	{
		// polar decomposition
		Eigen::JacobiSVD<Eigen::MatrixXd> svd; // def_grad == svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
		svd.compute(def_grad, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd R = svd.matrixU() * svd.matrixV().transpose();
		Eigen::MatrixXd Ubar = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
		if (svd.singularValues().minCoeff() <= 0)
			logger().error("Negative Deformation Gradient!");

		Eigen::MatrixXd x;
		{
			Eigen::VectorXd xi;
			projection(Ubar, xi);
			x = state->generate_linear_field(Ubar) + reduced_basis * xi;
		}

		const auto &bases = state->bases;
		const auto &gbases = state->geom_bases();
		const int n_elements = int(bases.size());

		RowVectorNd min, max;
		state->mesh->bounding_box(min, max);
		const double volume = (max - min).prod();

		// effective energy = average energy over unit cell
		energy = state->assembler.assemble_energy(state->formulation(), size() == 3, bases, gbases, state->ass_vals_cache, 0, x, x) / volume;

		// effective stress = average stress over unit cell
		{
			stress.setZero(size(), size());
			Eigen::MatrixXd stresses, avg_stress, tmp;

			for (int e = 0; e < n_elements; ++e)
			{
				assembler::ElementAssemblyValues vals;
				state->ass_vals_cache.compute(e, size() == 3, bases[e], gbases[e], vals);

				// io::Evaluator::interpolate_at_local_vals(e, size(), size(), vals, x, u, grad_u);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				state->assembler.compute_tensor_value(state->formulation(), e, bases[e], gbases[e], quadrature.points, x, stresses);
				tmp = stresses.transpose() * da;
				avg_stress = Eigen::Map<Eigen::MatrixXd>(tmp.data(), size(), size());
				stress += avg_stress;
			}

			stress = R * stress / volume;
		}

		// \bar{C}^{RB} = < C^{RB} > - \sum_{i,j} D^{-1}_{ij} < C^{RB} \cdot B^{(i)} > \cross < B^{(j)} \cdot C^{RB} >
		// D_{ij} = < B^{(i)} \cdot C \cdot B^{(j)} >
		{
			stiffness.setZero(size()*size(), size()*size());
			Eigen::MatrixXd stiffness_no_rotation = stiffness;

			Eigen::MatrixXd stiffnesses;
			for (int e = 0; e < n_elements; ++e)
			{
				assembler::ElementAssemblyValues vals;
				state->ass_vals_cache.compute(e, size() == 3, bases[e], gbases[e], vals);

				// io::Evaluator::interpolate_at_local_vals(e, size(), size(), vals, x, u, grad_u);

				const quadrature::Quadrature &quadrature = vals.quadrature;
				Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

				state->assembler.compute_stiffness_value(state->formulation(), e, bases[e], gbases[e], quadrature.points, x, stiffnesses);
				
			}

			for (int i = 0; i < size(); i++) for (int j = 0; j < size(); j++)
			for (int k = 0; k < size(); k++) for (int l = 0; l < size(); l++)
			for (int m = 0; m < size(); m++) for (int n = 0; n < size(); n++)
				stiffness(i * size() + j, k * size() + l) += R(i, m) * stiffness_no_rotation(m * size() + j, n * size() + l) * R(k, n);
		}
	}

	void MultiscaleRB::sample_def_grads(std::vector<Eigen::MatrixXd> &def_grads) const
	{
		const int unit_sphere_dim = (size() == 2) ? 2 : 5;
		Eigen::MatrixXd directions;
		sample_on_sphere(directions, unit_sphere_dim, n_sample_dir);
		
		const int Ndir = directions.rows();
		const int Ndet = sample_det.size();
		const int Namp = sample_amp.size();
		
		def_grads.clear();
		def_grads.resize(Ndet * Namp * Ndir);

		std::vector<Eigen::MatrixXd> Y;
		get_orthonomal_basis(Y, size());
		assert(Y.size() == unit_sphere_dim);

		int idx = 0;
		for (int n = 0; n < Ndir; n++)
		{
			Eigen::MatrixXd tmp1(size(), size());
			tmp1.setZero();
			for (int d = 0; d < unit_sphere_dim; d++)
				tmp1 += Y[d] * directions(n, d);
			
			for (int p = 0; p < Namp; p++)
			{
				Eigen::MatrixXd tmp2 = (sample_amp(p) * tmp1).exp();
				for (int m = 0; m < Ndet; m++)
				{	
					// equation (45)
					def_grads[idx] = std::pow(sample_det(m), 1./3) * tmp2;
					idx++;
				}
			}
		}
	}

	void MultiscaleRB::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		if (params.contains("microstructure"))
		{
			unit_cell_args = params["microstructure"];

			assert(params["det_samples"].is_array());
			assert(params["amp_samples"].is_array());

			sample_det = params["det_samples"];
			sample_amp = params["amp_samples"];
			n_sample_dir = params["n_dir_samples"];

			std::vector<Eigen::MatrixXd> def_grads;
			sample_def_grads(def_grads);
			create_reduced_basis(def_grads);
		}
	}

	void MultiscaleRB::set_size(const int size)
	{
		size_ = size;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	MultiscaleRB::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;

		double lambda, mu;
		// TODO!
		params_.lambda_mu(0, 0, 0, pt(0).getValue(), pt(1).getValue(), size() == 2 ? 0. : pt(2).getValue(), 0, lambda, mu);

		if (size() == 2)
			autogen::neo_hookean_2d_function(pt, lambda, mu, res);
		else if (size() == 3)
			autogen::neo_hookean_3d_function(pt, lambda, mu, res);
		else
			assert(false);

		return res;
	}

	Eigen::VectorXd
	MultiscaleRB::assemble_grad(const NonLinearAssemblerData &data) const
	{
		const auto &bs = data.vals.basis_values;
		Eigen::MatrixXd local_disp;
		local_disp.setZero(bs.size(), size());
		for (size_t i = 0; i < bs.size(); ++i)
		{
			const auto &b = bs[i];
			for (size_t ii = 0; ii < b.global.size(); ++ii)
				for (int d = 0; d < size(); ++d)
					local_disp(i, d) += b.global[ii].val * data.x(b.global[ii].index * size() + d);
		}

		Eigen::MatrixXd G;
		G.setZero(bs.size(), size());

		const int n_pts = data.da.size();
		Eigen::MatrixXd def_grad(size(), size()), stress_tensor, stiffness_tensor;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(bs.size(), size());
			for (size_t i = 0; i < bs.size(); ++i)
				grad.row(i) = bs[i].grad.row(p);

			Eigen::MatrixXd delF_delU = grad * data.vals.jac_it[p];

			def_grad.setZero();
			def_grad = local_disp.transpose() * delF_delU + Eigen::MatrixXd::Identity(size(), size());

			double energy = 0;
			homogenization(def_grad, energy, stress_tensor, stiffness_tensor);

			// {
			// 	Eigen::VectorXd fgrad, grad;
			// 	Eigen::VectorXd x0 = Eigen::Map<Eigen::MatrixXd>(def_grad.data(), def_grad.size(), 1);
			// 	fd::finite_gradient(
			// 		x0, [this](const Eigen::VectorXd &x) -> double 
			// 		{ 
			// 			Eigen::MatrixXd F = x;
			// 			F = Eigen::Map<Eigen::MatrixXd>(F.data(), state->mesh->dimension(), state->mesh->dimension());
			// 			double val;
			// 			Eigen::MatrixXd stress, stiffness;
			// 			this->homogenization(F, val, stress, stiffness);
			// 			return val; 
			// 		}, fgrad);

			// 	grad = Eigen::Map<Eigen::MatrixXd>(stress_tensor.data(), stress_tensor.size(), 1);
			// 	if (!fd::compare_gradient(grad, fgrad))
			// 	{
			// 		std::cout << "Gradient: " << grad.transpose() << std::endl;
			// 		std::cout << "Finite gradient: " << fgrad.transpose() << std::endl;
			// 		log_and_throw_error("Gradient mismatch");
			// 	}
			// }

			G += delF_delU * stress_tensor * data.da(p);
		}

		Eigen::MatrixXd G_T = G.transpose();

		Eigen::VectorXd temp(Eigen::Map<Eigen::VectorXd>(G_T.data(), G_T.size()));

		return temp;
	}

	Eigen::MatrixXd
	MultiscaleRB::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const auto &bs = data.vals.basis_values;
		Eigen::MatrixXd hessian;
		hessian.setZero(bs.size() * size(), bs.size() * size());
		Eigen::MatrixXd local_disp;
		local_disp.setZero(bs.size(), size());
		for (size_t i = 0; i < bs.size(); ++i)
		{
			const auto &b = bs[i];
			for (size_t ii = 0; ii < b.global.size(); ++ii)
				for (int d = 0; d < size(); ++d)
					local_disp(i, d) += b.global[ii].val * data.x(b.global[ii].index * size() + d);
		}

		const int n_pts = data.da.size();

		Eigen::MatrixXd def_grad(size(), size());
		Eigen::MatrixXd stress_tensor, hessian_temp;
		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::MatrixXd grad(bs.size(), size());

			for (size_t i = 0; i < bs.size(); ++i)
				grad.row(i) = bs[i].grad.row(p);

			Eigen::MatrixXd jac_it = data.vals.jac_it[p];
			
			def_grad = local_disp.transpose() * grad * jac_it + Eigen::MatrixXd::Identity(size(), size());

			double energy = 0;
			homogenization(def_grad, energy, stress_tensor, hessian_temp);
			// hessian_temp(i + j * size(), k + l * size());

			Eigen::MatrixXd delF_delU_tensor(jac_it.size(), grad.size());
			Eigen::MatrixXd temp;
			for (size_t j = 0; j < local_disp.cols(); ++j)
			{
				temp.setZero(size(), size());
				for (size_t i = 0; i < local_disp.rows(); ++i)
				{
					temp.row(j) = grad.row(i);
					temp = temp * jac_it;
					Eigen::VectorXd temp_flattened(Eigen::Map<Eigen::VectorXd>(temp.data(), temp.size()));
					delF_delU_tensor.col(i * size() + j) = temp_flattened;
				}
			}

			hessian += delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor * data.da(p);
		}

		return hessian;
	}

	void MultiscaleRB::compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
			return Eigen::MatrixXd(a);
		});
	}

	void MultiscaleRB::compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::Matrix<double, 1, 1> res;
			res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	void MultiscaleRB::assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			const Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(size(), size()) + displacement_grad;

			Eigen::MatrixXd stress_tensor, stiffness_tensor;
			double energy = 0;
			homogenization(def_grad, energy, stress_tensor, stiffness_tensor);

			all.row(p) = fun(stress_tensor);
		}
	}

	double MultiscaleRB::compute_energy(const NonLinearAssemblerData &data) const
	{
			Eigen::MatrixXd local_disp;
			local_disp.setZero(data.vals.basis_values.size(), size());
			for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
			{
				const auto &bs = data.vals.basis_values[i];
				for (size_t ii = 0; ii < bs.global.size(); ++ii)
					for (int d = 0; d < size(); ++d)
						local_disp(i, d) += bs.global[ii].val * data.x(bs.global[ii].index * size() + d);
			}
			
			double energy = 0;
			const int n_pts = data.da.size();

			Eigen::MatrixXd def_grad(size(), size());
			for (long p = 0; p < n_pts; ++p)
			{
				def_grad.setZero();

				for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
				{
					const auto &bs = data.vals.basis_values[i];

					for (int d = 0; d < size(); ++d)
						for (int c = 0; c < size(); ++c)
							def_grad(d, c) += bs.grad(p, c) * local_disp(i, d);
				}

				Eigen::MatrixXd jac_it(size(), size());
				for (long k = 0; k < jac_it.size(); ++k)
					jac_it(k) = data.vals.jac_it[p](k);
				
				def_grad = def_grad * jac_it + Eigen::MatrixXd::Identity(size(), size());
				
				double val = 0;
				Eigen::MatrixXd stress_tensor, stiffness_tensor;
				homogenization(def_grad, val, stress_tensor, stiffness_tensor);

				energy += val * data.da(p);
			}

			return energy;
	}

} // namespace polyfem::assembler
