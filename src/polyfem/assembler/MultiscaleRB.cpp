#include "MultiscaleRB.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/solver/DenseNewtonDescentSolver.hpp>

#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
#include <finitediff.hpp>
#include <filesystem>

#ifdef POLYSOLVE_WITH_SPECTRA
#include <SymEigsSolver.h>
#endif

namespace polyfem::assembler
{
	namespace {
		// uniform sampling on unit sphere
		void sample_on_sphere(Eigen::MatrixXd &directions, const int dim, const int n_samples) 
		{
			assert(dim == 2);
			directions.setZero(n_samples, dim);

			double dtheta = 1.0 / n_samples * M_PI * 2;
			for (int i = 0; i < n_samples; i++)
			{
				const double theta = i * dtheta;
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
		
		// S = sqrt(A), A is SPD, compute dS/dA
		Eigen::MatrixXd matrix_sqrt_grad(const Eigen::MatrixXd &S)
		{
			const int dim = S.rows();
			assert(S.cols() == dim);
			Eigen::MatrixXd id = Eigen::MatrixXd::Identity(dim, dim);

			return (Eigen::kroneckerProduct(S, id) + Eigen::kroneckerProduct(id, S)).inverse();
		}

		void polar_decomposition(const Eigen::MatrixXd &F, Eigen::MatrixXd &R, Eigen::MatrixXd &U)
		{
			const int dim = F.rows();
			assert(F.cols() == dim);
			Eigen::JacobiSVD<Eigen::MatrixXd> svd;
			svd.compute(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
			R = svd.matrixU() * svd.matrixV().transpose();
			U = R.transpose() * F;

			if (svd.singularValues().minCoeff() <= 0)
				logger().error("Negative Deformation Gradient!");
		}

		bool delta(int i, int j)
		{
			return i == j;
		}

		void my_polar_decomposition_grad(const Eigen::MatrixXd &F, const Eigen::MatrixXd &R, const Eigen::MatrixXd &U, Eigen::MatrixXd &dUdF)
		{
			const int dim = F.rows();
			assert(F.cols() == dim);

			Eigen::MatrixXd dU_dATA = matrix_sqrt_grad(U);

			Eigen::MatrixXd dATA_dA;
			dATA_dA.setZero(dim*dim, dim*dim);
			for (int i = 0; i < dim; i++)
			for (int p = 0; p < dim; p++)
			for (int q = 0; q < dim; q++)
			{
				dATA_dA(q + i * dim, p + q * dim) += F(p, i);
				dATA_dA(i + q * dim, p + q * dim) += F(p, i);
			}

			dUdF = dU_dATA * dATA_dA;

			{
				Eigen::MatrixXd tmp = dUdF;
				for (int i = 0; i < dim; i++)
				for (int j = 0; j < dim; j++)
				for (int k = 0; k < dim; k++)
				for (int l = 0; l < dim; l++)
					dUdF(i * dim + j, k * dim + l) = tmp(i + j * dim, k + l * dim);
			}
		}

		bool compare_matrix(
			const Eigen::MatrixXd& x,
			const Eigen::MatrixXd& y,
			const double test_eps = 1e-4)
		{
			assert(x.rows() == y.rows());

			bool same = true;
			double scale = std::max(x.norm(), y.norm());
			double error = (x - y).norm();
			
			// std::cout << "error: " << error << " scale: " << scale << "\n";

			if (error > scale * test_eps)
				same = false;

			return same;
		}

		Eigen::MatrixXd generate_linear_field(const State &state, const Eigen::MatrixXd &grad)
		{
			const int problem_dim = grad.rows();
			const int dim = state.mesh->dimension();
			assert(dim == grad.cols());

			Eigen::MatrixXd func(state.n_bases * problem_dim, 1);
			func.setZero();

			for (int i = 0; i < state.n_bases; i++)
			{
				func.block(i * problem_dim, 0, problem_dim, 1) = grad * state.mesh_nodes->node_position(i).transpose();
			}

			return func;
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
		logger().info("Compute deformation gradient dataset, in total {} solves...", def_grads.size());

		Eigen::MatrixXd sols;
		sols.setZero(state->n_bases * size(), def_grads.size());
	
		utils::maybe_parallel_for(def_grads.size(), [&](int start, int end, int thread_id) {
			Eigen::MatrixXd tmp;
			for (int idx = start; idx < end; idx++)
			// for (int idx = 0; idx < def_grads.size(); idx++)
			{
				// solve fluctuation field
				Eigen::MatrixXd grad = def_grads[idx] - Eigen::MatrixXd::Identity(size(), size());
				state->solve_homogenized_field(grad, tmp);
				sols.col(idx) = tmp - generate_linear_field(*state, grad);
			}
		});

		logger().info("Compute covariance matrix...");

		// compute covariance matrix
		StiffnessMatrix laplacian;
		state->assembler.assemble_problem("Laplacian", state->mesh->is_volume(), state->n_bases, state->bases, state->geom_bases(), state->ass_vals_cache, laplacian);

		Eigen::MatrixXd covariance;
		// covariance = Eigen::MatrixXd::Identity(def_grads.size(), def_grads.size());
		covariance.setZero(def_grads.size(), def_grads.size());
		utils::maybe_parallel_for(covariance.rows(), [&](int start, int end, int thread_id) {
			Eigen::MatrixXd sol_i, sol_j;
			for (int i = start; i < end; i++)
			{
				sol_i = utils::unflatten(sols.col(i), size());
				for (int j = 0; j <= i; j++)
				{
					sol_j = utils::unflatten(sols.col(j), size());
					for (int d = 0; d < size(); d++)
						covariance(i, j) += sol_i.col(d).transpose() * laplacian * sol_j.col(d);
				}
			}
		});
		for (int i = 0; i < covariance.rows(); i++)
			for (int j = i + 1; j < covariance.cols(); j++)
				covariance(i, j) = covariance(j, i);

		covariance /= microstructure_volume;

		logger().info("Schur decomposition...");

		// Schur Decomposition
		const int Ns = n_reduced_basis;
		Eigen::MatrixXd eigen_vectors;
		Eigen::VectorXd eigen_values;
		{
#ifdef POLYSOLVE_WITH_SPECTRA
			Spectra::DenseSymMatProd<double> op(covariance);
			Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<double>>  eigs(&op, Ns, std::min(Ns*2, (int)covariance.rows()));
			eigs.init();
			int nconv = eigs.compute();
			if(eigs.info() == Spectra::COMPUTATION_INFO::SUCCESSFUL)
			{
				eigen_values = eigs.eigenvalues();
				eigen_vectors = eigs.eigenvectors();
			}
			else
				log_and_throw_error("Spectra failed to converge!");

			reduced_basis = sols * eigen_vectors;
			for (int j = 0; j < Ns; j++)
				reduced_basis.col(j) /= sqrt(eigen_values(j));

			Eigen::MatrixXd residual = covariance * eigen_vectors - eigen_vectors * eigen_values.asDiagonal();
			logger().info("eigen vector error: {}, eigen values: {}", residual.array().abs().maxCoeff(), eigen_values.transpose());
#else
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(covariance);
			eigen_vectors = eigs.eigenvectors();
			eigen_values = eigs.eigenvalues();

			reduced_basis = sols * eigen_vectors.rightCols(Ns);
			for (int j = 0; j < Ns; j++)
				reduced_basis.col(j) /= sqrt(eigen_values(eigen_vectors.cols() - Ns + j));

			for (int i = 0; i < eigen_values.size() - 1; i++)
				if (eigen_values(i) > eigen_values(i+1))
				{
					std::cout << eigen_values.transpose() << "\n";
					log_and_throw_error("Eigenvalues not in increase order in Schur decomposition!");
				}
#endif
		}

		if (std::isnan(reduced_basis.norm()))
		{
			logger().error("Covariance eigenvalues: {}", eigen_values.transpose());
			log_and_throw_error("NAN in reduced basis!");
		}

		// for (int i = 0; i < reduced_basis.cols(); i++)
		// {
		// 	state->sol = reduced_basis.col(i);
		// 	state->out_geom.export_data(
		// 		*state,
		// 		!state->args["time"].is_null(),
		// 		0, 0,
		// 		io::OutGeometryData::ExportOptions(state->args, state->mesh->is_linear(), state->problem->is_scalar(), state->solve_export_to_file),
		// 		"rb_" + std::to_string(i) + ".vtu","","","","",
		// 		state->is_contact_enabled(), state->solution_frames);
		// }

		logger().info("Reduced basis created!");
	}

	void MultiscaleRB::projection(const Eigen::MatrixXd &F, Eigen::MatrixXd &x) const
	{
		Eigen::VectorXd xi;
		xi.setZero(reduced_basis.cols());

		std::shared_ptr<MultiscaleRBProblem> nl_problem = std::make_shared<MultiscaleRBProblem>(state, reduced_basis);
		nl_problem->set_linear_disp(generate_linear_field(*state, F));
		std::shared_ptr<cppoptlib::NonlinearSolver<MultiscaleRBProblem>> nlsolver = std::make_shared<cppoptlib::DenseNewtonDescentSolver<MultiscaleRBProblem>>(
				state->args["solver"]["nonlinear"], state->args["solver"]["linear"]);
		nlsolver->disable_logging();
		nlsolver->minimize(*nl_problem, xi);
		x = nl_problem->coeff_to_field(xi);

		// {
		// 	Eigen::MatrixXd avg = homogenize_def_grad(x);

		// 	double err = (F - avg).norm() / F.norm();
		// 	if (err > 1e-6)
		// 		logger().error("def grad err: {}", err);
		// }

		// {
		// 	static int idx_proj = 0;
		// 	state->sol = x;
		// 	state->out_geom.export_data(
		// 		*state,
		// 		!state->args["time"].is_null(),
		// 		0, 0,
		// 		io::OutGeometryData::ExportOptions(state->args, state->mesh->is_linear(), state->problem->is_scalar(), state->solve_export_to_file),
		// 		"proj_" + std::to_string(idx_proj) + ".vtu",
		// 		"", // nodes_path,
		// 		"", // solution_path,
		// 		"", // stress_path,
		// 		"", // mises_path,
		// 		state->is_contact_enabled(), state->solution_frames);
		// 	idx_proj++;
		// }

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
	}

	void MultiscaleRB::homogenize_stiffness(const Eigen::MatrixXd &x, Eigen::MatrixXd &stiffness) const
	{
		double time;
		POLYFEM_SCOPED_TIMER("homogenize variables", time);

		const auto &bases = state->bases;
		const auto &gbases = state->geom_bases();

		Eigen::MatrixXd avg_stiffness, term2;
		avg_stiffness.setZero(size()*size(), size()*size());
		term2.setZero(size()*size(), size()*size());

		Eigen::MatrixXd stiffnesses, tmp;
		Eigen::MatrixXd u, grad_u;
		Eigen::MatrixXd CB;
		CB.setZero(size()*size(), n_reduced_basis);
		for (int e = 0; e < bases.size(); ++e)
		{
			assembler::ElementAssemblyValues vals;
			state->ass_vals_cache.compute(e, size() == 3, bases[e], gbases[e], vals);

			const quadrature::Quadrature &quadrature = vals.quadrature;
			Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

			state->assembler.compute_stiffness_value(state->formulation(), vals, quadrature.points, x, stiffnesses);
			avg_stiffness += utils::unflatten(stiffnesses.transpose() * da, size()*size());

			for (int i = 0; i < n_reduced_basis; i++)
			{
				io::Evaluator::interpolate_at_local_vals(e, size(), size(), vals, reduced_basis.col(i), u, grad_u);

				for (int a = 0, idx = 0; a < size(); a++)
				for (int b = 0; b < size(); b++)
				for (int k = 0; k < size(); k++)
				for (int l = 0; l < size(); l++)
				{
					CB(a * size() + b, i) += (stiffnesses.col(idx).array() * grad_u.col(k * size() + l).array() * da.array()).sum();
					idx++;
				}
			}
		}
		avg_stiffness /= microstructure_volume;

		// compute term2 given CB
		{
			Eigen::MatrixXd Dinv;
			{
				Eigen::MatrixXd hessian;
				state->assembler.assemble_energy_hessian(
					state->formulation(), size() == 3, state->n_bases, false, state->bases,
					state->geom_bases(), state->ass_vals_cache, 0, x, x, reduced_basis, hessian);

				Dinv = hessian.inverse();
			}
			term2 = CB * Dinv * CB.transpose() / microstructure_volume;
		}

		stiffness = avg_stiffness - term2;
	}

	void MultiscaleRB::homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &stiffness) const
	{
		Eigen::MatrixXd R, Ubar, dUdF;
		{
			polar_decomposition(def_grad, R, Ubar);
			my_polar_decomposition_grad(def_grad, R, Ubar, dUdF);
		}

		Eigen::MatrixXd x;
		{
			double time;
			POLYFEM_SCOPED_TIMER("coefficient newton", time);
			Eigen::MatrixXd disp_grad = Ubar - Eigen::MatrixXd::Identity(size(), size());
			projection(disp_grad, x);
		}

		// effective energy = average energy over unit cell
		energy = homogenize_energy(x);

		// effective stress = average stress over unit cell
		Eigen::MatrixXd stress_no_rotation;
		homogenize_stress(x, stress_no_rotation);
		stress_no_rotation = utils::flatten(stress_no_rotation);

		// effective stiffness
		// \bar{C}^{RB} = < C^{RB} > - \sum_{i,j} (D^{-1})_{ij} < C^{RB} \cdot B^{(i)} > \cross < B^{(j)} \cdot C^{RB} >
		// D_{ij} = < B^{(i)} \cdot C \cdot B^{(j)} >
		Eigen::MatrixXd stiffness_no_rotation;
		homogenize_stiffness(x, stiffness_no_rotation);

		// paper version
		// stress = R * stress_no_rotation;
		// stiffness.setZero(size()*size(), size()*size());
		// for (int i = 0; i < size(); i++) for (int j = 0; j < size(); j++)
		// for (int k = 0; k < size(); k++) for (int l = 0; l < size(); l++)
		// for (int m = 0; m < size(); m++) for (int n = 0; n < size(); n++)
		// {
		// 	stiffness(i * size() + j, k * size() + l) += R(i, m) * stiffness_no_rotation(m * size() + j, n * size() + l) * R(k, n);
		// }
		
		// my version
		stress = utils::unflatten(dUdF.transpose() * stress_no_rotation, size());
		stiffness = dUdF.transpose() * stiffness_no_rotation * dUdF;
		{
			Eigen::MatrixXd fjacobian;
			fd::finite_jacobian(
			utils::flatten(def_grad),
			[this, &stress_no_rotation](const Eigen::VectorXd &x) -> Eigen::VectorXd {
				Eigen::MatrixXd F = utils::unflatten(x, this->size());
				Eigen::MatrixXd R, Ubar, dUdF;
				polar_decomposition(F, R, Ubar);
				my_polar_decomposition_grad(F, R, Ubar, dUdF);
				Eigen::VectorXd tmp = dUdF.transpose() * stress_no_rotation;
				return tmp;
			},
			fjacobian);

			stiffness += fjacobian;
		}

		// std::cout << "cauchy stress symmetry: " << (stress_no_rotation * Ubar - Ubar * stress_no_rotation.transpose()).norm() / (Ubar * stress_no_rotation.transpose()).norm() << "\n";
	}

	void MultiscaleRB::homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress) const
	{
		Eigen::MatrixXd R, Ubar, dUdF;
		{
			polar_decomposition(def_grad, R, Ubar);
			my_polar_decomposition_grad(def_grad, R, Ubar, dUdF);
		}

		Eigen::MatrixXd x;
		{
			double time;
			POLYFEM_SCOPED_TIMER("coefficient newton", time);
			Eigen::MatrixXd disp_grad = Ubar - Eigen::MatrixXd::Identity(size(), size());
			projection(disp_grad, x);
		}

		// effective energy = average energy over unit cell
		energy = homogenize_energy(x);

		// effective stress = average stress over unit cell
		Eigen::MatrixXd stress_no_rotation;
		homogenize_stress(x, stress_no_rotation);
		stress_no_rotation = utils::flatten(stress_no_rotation);
		
		// my version
		stress = utils::unflatten(dUdF.transpose() * stress_no_rotation, size());

		// std::cout << "cauchy stress symmetry: " << (stress_no_rotation * Ubar - Ubar * stress_no_rotation.transpose()).norm() / (Ubar * stress_no_rotation.transpose()).norm() << "\n";
	}

	void MultiscaleRB::homogenization(const Eigen::MatrixXd &def_grad, double &energy) const
	{
		Eigen::MatrixXd R, Ubar, dUdF;
		{
			polar_decomposition(def_grad, R, Ubar);
			my_polar_decomposition_grad(def_grad, R, Ubar, dUdF);
		}

		Eigen::MatrixXd x;
		{
			double time;
			POLYFEM_SCOPED_TIMER("coefficient newton", time);
			Eigen::MatrixXd disp_grad = Ubar - Eigen::MatrixXd::Identity(size(), size());
			projection(disp_grad, x);
		}

		// effective energy = average energy over unit cell
		energy = homogenize_energy(x);
	}

	void MultiscaleRB::sample_def_grads(const Eigen::VectorXd &sample_det, const Eigen::VectorXd &sample_amp, const int n_sample_dir, std::vector<Eigen::MatrixXd> &def_grads) const
	{		
		const int unit_sphere_dim = (size() == 2) ? 2 : 5;
		Eigen::MatrixXd directions;
		sample_on_sphere(directions, unit_sphere_dim, n_sample_dir);

		const int Ndir = directions.rows();
		const int Ndet = sample_det.size();
		const int Namp = sample_amp.size();
		
		def_grads.clear();
		def_grads.resize(Ndet * Namp * Ndir);

		if (def_grads.size() == 0)
			log_and_throw_error("Zero deformation gradient sampling!");

		std::vector<Eigen::MatrixXd> Y;
		get_orthonomal_basis(Y, size());
		assert(Y.size() == unit_sphere_dim);

		int idx = 0;
		Eigen::MatrixXd tmp1, tmp2;
		for (int n = 0; n < Ndir; n++)
		{
			tmp1.setZero(size(), size());
			for (int d = 0; d < unit_sphere_dim; d++)
				tmp1 += Y[d] * directions(n, d);
			
			for (int p = 0; p < Namp; p++)
			{
				tmp2 = (sample_amp(p) * tmp1).exp();
				for (int m = 0; m < Ndet; m++)
				{	
					// equation (45)
					def_grads[idx] = std::pow(sample_det(m), 1./size()) * tmp2;
					idx++;
				}
			}
		}
	}

	void MultiscaleRB::add_multimaterial(const int index, const json &params)
	{
		Multiscale::add_multimaterial(index, params);

		if (params.contains("type") && params["type"] == name())
		{
			if (params["load_reduced_basis"] != "")
			{
				const std::string path = params["load_reduced_basis"].get<std::string>();
				if (std::filesystem::is_regular_file(path))
				{
					polyfem::io::read_matrix(path, reduced_basis);
					n_reduced_basis = reduced_basis.cols();
					if (reduced_basis.rows() != state->n_bases * state->mesh->dimension())
						log_and_throw_error("Inconsistent dof and reduced basis!");
					logger().info("Read reduced basis from file finished!");
				}
			}
			else
			{
				assert(params["det_samples"].is_array());
				assert(params["amp_samples"].is_array());

				const Eigen::VectorXd sample_det = params["det_samples"];
				const Eigen::VectorXd sample_amp = params["amp_samples"];
				const int n_sample_dir = params["n_dir_samples"];
				
				std::vector<Eigen::MatrixXd> def_grads;
				sample_def_grads(sample_det, sample_amp, n_sample_dir, def_grads);

				n_reduced_basis = params["n_reduced_basis"];
				create_reduced_basis(def_grads);

				if (params["save_reduced_basis"] != "")
				{
					const std::string path = params["save_reduced_basis"].get<std::string>();
					polyfem::io::write_matrix(path, reduced_basis);
					logger().info("Write reduced basis to file finished!");

					exit(0);
				}
			}
		
			if (params.contains("test_det_samples") && params.contains("test_amp_samples") && params.contains("n_test_dir_samples"))
			{
				const Eigen::VectorXd test_sample_det = params["test_det_samples"];
				const Eigen::VectorXd test_sample_amp = params["test_amp_samples"];
				const int n_test_sample_dir = params["n_test_dir_samples"];
				std::vector<Eigen::MatrixXd> def_grads;
				sample_def_grads(test_sample_det, test_sample_amp, n_test_sample_dir, def_grads);
				{
					def_grads.clear();
					Eigen::Matrix2d A;
					for (int i = 0; i >= -100; i--)
					{
						A << 1 + i / 120., 0, 0, 1 + i / 120.; // 1, i / 120.0, i / 120.0, 1;
						def_grads.push_back(A);
					}
				}

				logger().info("Test trained model on another dataset with {} samples!", def_grads.size());

				Eigen::VectorXd energy_err, stress_err;
				test_reduced_basis(def_grads, energy_err, stress_err);
				
				Eigen::MatrixXd data(def_grads.size(), def_grads[0].size() + 2);
				for (int i = 0; i < def_grads.size(); i++)
				{
					data.block(i, 0, 1, def_grads[i].size()) = utils::flatten(def_grads[i]).transpose();
					data(i, def_grads[i].size()) = energy_err(i);
					data(i, def_grads[i].size()+1) = stress_err(i);
				}
				polyfem::io::write_matrix("test.txt", data);

				exit(0);
			}
		}
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
			// {
			// 	Eigen::MatrixXd fhess;
			// 	Eigen::VectorXd x0 = utils::flatten(def_grad);
			// 	fd::finite_jacobian(
			// 		x0,
			// 		[this](const Eigen::VectorXd &x) -> Eigen::VectorXd {
			// 			Eigen::MatrixXd F = utils::unflatten(x, this->size());
			// 			double val;
			// 			Eigen::MatrixXd stress;
			// 			this->homogenization(F, val, stress);
			// 			return utils::flatten(stress);
			// 		},
			// 		fhess,
			// 		fd::AccuracyOrder::SECOND,
			// 		1e-6);

			// 	if (!compare_matrix(hessian_temp, fhess))
			// 	{
			// 		std::cout << "Hessian: " << hessian_temp << std::endl;
			// 		std::cout << "Finite hessian: " << fhess << std::endl;
			// 		logger().error("Hessian mismatch!");
			// 		// hessian_temp = fhess;
			// 	}
			// }

			{
				Eigen::MatrixXd hessian_temp2 = hessian_temp;
				for (int i = 0; i < size(); i++)
				for (int j = 0; j < size(); j++)
				for (int k = 0; k < size(); k++)
				for (int l = 0; l < size(); l++)
					hessian_temp(i + j * size(), k + l * size()) = hessian_temp2(i * size() + j, k * size() + l);
			}

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

	void MultiscaleRB::test_reduced_basis(const std::vector<Eigen::MatrixXd> &def_grads, Eigen::VectorXd &energy_errors, Eigen::VectorXd &stress_errors)
	{
		energy_errors.setZero(def_grads.size());
		stress_errors.setZero(def_grads.size());
		// Eigen::MatrixXd fluctuated;
		// utils::maybe_parallel_for(def_grads.size(), [&](int start, int end, int thread_id) {
		// 	for (int i = start; i < end; i++)
			for (int i = 0; i < def_grads.size(); i++)
			{
				const auto &F = def_grads[i];

				double val, val_ref;
				Eigen::MatrixXd stress, stress_ref;

				homogenization(F, val, stress);
				// brute_force_homogenization(F, val_ref, stress_ref, fluctuated);
				Multiscale::homogenization(F, val_ref, stress_ref);

				if (val_ref != 0)
				{
					energy_errors(i) = std::abs((val - val_ref) / val_ref);
					stress_errors(i) = (stress - stress_ref).norm() / stress_ref.norm();
				}
			}
	// });
	}

	MultiscaleRBProblem::MultiscaleRBProblem(const std::shared_ptr<State> &state_ptr, const Eigen::MatrixXd &reduced_basis): state(state_ptr), reduced_basis_(reduced_basis) 
	{
		RowVectorNd min, max;
		state->mesh->bounding_box(min, max);
		microstructure_volume = (max - min).prod();
	}

	double MultiscaleRBProblem::value(const TVector &x, const bool only_elastic)
	{
		Eigen::MatrixXd sol = coeff_to_field(x);
		return state->assembler.assemble_energy(
			state->formulation(), state->mesh->is_volume(), state->bases, state->geom_bases(),
			state->ass_vals_cache, 0, sol, sol) / microstructure_volume;
	}

	void MultiscaleRBProblem::gradient(const TVector &x, TVector &gradv, const bool only_elastic)
	{
		Eigen::MatrixXd sol = coeff_to_field(x);
		Eigen::MatrixXd grad;
		state->assembler.assemble_energy_gradient(
			state->formulation(), state->mesh->is_volume(), state->n_bases, state->bases, state->geom_bases(),
			state->ass_vals_cache, 0, sol, sol, reduced_basis_, grad);
		gradv = grad / microstructure_volume;
	}

	void MultiscaleRBProblem::hessian(const TVector &x, THessian &hessian)
	{
		Eigen::MatrixXd sol = coeff_to_field(x);
		Eigen::MatrixXd tmp;
		state->assembler.assemble_energy_hessian(
			state->formulation(), state->mesh->is_volume(), state->n_bases, false, state->bases,
			state->geom_bases(), state->ass_vals_cache, 0, sol, sol, reduced_basis_, tmp);
		hessian = tmp.sparseView();
		hessian /= microstructure_volume;
	}

	void MultiscaleRBProblem::hessian(const TVector &x, Eigen::MatrixXd &hessian)
	{
		Eigen::MatrixXd sol = coeff_to_field(x);
		state->assembler.assemble_energy_hessian(
			state->formulation(), state->mesh->is_volume(), state->n_bases, false, state->bases,
			state->geom_bases(), state->ass_vals_cache, 0, sol, sol, reduced_basis_, hessian);
		hessian /= microstructure_volume;
	}
} // namespace polyfem::assembler
