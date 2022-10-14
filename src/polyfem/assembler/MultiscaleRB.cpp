#include "MultiscaleRB.hpp"

#include <polyfem/basis/Basis.hpp>
#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <polyfem/utils/MatrixUtils.hpp>
#include <igl/Timer.h>
#include <polyfem/State.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/Evaluator.hpp>

#include <cppoptlib/problem.h>
#include <polyfem/solver/NonlinearSolver.hpp>
#include <polyfem/solver/SparseNewtonDescentSolver.hpp>

#include <unsupported/Eigen/KroneckerProduct>

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
			if (std::abs(R.determinant() - 1) > 1e-3)
				logger().error("Polar decomposition failed!");
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
			for (int j = 0; j < dim; j++)
			for (int p = 0; p < dim; p++)
			for (int q = 0; q < dim; q++)
			{
				dATA_dA(i + j * dim, p + q * dim) += delta(i, q) * F(p, j) + delta(j, q) * F(p, i);
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
			
			std::cout << "error: " << error << " scale: " << scale << "\n";

			if (error > scale * test_eps)
				same = false;

			return same;
		}
	
		class MultiscaleRBProblem : public cppoptlib::Problem<double>
		{
			public:
				using typename cppoptlib::Problem<double>::Scalar;
				using typename cppoptlib::Problem<double>::TVector;
				typedef StiffnessMatrix THessian;

				MultiscaleRBProblem(const Eigen::MatrixXd &reduced_basis): reduced_basis_(reduced_basis) 
				{
					RowVectorNd min, max;
					state->mesh->bounding_box(min, max);
					volume = (max - min).prod();
				}
				~MultiscaleRBProblem() = default;

				void set_linear_disp(const Eigen::MatrixXd &linear_sol) { linear_sol_ = linear_sol; }

				double value(const TVector &x) { return value(x, false); }
				double value(const TVector &x, const bool only_elastic)
				{
					Eigen::MatrixXd sol = coeff_to_field(x);
					return state->assembler.assemble_energy(
						state->formulation(), state->mesh->is_volume(), state->bases, state->geom_bases(),
						state->ass_vals_cache, 0, sol, sol) / volume;
				}
				double target_value(const TVector &x) { return value(x); }
				void gradient(const TVector &x, TVector &gradv) { gradient(x, gradv, false); }
				void gradient(const TVector &x, TVector &gradv, const bool only_elastic)
				{
					Eigen::MatrixXd sol = coeff_to_field(x);
					Eigen::MatrixXd grad;
					state->assembler.assemble_energy_gradient(
						state->formulation(), state->mesh->is_volume(), state->n_bases, state->bases, state->geom_bases(),
						state->ass_vals_cache, 0, sol, sol, grad);
					gradv = (reduced_basis_.transpose() * grad) / volume;
				}
				void target_gradient(const TVector &x, TVector &gradv) { gradient(x, gradv); }
				void hessian(const TVector &x, THessian &hessian)
				{
					Eigen::MatrixXd sol = coeff_to_field(x);
					StiffnessMatrix hessian_;
					state->assembler.assemble_energy_hessian(
						state->formulation(), state->mesh->is_volume(), state->n_bases, false, state->bases,
						state->geom_bases(), state->ass_vals_cache, 0, sol, sol, mat_cache_, hessian_);
					hessian = (reduced_basis_.transpose() * hessian_ * reduced_basis_).sparseView();
					hessian /= volume;
				}

				Eigen::MatrixXd coeff_to_field(const TVector &x)
				{
					return linear_sol_ + reduced_basis_ * x;
				}

				bool is_step_valid(const TVector &x0, const TVector &x1)
				{
					TVector gradv;
					gradient(x1, gradv);
					if (std::isnan(gradv.norm()))
						return false;
					return true;
				}
				
				void set_project_to_psd(bool val) {}
				void save_to_file(const TVector &x0) {}
				void solution_changed(const TVector &newX) {}
				void line_search_begin(const TVector &x0, const TVector &x1) {}
				void line_search_end(bool failed) {}
				void post_step(const int iter_num, const TVector &x) {}
				void smoothing(const TVector &x, TVector &new_x) {}
				bool is_intersection_free(const TVector &x) { return true; }
				bool stop(const TVector &x) { return false; }
				bool remesh(TVector &x) { return false; }
				TVector force_inequality_constraint(const TVector &x0, const TVector &dx) { return x0 + dx; }
				double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
				bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }
				int n_inequality_constraints() { return 0; }
				double inequality_constraint_val(const TVector &x, const int index)
				{
					assert(false);
					return std::nan("");
				}
				TVector inequality_constraint_grad(const TVector &x, const int index)
				{
					assert(false);
					return TVector();
				}

			private:

				Eigen::MatrixXd linear_sol_;
				const Eigen::MatrixXd &reduced_basis_;
				utils::SpareMatrixCache mat_cache_;
				double volume;
		};
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
			Eigen::MatrixXd tmp;
			for (int idx = start; idx < end; idx++)
			// for (int idx = 0; idx < def_grads.size(); idx++)
			{
				// solve fluctuation field
				Eigen::MatrixXd grad = def_grads[idx] - Eigen::MatrixXd::Identity(size(), size());
				state->solve_homogenized_field(grad, tmp);
				sols.col(idx) = tmp;
			}
		});

		// for (int i = 0; i < sols.cols(); i++)
		// {
		// 	state->sol = sols.col(i);
		// 	state->out_geom.export_data(
		// 		*state,
		// 		!state->args["time"].is_null(),
		// 		0, 0,
		// 		io::OutGeometryData::ExportOptions(state->args, state->mesh->is_linear(), state->problem->is_scalar(), state->solve_export_to_file),
		// 		"step_" + std::to_string(i) + ".vtu",
		// 		"", // nodes_path,
		// 		"", // solution_path,
		// 		"", // stress_path,
		// 		"", // mises_path,
		// 		state->is_contact_enabled(), state->solution_frames);
		// }

		logger().info("Compute covarianece matrix...");

		// compute covariance matrix
		StiffnessMatrix laplacian;
		state->assembler.assemble_problem("Laplacian", state->mesh->is_volume(), state->n_bases, state->bases, state->geom_bases(), state->ass_vals_cache, laplacian);

		RowVectorNd min, max;
		state->mesh->bounding_box(min, max);
		const double volume = (max - min).prod();

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

		covariance /= volume;

		// Schur Decomposition
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covariance);
		Eigen::MatrixXd ev = es.eigenvectors();

		// Setup reduced basis
		const int Ns = n_reduced_basis;
		reduced_basis = sols * ev.rightCols(Ns);
		for (int j = 0; j < Ns; j++)
			reduced_basis.col(j) /= sqrt(es.eigenvalues()(ev.cols() - Ns + j));

		if (std::isnan(reduced_basis.norm()))
		{
			logger().error("Covariance eigenvalues: {}", es.eigenvalues().transpose());
			log_and_throw_error("NAN in reduced basis!");
		}

		for (int i = 0; i < es.eigenvalues().size() - 1; i++)
			if (es.eigenvalues()(i) > es.eigenvalues()(i+1))
			{
				std::cout << es.eigenvalues().transpose() << "\n";
				log_and_throw_error("Eigenvalues not in increase order in Schur decomposition!");
			}

		logger().info("Reduced basis created!");
	}

	void MultiscaleRB::projection(const Eigen::MatrixXd &F, Eigen::MatrixXd &x) const
	{
		Eigen::VectorXd xi;
		xi.setZero(reduced_basis.cols());

		std::shared_ptr<MultiscaleRBProblem> nl_problem = std::make_shared<MultiscaleRBProblem>(reduced_basis);
		nl_problem->set_linear_disp(state->generate_linear_field(F));
		std::shared_ptr<cppoptlib::NonlinearSolver<MultiscaleRBProblem>> nlsolver = std::make_shared<cppoptlib::SparseNewtonDescentSolver<MultiscaleRBProblem>>(
				state->args["solver"]["nonlinear"], state->args["solver"]["linear"]);
		
		nlsolver->minimize(*nl_problem, xi);
		x = nl_problem->coeff_to_field(xi);

		// {
		// 	static int idx = 0;
		// 	state->sol = x;
		// 	state->out_geom.export_data(
		// 		*state,
		// 		!state->args["time"].is_null(),
		// 		0, 0,
		// 		io::OutGeometryData::ExportOptions(state->args, state->mesh->is_linear(), state->problem->is_scalar(), state->solve_export_to_file),
		// 		"proj_" + std::to_string(idx) + ".vtu",
		// 		"", // nodes_path,
		// 		"", // solution_path,
		// 		"", // stress_path,
		// 		"", // mises_path,
		// 		state->is_contact_enabled(), state->solution_frames);
		// 	idx++;
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

	double MultiscaleRB::homogenize_energy(const Eigen::MatrixXd &x) const
	{
		RowVectorNd min, max;
		state->mesh->bounding_box(min, max);
		const double volume = (max - min).prod();

		const auto &bases = state->bases;
		const auto &gbases = state->geom_bases();

		return state->assembler.assemble_energy(state->formulation(), size() == 3, bases, gbases, state->ass_vals_cache, 0, x, x) / volume;
	}

	void MultiscaleRB::homogenize_stress(const Eigen::MatrixXd &x, Eigen::MatrixXd &stress) const
	{
		RowVectorNd min, max;
		state->mesh->bounding_box(min, max);
		const double volume = (max - min).prod();

		const auto &bases = state->bases;
		const auto &gbases = state->geom_bases();

		stress.setZero(size(), size());
		Eigen::MatrixXd stresses, avg_stress, tmp;

		for (int e = 0; e < bases.size(); ++e)
		{
			assembler::ElementAssemblyValues vals;
			state->ass_vals_cache.compute(e, size() == 3, bases[e], gbases[e], vals);

			const quadrature::Quadrature &quadrature = vals.quadrature;
			Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();

			state->assembler.compute_tensor_value(state->formulation(), e, bases[e], gbases[e], quadrature.points, x, stresses);
			tmp = stresses.transpose() * da;
			avg_stress = Eigen::Map<Eigen::MatrixXd>(tmp.data(), size(), size());
			stress += avg_stress;
		}

		stress /= volume;
	}

	void MultiscaleRB::homogenize_stiffness(const Eigen::MatrixXd &x, Eigen::MatrixXd &stiffness) const
	{
		RowVectorNd min, max;
		state->mesh->bounding_box(min, max);
		const double volume = (max - min).prod();

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

			state->assembler.compute_stiffness_value(state->formulation(), e, bases[e], gbases[e], quadrature.points, x, stiffnesses);
			// tmp = stiffnesses.transpose() * da;
			// for (int i = 0, idx = 0; i < size(); i++)
			// for (int j = 0; j < size(); j++)
			// for (int k = 0; k < size(); k++)
			// for (int l = 0; l < size(); l++)
			// {
			// 	avg_stiffness(i * size() + j, k * size() + l) += tmp(idx);
			// 	idx++;
			// }
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
		avg_stiffness /= volume;

		// compute term2 given CB
		{
			Eigen::MatrixXd Dinv;
			{
				StiffnessMatrix hessian;
				utils::SpareMatrixCache mat_cache_;
				state->assembler.assemble_energy_hessian(
					state->formulation(), size() == 3, state->n_bases, false, state->bases,
					state->geom_bases(), state->ass_vals_cache, 0, x, x, mat_cache_, hessian);
				Dinv = (reduced_basis.transpose() * hessian * reduced_basis).inverse();
			}
			term2 = CB * Dinv * CB.transpose() / volume;
		}

		stiffness = avg_stiffness - term2;
	}

	void MultiscaleRB::homogenization(const Eigen::MatrixXd &def_grad, double &energy, Eigen::MatrixXd &stress, Eigen::MatrixXd &stiffness) const
	{
		// polar decomposition
		// Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner> svd; // def_grad == svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
		// svd.compute(def_grad, Eigen::ComputeThinU | Eigen::ComputeThinV);
		// const Eigen::MatrixXd R = true ? (Eigen::MatrixXd)(svd.matrixU() * svd.matrixV().transpose()) : Eigen::MatrixXd::Identity(size(), size());
		// const Eigen::MatrixXd Ubar = R.transpose() * def_grad; // svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
		Eigen::MatrixXd R, Ubar;
		polar_decomposition(def_grad, R, Ubar);

		Eigen::MatrixXd dUdF;
		my_polar_decomposition_grad(def_grad, R, Ubar, dUdF);

		Eigen::MatrixXd x;
		{
			Eigen::MatrixXd disp_grad = Ubar - Eigen::MatrixXd::Identity(size(), size());
			projection(disp_grad, x);
		}

		// effective energy = average energy over unit cell
		energy = homogenize_energy(x);

		// effective stress = average stress over unit cell
		Eigen::MatrixXd stress_no_rotation;
		homogenize_stress(x, stress_no_rotation);
		// stress = R * stress_no_rotation; 
		stress = utils::unflatten(dUdF.transpose() * utils::flatten(stress_no_rotation), size());

		// \bar{C}^{RB} = < C^{RB} > - \sum_{i,j} (D^{-1})_{ij} < C^{RB} \cdot B^{(i)} > \cross < B^{(j)} \cdot C^{RB} >
		// D_{ij} = < B^{(i)} \cdot C \cdot B^{(j)} >
		Eigen::MatrixXd stiffness_no_rotation;
		homogenize_stiffness(x, stiffness_no_rotation);
		stiffness.setZero(size()*size(), size()*size());
		for (int i = 0; i < size(); i++) for (int j = 0; j < size(); j++)
		for (int k = 0; k < size(); k++) for (int l = 0; l < size(); l++)
		for (int m = 0; m < size(); m++) for (int n = 0; n < size(); n++)
		{
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
			n_reduced_basis = params["n_reduced_basis"];

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
			// 	Eigen::VectorXd x0 = utils::flatten(def_grad);
			// 	fd::finite_gradient(
			// 		x0, [this](const Eigen::VectorXd &x) -> double 
			// 		{ 
			// 			Eigen::MatrixXd F = utils::unflatten(x, this->size());
			// 			double val;
			// 			Eigen::MatrixXd stress, stiffness;
			// 			this->homogenization(F, val, stress, stiffness);
			// 			return val;
			// 		}, fgrad, fd::AccuracyOrder::SECOND, 1e-6);

			// 	grad = utils::flatten(stress_tensor);
			// 	if (!compare_matrix(grad, fgrad))
			// 	{
			// 		std::cout << "Gradient: " << grad.transpose() << std::endl;
			// 		std::cout << "Finite gradient: " << fgrad.transpose() << std::endl;
			// 		log_and_throw_error("Gradient mismatch");
			// 	}
			// 	else
			// 	{
			// 		logger().info("Gradient match!");
			// 	}
			// }

			G += delF_delU * stress_tensor.transpose() * data.da(p);
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
			// {
			// 	Eigen::MatrixXd fhess;
			// 	Eigen::VectorXd x0 = utils::flatten(def_grad);
			// 	fd::finite_jacobian(
			// 		x0,
			// 		[this](const Eigen::VectorXd &x) -> Eigen::VectorXd {
			// 			Eigen::MatrixXd F = utils::unflatten(x, this->size());
			// 			double val;
			// 			Eigen::MatrixXd stress, stiffness;
			// 			this->homogenization(F, val, stress, stiffness);
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
			// 		hessian_temp = fhess;
			// 	}
			// 	else
			// 	{
			// 		logger().info("Hessian match!");
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
