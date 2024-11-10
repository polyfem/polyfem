#include "ElasticForm.hpp"

#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

using namespace polyfem::assembler;
using namespace polyfem::utils;
using namespace polyfem::quadrature;

namespace polyfem::solver
{
	namespace
	{
		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			assembler::ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};

		Eigen::MatrixXd refined_nodes(const int dim, const int i)
		{
			Eigen::MatrixXd A(dim + 1, dim);
			if (dim == 2)
			{
				A << 0., 0., 
					1., 0., 
					0., 1.;
				switch (i)
				{
				case 0:
					break;
				case 1:
					A.col(0).array() += 1;
					break;
				case 2:
					A.col(1).array() += 1;
					break;
				case 3:
					A.array() -= 1;
					A *= -1;
					break;
				default:
					throw std::runtime_error("Invalid node index");
				}
			}
			else
			{
				A << 0, 0, 0, 
					1, 0, 0, 
					0, 1, 0, 
					0, 0, 1;
				switch (i)
				{
				case 0:
					break;
				case 1:
					A.col(0).array() += 1;
					break;
				case 2:
					A.col(1).array() += 1;
					break;
				case 3:
					A.col(2).array() += 1;
					break;
				case 4:
				{
					Eigen::VectorXd tmp = 1 - A.col(1).array() - A.col(2).array();
					A.col(2) += A.col(0) + A.col(1);
					A.col(0) = tmp;
					break;
				}
				case 5:
				{
					Eigen::VectorXd tmp = 1. - A.col(1).array();
					A.col(2) += A.col(1);
					A.col(1) += A.col(0);
					A.col(0) = tmp;
					break;
				}
				case 6:
				{
					Eigen::VectorXd tmp = A.col(0) + A.col(1);
					A.col(1) = 1. - A.col(0).array();
					A.col(0) = tmp;
					break;
				}
				case 7:
				{
					Eigen::VectorXd tmp = 1. - A.col(0).array() - A.col(1).array();
					A.col(1) += A.col(2);
					A.col(2) = tmp;
					break;
				}
				default:
					throw std::runtime_error("Invalid node index");
				}
			}
			return A / 2;
		}

		/// @brief given the position of the vertices of a triangle, extract the subtriangle vertices based on the tree
		/// @param pts vertices of the triangle
		/// @param tree refinement hiararchy
		/// @return vertices of the subtriangles, and the refinement level of each subtriangle
		std::tuple<Eigen::MatrixXd, std::vector<int>> extract_subelement(const Eigen::MatrixXd &pts, const Tree &tree)
		{
			if (!tree.has_children())
				return {pts, std::vector<int>{0}};

			const int dim = pts.cols();
			Eigen::MatrixXd out;
			std::vector<int> levels;
			for (int i = 0; i < tree.n_children(); i++)
			{
				Eigen::MatrixXd uv;
				uv.setZero(dim+1, dim+1);
				uv.rightCols(dim) = refined_nodes(dim, i);
				if (dim == 2)
					uv.col(0) = 1. - uv.col(2).array() - uv.col(1).array();
				else
					uv.col(0) = 1. - uv.col(3).array() - uv.col(1).array() - uv.col(2).array();
				
				Eigen::MatrixXd pts_ = uv * pts;

				auto [tmp, L] = extract_subelement(pts_, tree.child(i));
				if (out.size() == 0)
					out = tmp;
				else
				{
					out.conservativeResize(out.rows() + tmp.rows(), Eigen::NoChange);
					out.bottomRows(tmp.rows()) = tmp;
				}
				for (int &i : L)
					++i;
				levels.insert(levels.end(), L.begin(), L.end());
			}
			return {out, levels};
		}
	
		quadrature::Quadrature refine_quadrature(const Tree &tree, const int dim, const int order)
		{
			Eigen::MatrixXd pts(dim + 1, dim);
			if (dim == 2)
				pts << 0., 0.,
					1., 0.,
					0., 1.;
			else
				pts << 0, 0, 0,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1;
			auto [quad_points, levels] = extract_subelement(pts, tree);

			Quadrature tmp, quad;
			if (dim == 2)
			{
				TriQuadrature tri_quadrature(true);
				tri_quadrature.get_quadrature(order, tmp);
				tmp.points.conservativeResize(tmp.points.rows(), dim + 1);
				tmp.points.col(dim) = 1. - tmp.points.col(0).array() - tmp.points.col(1).array();
			}
			else
			{
				TetQuadrature tet_quadrature(true);
				tet_quadrature.get_quadrature(order, tmp);
				tmp.points.conservativeResize(tmp.points.rows(), dim + 1);
				tmp.points.col(dim) = 1. - tmp.points.col(0).array() - tmp.points.col(1).array() - tmp.points.col(2).array();
			}

			quad.points.resize(tmp.size() * levels.size(), dim);
			quad.weights.resize(tmp.size() * levels.size());

			for (int i = 0; i < levels.size(); i++)
			{
				quad.points.middleRows(i * tmp.size(), tmp.size()) = tmp.points * quad_points.middleRows(i * (dim + 1), dim + 1);
				quad.weights.segment(i * tmp.size(), tmp.size()) = tmp.weights / pow(2, dim * levels[i]);
			}
			assert (fabs(quad.weights.sum() - tmp.weights.sum()) < 1e-8);

			return quad;
		}

		// Eigen::MatrixXd evaluate_jacobian(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &uv, const Eigen::VectorXd &disp)
		// {
		// 	assembler::ElementAssemblyValues vals;
		// 	vals.compute(0, uv.cols() == 3, uv, bs, gbs);

		// 	Eigen::MatrixXd out(uv.rows(), 2);
		// 	for (long p = 0; p < uv.rows(); ++p)
		// 	{
		// 		Eigen::MatrixXd disp_grad;
		// 		disp_grad.setZero(uv.cols(), uv.cols());

		// 		for (std::size_t j = 0; j < vals.basis_values.size(); ++j)
		// 		{
		// 			const auto &loc_val = vals.basis_values[j];

		// 			for (int d = 0; d < uv.cols(); ++d)
		// 			{
		// 				for (std::size_t ii = 0; ii < loc_val.global.size(); ++ii)
		// 				{
		// 					disp_grad.row(d) += loc_val.global[ii].val * loc_val.grad.row(p) * disp(loc_val.global[ii].index * uv.cols() + d);
		// 				}
		// 			}
		// 		}

		// 		disp_grad = disp_grad * vals.jac_it[p] + Eigen::MatrixXd::Identity(uv.cols(), uv.cols());
		// 		out.row(p) << disp_grad.determinant(), disp_grad.determinant() / vals.jac_it[p].determinant();
		// 	}
		// 	return out;
		// }

		void update_quadrature(const int invalidID, const int dim, Tree &tree, const int quad_order, basis::ElementBases &bs, const basis::ElementBases &gbs, assembler::AssemblyValsCache &ass_vals_cache)
		{
			// update quadrature to capture the point with negative jacobian
			const Quadrature quad = refine_quadrature(tree, dim, quad_order);

			// capture the flipped point by refining the quadrature
			bs.set_quadrature([quad](Quadrature &quad_) {
				quad_ = quad;
			});
			logger().debug("New number of quadrature points: {}, level: {}", quad.size(), tree.depth());

			if (ass_vals_cache.is_initialized())
				ass_vals_cache.update(invalidID, dim == 3, bs, gbs);
		}
	} // namespace

	ElasticForm::ElasticForm(const int n_bases,
							 std::vector<basis::ElementBases> &bases,
							 const std::vector<basis::ElementBases> &geom_bases,
							 const assembler::Assembler &assembler,
							 assembler::AssemblyValsCache &ass_vals_cache,
							 const double t, const double dt,
							 const bool is_volume,
							 const double jacobian_threshold,
							 const ElementInversionCheck check_inversion)
		: n_bases_(n_bases),
		  bases_(bases),
		  geom_bases_(geom_bases),
		  assembler_(assembler),
		  ass_vals_cache_(ass_vals_cache),
		  t_(t),
		  jacobian_threshold_(jacobian_threshold),
		  check_inversion_(check_inversion),
		  dt_(dt),
		  is_volume_(is_volume)
	{
		if (assembler_.is_linear())
			compute_cached_stiffness();
		// mat_cache_ = std::make_unique<utils::DenseMatrixCache>();
		mat_cache_ = std::make_unique<utils::SparseMatrixCache>();
		quadrature_hierarchy_.resize(bases_.size());

		quadrature_order_ = AssemblerUtils::quadrature_order(assembler_.name(), bases_[0].bases[0].order(), AssemblerUtils::BasisType::SIMPLEX_LAGRANGE, is_volume_ ? 3 : 2);
	
		if (check_inversion_ != ElementInversionCheck::Discrete)
		{
			Eigen::VectorXd x0;
			x0.setZero(n_bases_ * (is_volume_ ? 3 : 2));
			if (!is_step_collision_free(x0, x0))
				log_and_throw_error("Initial state has inverted elements!");
				
			int basis_order = 0;
			int gbasis_order = 0;
			for (int e = 0; e < bases_.size(); e++)
			{
				if (basis_order == 0)
					basis_order = bases_[e].bases.front().order();
				else if (basis_order != bases_[e].bases.front().order())
					log_and_throw_error("Non-uniform basis order not supported for conservative Jacobian check!!");
				if (gbasis_order == 0)
					gbasis_order = geom_bases_[e].bases.front().order();
				else if (gbasis_order != geom_bases_[e].bases.front().order())
					log_and_throw_error("Non-uniform gbasis order not supported for conservative Jacobian check!!");
			}
		}
	}

	double ElasticForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return assembler_.assemble_energy(
			is_volume_,
			bases_, geom_bases_, ass_vals_cache_, t_, dt_, x, x_prev_);
	}

	Eigen::VectorXd ElasticForm::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd out = assembler_.assemble_energy_per_element(
			is_volume_, bases_, geom_bases_, ass_vals_cache_, t_, dt_, x, x_prev_);
		assert(abs(out.sum() - value_unweighted(x)) < std::max(1e-10 * out.sum(), 1e-10));
		return out;
	}

	void ElasticForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_gradient(is_volume_, n_bases_, bases_, geom_bases_,
									 ass_vals_cache_, t_, dt_, x, x_prev_, grad);
		gradv = grad;
	}

	void ElasticForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("elastic hessian");

		hessian.resize(x.size(), x.size());

		if (assembler_.is_linear())
		{
			assert(cached_stiffness_.rows() == x.size() && cached_stiffness_.cols() == x.size());
			hessian = cached_stiffness_;
		}
		else
		{
			// NOTE: mat_cache_ is marked as mutable so we can modify it here
			assembler_.assemble_hessian(
				is_volume_, n_bases_, project_to_psd_, bases_,
				geom_bases_, ass_vals_cache_, t_, dt_, x, x_prev_, *mat_cache_, hessian);
		}
	}

	void ElasticForm::finish()
	{
		for (auto &t : quadrature_hierarchy_)
			t = Tree();
	}

	double ElasticForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		if (check_inversion_ == ElementInversionCheck::Discrete)
			return 1.;

		const int dim = is_volume_ ? 3 : 2;
		double step, invalidStep;
		int invalidID;

		Tree subdivision_tree;
		{
			double transient_check_time = 0;
			{
				POLYFEM_SCOPED_TIMER("Transient Jacobian Check", transient_check_time);
				std::tie(step, invalidID, invalidStep, subdivision_tree) = max_time_step(dim, bases_, geom_bases_, x0, x1);
			}

			logger().log(step == 0 ? spdlog::level::warn : (step == 1. ? spdlog::level::trace : spdlog::level::debug), 
				"Jacobian max step size: {} at element {}, invalid step size: {}, tree depth {}, runtime {} sec", step, invalidID, invalidStep, subdivision_tree.depth(), transient_check_time);
		}

		if (invalidID >= 0 && step <= 0.5)
		{
			auto& bs = bases_[invalidID];
			auto& gbs = geom_bases_[invalidID];
			if (quadrature_hierarchy_[invalidID].merge(subdivision_tree)) // if the tree is refined
				update_quadrature(invalidID, dim, quadrature_hierarchy_[invalidID], quadrature_order_, bs, gbs, ass_vals_cache_);

			// verify that new quadrature points don't make x0 invalid
			// {
			// 	Quadrature quad;
			// 	bs.compute_quadrature(quad);
			// 	const Eigen::MatrixXd jacs0 = evaluate_jacobian(bs, gbs, quad.points, x0);
			// 	const Eigen::MatrixXd jacs1 = evaluate_jacobian(bs, gbs, quad.points, x0 + (x1 - x0) * step);
			// 	const Eigen::VectorXd min_jac0 = jacs0.colwise().minCoeff();
			// 	const Eigen::VectorXd min_jac1 = jacs1.colwise().minCoeff();
			// 	logger().debug("Min jacobian on quadrature points: before step {}, {}; after step {}, {}", min_jac0(0), min_jac0(1), min_jac1(0), min_jac1(1));
			// }
		}

		return step;
	}

	bool ElasticForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{		
		if (check_inversion_ == ElementInversionCheck::Discrete)
			return true;

		const auto [isvalid, id, tree] = is_valid(is_volume_ ? 3 : 2, bases_, geom_bases_, x1);
		return isvalid;
	}

	bool ElasticForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// check inversion on quadrature points
		Eigen::VectorXd grad;
		first_derivative(x1, grad);
		if (grad.array().isNaN().any())
			return false;

		return true;

		// Check the scalar field in the output does not contain NANs.
		// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
		//          This causes small step lengths in the LS.
		// TVector x1_full;
		// reduced_to_full(x1, x1_full);
		// return state_.check_scalar_value(x1_full, true, false);
		// return true;
	}

	void ElasticForm::solution_changed(const Eigen::VectorXd &new_x)
	{
	}

	void ElasticForm::compute_cached_stiffness()
	{
		if (assembler_.is_linear() && cached_stiffness_.size() == 0)
		{
			assembler_.assemble(is_volume_, n_bases_, bases_, geom_bases_,
								ass_vals_cache_, t_, cached_stiffness_);
		}
	}

	void ElasticForm::force_material_derivative(const double t, const Eigen::MatrixXd &x, const Eigen::MatrixXd &x_prev, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		const int dim = is_volume_ ? 3 : 2;

		const int n_elements = int(bases_.size());

		if (assembler_.name() == "ViscousDamping")
		{
			term.setZero(2);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x_prev, prev_u, prev_grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_p_i, grad_u_i, prev_grad_u_i;
						vector2matrix(grad_p.row(q), grad_p_i);
						vector2matrix(grad_u.row(q), grad_u_i);
						vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

						Eigen::MatrixXd f_prime_dpsi, f_prime_dphi;
						assembler::ViscousDamping::compute_dstress_dpsi_dphi(OptAssemblerData(t, dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i), prev_grad_u_i, f_prime_dpsi, f_prime_dphi);

						// This needs to be a sum over material parameter basis.
						local_storage.vec(0) += -matrix_inner_product<double>(f_prime_dpsi, grad_p_i) * local_storage.da(q);
						local_storage.vec(1) += -matrix_inner_product<double>(f_prime_dphi, grad_p_i) * local_storage.da(q);
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
		else
		{
			term.setZero(n_elements * 2, 1);

			auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_p_i, grad_u_i;
						vector2matrix(grad_p.row(q), grad_p_i);
						vector2matrix(grad_u.row(q), grad_u_i);

						Eigen::MatrixXd f_prime_dmu, f_prime_dlambda;
						assembler_.compute_dstress_dmu_dlambda(OptAssemblerData(t, dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i), f_prime_dmu, f_prime_dlambda);

						// This needs to be a sum over material parameter basis.
						local_storage.vec(e + n_elements) += -matrix_inner_product<double>(f_prime_dmu, grad_p_i) * local_storage.da(q);
						local_storage.vec(e) += -matrix_inner_product<double>(f_prime_dlambda, grad_p_i) * local_storage.da(q);
					}
				}
			});

			for (const LocalThreadVecStorage &local_storage : storage)
				term += local_storage.vec;
		}
	}

	void ElasticForm::force_shape_derivative(const double t, const int n_verts, const Eigen::MatrixXd &x, const Eigen::MatrixXd &x_prev, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		const int dim = is_volume_ ? 3 : 2;
		const int actual_dim = (assembler_.name() == "Laplacian") ? 1 : dim;

		const int n_elements = int(bases_.size());
		term.setZero(n_verts * dim, 1);

		auto storage = utils::create_thread_storage(LocalThreadVecStorage(term.size()));

		if (assembler_.name() == "ViscousDamping")
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, is_volume_, vals.quadrature.points, geom_bases_[e], geom_bases_[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, prev_u, prev_grad_u, p, grad_p;
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, x_prev, prev_u, prev_grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, dim, vals, adjoint, p, grad_p);

					Eigen::MatrixXd grad_u_i, grad_p_i, prev_grad_u_i;
					Eigen::MatrixXd grad_v_i;
					Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
					Eigen::MatrixXd f_prev_prime_prev_gradu_gradv;

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						vector2matrix(grad_u.row(q), grad_u_i);
						vector2matrix(grad_p.row(q), grad_p_i);
						vector2matrix(prev_grad_u.row(q), prev_grad_u_i);

						for (auto &v : gvals.basis_values)
						{
							Eigen::MatrixXd stress_grad, stress_prev_grad;
							assembler_.compute_stress_grad(OptAssemblerData(t, dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i), prev_grad_u_i, stress_tensor, stress_grad);
							assembler_.compute_stress_prev_grad(OptAssemblerData(t, dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i), prev_grad_u_i, stress_prev_grad);
							for (int d = 0; d < dim; d++)
							{
								grad_v_i.setZero(dim, dim);
								grad_v_i.row(d) = v.grad_t_m.row(q);

								f_prime_gradu_gradv.setZero(dim, dim);
								Eigen::MatrixXd tmp = grad_u_i * grad_v_i;
								for (int i = 0; i < f_prime_gradu_gradv.rows(); i++)
									for (int j = 0; j < f_prime_gradu_gradv.cols(); j++)
										for (int k = 0; k < tmp.rows(); k++)
											for (int l = 0; l < tmp.cols(); l++)
												f_prime_gradu_gradv(i, j) += stress_grad(i * dim + j, k * dim + l) * tmp(k, l);

								f_prev_prime_prev_gradu_gradv.setZero(dim, dim);
								tmp = prev_grad_u_i * grad_v_i;
								for (int i = 0; i < f_prev_prime_prev_gradu_gradv.rows(); i++)
									for (int j = 0; j < f_prev_prime_prev_gradu_gradv.cols(); j++)
										for (int k = 0; k < tmp.rows(); k++)
											for (int l = 0; l < tmp.cols(); l++)
												f_prev_prime_prev_gradu_gradv(i, j) += stress_prev_grad(i * dim + j, k * dim + l) * tmp(k, l);

								tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(dim, dim);
								local_storage.vec(v.global[0].index * dim + d) -= matrix_inner_product<double>(f_prime_gradu_gradv + f_prev_prime_prev_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) * local_storage.da(q);
							}
						}
					}
				}
			});
		}
		else
		{
			utils::maybe_parallel_for(n_elements, [&](int start, int end, int thread_id) {
				LocalThreadVecStorage &local_storage = utils::get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					assembler::ElementAssemblyValues &vals = local_storage.vals;
					ass_vals_cache_.compute(e, is_volume_, bases_[e], geom_bases_[e], vals);
					assembler::ElementAssemblyValues gvals;
					gvals.compute(e, is_volume_, vals.quadrature.points, geom_bases_[e], geom_bases_[e]);

					const quadrature::Quadrature &quadrature = vals.quadrature;
					local_storage.da = vals.det.array() * quadrature.weights.array();

					Eigen::MatrixXd u, grad_u, p, grad_p; //, stiffnesses;
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, x, u, grad_u);
					io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, adjoint, p, grad_p);
					// assembler_.compute_stiffness_value(formulation_, vals, quadrature.points, x, stiffnesses);

					for (int q = 0; q < local_storage.da.size(); ++q)
					{
						Eigen::MatrixXd grad_u_i, grad_p_i, stiffness_i;
						if (actual_dim == 1)
						{
							grad_u_i = grad_u.row(q);
							grad_p_i = grad_p.row(q);
						}
						else
						{
							vector2matrix(grad_u.row(q), grad_u_i);
							vector2matrix(grad_p.row(q), grad_p_i);
						}

						// stiffness_i = utils::unflatten(stiffnesses.row(q).transpose(), actual_dim * dim);

						for (auto &v : gvals.basis_values)
						{
							for (int d = 0; d < dim; d++)
							{
								Eigen::MatrixXd grad_v_i;
								grad_v_i.setZero(dim, dim);
								grad_v_i.row(d) = v.grad_t_m.row(q);

								Eigen::MatrixXd stress_tensor, f_prime_gradu_gradv;
								assembler_.compute_stress_grad_multiply_mat(OptAssemblerData(t, dt_, e, quadrature.points.row(q), vals.val.row(q), grad_u_i), grad_u_i * grad_v_i, stress_tensor, f_prime_gradu_gradv);
								// f_prime_gradu_gradv = utils::unflatten(stiffness_i * utils::flatten(grad_u_i * grad_v_i), dim);

								Eigen::MatrixXd tmp = grad_v_i - grad_v_i.trace() * Eigen::MatrixXd::Identity(dim, dim);
								local_storage.vec(v.global[0].index * dim + d) -= matrix_inner_product<double>(f_prime_gradu_gradv + stress_tensor * tmp.transpose(), grad_p_i) * local_storage.da(q);
							}
						}
					}
				}
			});
		}

		for (const LocalThreadVecStorage &local_storage : storage)
			term += local_storage.vec;
	}
} // namespace polyfem::solver
