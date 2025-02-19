#include "NLHomoProblem.hpp"
#include <polyfem/State.hpp>
#include "forms/PeriodicContactForm.hpp"
#include "forms/lagrangian/MacroStrainLagrangianForm.hpp"
#include <polyfem/assembler/MacroStrain.hpp>
#include <polyfem/io/Evaluator.hpp>

namespace polyfem::solver
{
	NLHomoProblem::NLHomoProblem(const int full_size,
								 const assembler::MacroStrainValue &macro_strain_constraint,
								 const State &state,
								 const double t,
								 const std::vector<std::shared_ptr<Form>> &forms,
								 const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
								 const bool solve_symmetric_macro_strain)
		: NLProblem(full_size, state.periodic_bc, t, forms, penalty_forms),
		  state_(state),
		  only_symmetric(solve_symmetric_macro_strain),
		  macro_strain_constraint_(macro_strain_constraint)
	{
		init_projection();
	}

	void NLHomoProblem::init_projection()
	{
		const int dim = state_.mesh->dimension();
		if (only_symmetric)
		{
			macro_mid_to_full_.setZero(dim * dim, (dim * (dim + 1)) / 2);
			macro_full_to_mid_.setZero((dim * (dim + 1)) / 2, dim * dim);
			for (int i = 0, idx = 0; i < dim; i++)
			{
				for (int j = i; j < dim; j++)
				{
					macro_full_to_mid_(idx, i * dim + j) = 1;

					macro_mid_to_full_(j * dim + i, idx) = 1;
					macro_mid_to_full_(i * dim + j, idx) = 1;

					idx++;
				}
			}
		}
		else
		{
			macro_mid_to_full_.setIdentity(dim * dim, dim * dim);
			macro_full_to_mid_.setIdentity(dim * dim, dim * dim);
		}
		macro_mid_to_reduced_.setIdentity(macro_full_to_mid_.rows(), macro_full_to_mid_.rows());
	}

	Eigen::VectorXd NLHomoProblem::extended_to_reduced(const Eigen::VectorXd &extended) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		Eigen::VectorXd reduced(dof1 + dof2);
		reduced.head(dof1) = NLProblem::full_to_reduced(extended.head(extended.size() - dim * dim));
		reduced.tail(dof2) = macro_full_to_reduced(extended.tail(dim * dim));

		return reduced;
	}

	Eigen::VectorXd NLHomoProblem::reduced_to_extended(const Eigen::VectorXd &reduced, bool homogeneous) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();
		assert(reduced.size() == dof1 + dof2);

		Eigen::VectorXd fluctuation = NLProblem::reduced_to_full(reduced.head(dof1));
		Eigen::VectorXd disp_grad = macro_reduced_to_full(reduced.tail(dof2), homogeneous);
		Eigen::VectorXd extended(fluctuation.size() + disp_grad.size());
		extended << fluctuation, disp_grad;

		return extended;
	}

	Eigen::VectorXd NLHomoProblem::extended_to_reduced_grad(const Eigen::VectorXd &extended) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		Eigen::VectorXd grad(dof1 + dof2);
		grad.head(dof1) = NLProblem::full_to_reduced_grad(extended.head(extended.size() - dim * dim));
		grad.tail(dof2) = macro_full_to_reduced_grad(extended.tail(dim * dim));

		return grad;
	}

	NLHomoProblem::TVector NLHomoProblem::reduced_to_full_shape_derivative(const Eigen::MatrixXd &disp_grad, const TVector &adjoint_full) const
	{
		const int dim = state_.mesh->dimension();

		Eigen::VectorXd term;
		term.setZero(state_.n_bases * dim);
		for (int i = 0; i < state_.n_bases; i++)
			term.segment(i * dim, dim) += disp_grad.transpose() * adjoint_full.segment(i * dim, dim);

		return state_.basis_nodes_to_gbasis_nodes * term;
	}

	double NLHomoProblem::value(const TVector &x)
	{
		double val = NLProblem::value(x);
		for (auto &form : homo_forms)
			if (form->enabled())
				val += form->value(reduced_to_extended(x));

		return val;
	}
	void NLHomoProblem::gradient(const TVector &x, TVector &gradv)
	{
		NLProblem::gradient(x, gradv);

		for (auto &form : homo_forms)
			if (form->enabled())
			{
				Eigen::VectorXd grad_extended;
				form->first_derivative(reduced_to_extended(x), grad_extended);
				gradv += extended_to_reduced_grad(grad_extended);
			}
	}
	void NLHomoProblem::extended_hessian_to_reduced_hessian(const THessian &extended, THessian &reduced) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		Eigen::MatrixXd A12, A22;
		{
			Eigen::MatrixXd tmp = Eigen::MatrixXd(extended.rightCols(dim * dim));
			A12 = macro_full_to_reduced_grad(tmp.topRows(tmp.rows() - dim * dim).transpose()).transpose();
			A22 = macro_full_to_reduced_grad(macro_full_to_reduced_grad(tmp.bottomRows(dim * dim)).transpose());
		}

		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(extended.nonZeros() + A12.size() * 2 + A22.size());

		for (int k = 0; k < full_size_; ++k)
			for (StiffnessMatrix::InnerIterator it(extended, k); it; ++it)
				if (it.row() < full_size_ && it.col() < full_size_)
					entries.emplace_back(it.row(), it.col(), it.value());

		for (int i = 0; i < A12.rows(); i++)
		{
			for (int j = 0; j < A12.cols(); j++)
			{
				entries.emplace_back(i, full_size_ + j, A12(i, j));
				entries.emplace_back(full_size_ + j, i, A12(i, j));
			}
		}

		for (int i = 0; i < A22.rows(); i++)
			for (int j = 0; j < A22.cols(); j++)
				entries.emplace_back(i + full_size_, j + full_size_, A22(i, j));

		// Eigen::VectorXd tmp = macro_full_to_reduced_grad(Eigen::VectorXd::Ones(dim*dim));
		// for (int i = 0; i < tmp.size(); i++)
		//     entries.emplace_back(i + full_size_, i + full_size_, 1 - tmp(i));

		THessian mid(full_size_ + dof2, full_size_ + dof2);
		mid.setFromTriplets(entries.begin(), entries.end());

		NLProblem::full_hessian_to_reduced_hessian(mid, reduced);
	}
	Eigen::MatrixXd NLHomoProblem::reduced_to_disp_grad(const TVector &reduced, bool homogeneous) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		return utils::unflatten(macro_reduced_to_full(reduced.tail(dof2), homogeneous), dim);
	}
	void NLHomoProblem::hessian(const TVector &x, THessian &hessian)
	{
		NLProblem::hessian(x, hessian);

		for (auto &form : homo_forms)
			if (form->enabled())
			{
				THessian hess_extended;
				form->second_derivative(reduced_to_extended(x), hess_extended);

				THessian hess;
				extended_hessian_to_reduced_hessian(hess_extended, hess);
				hessian += hess;
			}
	}

	void NLHomoProblem::set_fixed_entry(const Eigen::VectorXi &fixed_entry)
	{
		const int dim = state_.mesh->dimension();

		Eigen::VectorXd fixed_mask;
		fixed_mask.setZero(dim * dim);
		fixed_mask(fixed_entry.array()).setOnes();
		fixed_mask = (macro_full_to_mid_ * fixed_mask).eval();

		fixed_mask_.setZero(fixed_mask.size());
		for (int i = 0; i < fixed_mask.size(); i++)
			if (abs(fixed_mask(i)) > 1e-8)
				fixed_mask_(i) = true;

		const int new_reduced_size = fixed_mask_.size() - fixed_mask_.sum();
		macro_mid_to_reduced_.setZero(new_reduced_size, fixed_mask_.size());
		for (int i = 0, j = 0; i < fixed_mask_.size(); i++)
			if (!fixed_mask_(i))
				macro_mid_to_reduced_(j++, i) = 1;
	}

	void NLHomoProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		Eigen::MatrixXd tmp = constraint_grad();
		Eigen::MatrixXd A12 = full * tmp.transpose();
		Eigen::MatrixXd A22 = tmp * A12;

		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(full.nonZeros() + A12.size() * 2 + A22.size());

		for (int k = 0; k < full.outerSize(); ++k)
			for (StiffnessMatrix::InnerIterator it(full, k); it; ++it)
				entries.emplace_back(it.row(), it.col(), it.value());

		for (int i = 0; i < A12.rows(); i++)
			for (int j = 0; j < A12.cols(); j++)
			{
				entries.emplace_back(i, full.cols() + j, A12(i, j));
				entries.emplace_back(full.rows() + j, i, A12(i, j));
			}

		for (int i = 0; i < A22.rows(); i++)
			for (int j = 0; j < A22.cols(); j++)
				entries.emplace_back(i + full.rows(), j + full.cols(), A22(i, j));

		THessian mid(full.rows() + dof2, full.cols() + dof2);
		mid.setFromTriplets(entries.begin(), entries.end());

		NLProblem::full_hessian_to_reduced_hessian(mid, reduced);
	}

	NLHomoProblem::TVector NLHomoProblem::full_to_reduced(const TVector &full) const
	{
		log_and_throw_error("Invalid function!");
		return TVector();
	}

	NLHomoProblem::TVector NLHomoProblem::full_to_reduced(const TVector &full, const Eigen::MatrixXd &disp_grad) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		TVector reduced;
		reduced.setZero(dof1 + dof2);

		reduced.head(dof1) = NLProblem::full_to_reduced(full - io::Evaluator::generate_linear_field(state_.n_bases, state_.mesh_nodes, disp_grad));
		reduced.tail(dof2) = macro_full_to_reduced(utils::flatten(disp_grad));

		return reduced;
	}
	NLHomoProblem::TVector NLHomoProblem::full_to_reduced_grad(const TVector &full) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		TVector reduced;
		reduced.setZero(dof1 + dof2);

		reduced.head(dof1) = NLProblem::full_to_reduced_grad(full);
		reduced.tail(dof2) = constraint_grad() * full;

		return reduced;
	}
	NLHomoProblem::TVector NLHomoProblem::reduced_to_full(const TVector &reduced) const
	{
		const int dim = state_.mesh->dimension();
		const int dof2 = macro_reduced_size();
		const int dof1 = reduced_size();

		Eigen::MatrixXd disp_grad = utils::unflatten(macro_reduced_to_full(reduced.tail(dof2)), dim);
		return NLProblem::reduced_to_full(reduced.head(dof1)) + io::Evaluator::generate_linear_field(state_.n_bases, state_.mesh_nodes, disp_grad);
	}

	int NLHomoProblem::macro_reduced_size() const
	{
		return macro_mid_to_reduced_.rows();
	}
	NLHomoProblem::TVector NLHomoProblem::macro_full_to_reduced(const TVector &full) const
	{
		return macro_mid_to_reduced_ * macro_full_to_mid_ * full;
	}
	Eigen::MatrixXd NLHomoProblem::macro_full_to_reduced_grad(const Eigen::MatrixXd &full) const
	{
		return macro_mid_to_reduced_ * macro_mid_to_full_.transpose() * full;
	}
	NLHomoProblem::TVector NLHomoProblem::macro_reduced_to_full(const TVector &reduced, bool homogeneous) const
	{
		TVector mid = macro_mid_to_reduced_.transpose() * reduced;
		const TVector fixed_values = homogeneous ? TVector::Zero(macro_full_to_mid_.rows()) : TVector(macro_full_to_mid_ * utils::flatten(macro_strain_constraint_.eval(t_)));
		for (int i = 0; i < fixed_mask_.size(); i++)
			if (fixed_mask_(i))
				mid(i) = fixed_values(i);

		return macro_mid_to_full_ * mid;
	}

	void NLHomoProblem::init(const TVector &x0)
	{
		for (auto &form : homo_forms)
			form->init(reduced_to_extended(x0));
		FullNLProblem::init(reduced_to_full(x0));
	}

	bool NLHomoProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		bool flag = NLProblem::is_step_valid(x0, x1);
		for (auto &form : homo_forms)
			if (form->enabled())
				flag &= form->is_step_valid(reduced_to_extended(x0), reduced_to_extended(x1));

		return flag;
	}
	bool NLHomoProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		bool flag = NLProblem::is_step_collision_free(x0, x1);
		for (auto &form : homo_forms)
			if (form->enabled())
				flag &= form->is_step_collision_free(reduced_to_extended(x0), reduced_to_extended(x1));

		return flag;
	}
	double NLHomoProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double size = NLProblem::max_step_size(x0, x1);
		for (auto &form : homo_forms)
			if (form->enabled())
				size = std::min(size, form->max_step_size(reduced_to_extended(x0), reduced_to_extended(x1)));
		return size;
	}

	void NLHomoProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		NLProblem::line_search_begin(x0, x1);
		for (auto &form : homo_forms)
			form->line_search_begin(reduced_to_extended(x0), reduced_to_extended(x1));
	}
	void NLHomoProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		NLProblem::post_step(data);
		for (auto &form : homo_forms)
			form->post_step(polysolve::nonlinear::PostStepData(
				data.iter_num, data.solver_info, reduced_to_extended(data.x), reduced_to_extended(data.grad)));
	}

	void NLHomoProblem::solution_changed(const TVector &new_x)
	{
		NLProblem::solution_changed(new_x);
		for (auto &form : homo_forms)
			form->solution_changed(reduced_to_extended(new_x));
	}

	void NLHomoProblem::init_lagging(const TVector &x)
	{
		NLProblem::init_lagging(x);
		for (auto &form : homo_forms)
			form->init_lagging(reduced_to_extended(x));
	}
	void NLHomoProblem::update_lagging(const TVector &x, const int iter_num)
	{
		NLProblem::update_lagging(x, iter_num);
		for (auto &form : homo_forms)
			form->update_lagging(reduced_to_extended(x), iter_num);
	}

	void NLHomoProblem::update_quantities(const double t, const TVector &x)
	{
		NLProblem::update_quantities(t, x);
		for (auto &form : homo_forms)
			form->update_quantities(t, reduced_to_extended(x));
	}

	Eigen::MatrixXd NLHomoProblem::constraint_grad() const
	{
		const int dim = state_.mesh->dimension();
		Eigen::MatrixXd jac; // (dim*dim) x (dim*n_bases)

		Eigen::MatrixXd X = io::Evaluator::get_bases_position(state_.n_bases, state_.mesh_nodes);

		jac.setZero(dim * dim, full_size_);
		for (int i = 0; i < X.rows(); i++)
			for (int j = 0; j < dim; j++)
				for (int k = 0; k < dim; k++)
					jac(j * dim + k, i * dim + j) = X(i, k);

		return macro_full_to_reduced_grad(jac);
	}

	Eigen::MatrixXd NLHomoProblem::constraint_values(const TVector &) const
	{
		Eigen::MatrixXd result = Eigen::MatrixXd::Zero(full_size(), 1);
		return result;
	}
} // namespace polyfem::solver
