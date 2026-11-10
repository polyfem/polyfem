#include "NavierStokesFSIForm.hpp"

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <cassert>

namespace polyfem::solver
{
	namespace
	{
		bool same_quadrature(const quadrature::Quadrature &a, const quadrature::Quadrature &b)
		{
			return a.points.rows() == b.points.rows()
				   && a.points.cols() == b.points.cols()
				   && a.weights.size() == b.weights.size()
				   && (a.points - b.points).norm() < 1e-14
				   && (a.weights - b.weights).norm() < 1e-14;
		}
	} // namespace

	NavierStokesFSIAveragePressureForm::NavierStokesFSIAveragePressureForm(
		const int total_size,
		const int n_velocity_bases,
		const int n_pressure_bases,
		const int n_mesh_displacement_bases,
		const int multiplier_offset,
		const int dim,
		const std::vector<basis::ElementBases> &pressure_bases,
		const std::vector<basis::ElementBases> &mesh_displacement_bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::AssemblyValsCache &pressure_cache,
		const assembler::AssemblyValsCache &mesh_displacement_cache,
		const bool is_volume)
		: total_size_(total_size),
		  dim_(dim),
		  n_pressure_bases_(n_pressure_bases),
		  n_mesh_displacement_bases_(n_mesh_displacement_bases),
		  pressure_offset_(n_velocity_bases * dim),
		  mesh_displacement_offset_(pressure_offset_ + n_pressure_bases),
		  multiplier_offset_(multiplier_offset),
		  pressure_bases_(pressure_bases),
		  mesh_displacement_bases_(mesh_displacement_bases),
		  geom_bases_(geom_bases),
		  pressure_cache_(pressure_cache),
		  mesh_displacement_cache_(mesh_displacement_cache),
		  is_volume_(is_volume)
	{
		assert(dim_ == 2 || dim_ == 3);
		assert(n_pressure_bases_ > 0);
		assert(n_mesh_displacement_bases_ > 0);
		assert(multiplier_offset_ >= mesh_displacement_offset_ + n_mesh_displacement_bases * dim);
		assert(total_size_ == multiplier_offset_ + 1);
		assert(pressure_bases_.size() == mesh_displacement_bases_.size());
		assert(pressure_bases_.size() == geom_bases_.size());
	}

	double NavierStokesFSIAveragePressureForm::value_unweighted(const Eigen::VectorXd &) const
	{
		log_and_throw_error("NavierStokesFSIAveragePressureForm is a residual form and has no value()");
	}

	void NavierStokesFSIAveragePressureForm::compute_constraint(
		const Eigen::VectorXd &x,
		Eigen::VectorXd &weights,
		Eigen::MatrixXd &weight_derivative) const
	{
		assert(x.size() == total_size_);
		const int mesh_ndof = n_mesh_displacement_bases_ * dim_;
		weights = Eigen::VectorXd::Zero(n_pressure_bases_);
		weight_derivative = Eigen::MatrixXd::Zero(n_pressure_bases_, mesh_ndof);
		Eigen::VectorXd volume_derivative = Eigen::VectorXd::Zero(mesh_ndof);
		double volume = 0;

		for (int e = 0; e < int(geom_bases_.size()); ++e)
		{
			assembler::ElementAssemblyValues pressure_vals, displacement_vals;
			pressure_cache_.compute(e, is_volume_, pressure_bases_[e], geom_bases_[e], pressure_vals);
			mesh_displacement_cache_.compute(e, is_volume_, mesh_displacement_bases_[e], geom_bases_[e], displacement_vals);

			const quadrature::Quadrature quadrature =
				pressure_vals.quadrature.weights.size() >= displacement_vals.quadrature.weights.size()
					? pressure_vals.quadrature
					: displacement_vals.quadrature;
			if (!same_quadrature(pressure_vals.quadrature, quadrature))
			{
				pressure_vals.compute(e, is_volume_, quadrature.points, pressure_bases_[e], geom_bases_[e]);
				pressure_vals.quadrature = quadrature;
			}
			if (!same_quadrature(displacement_vals.quadrature, quadrature))
			{
				displacement_vals.compute(e, is_volume_, quadrature.points, mesh_displacement_bases_[e], geom_bases_[e]);
				displacement_vals.quadrature = quadrature;
			}

			Eigen::VectorXd local_displacement = Eigen::VectorXd::Zero(
				int(displacement_vals.basis_values.size()) * dim_);
			for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
				for (int c = 0; c < dim_; ++c)
					for (const auto &global : displacement_vals.basis_values[a].global)
						local_displacement(a * dim_ + c) +=
							global.val * x(mesh_displacement_offset_ + global.index * dim_ + c);

			for (int q = 0; q < quadrature.weights.size(); ++q)
			{
				Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dim_, dim_);
				for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
				{
					const Eigen::RowVectorXd grad =
						displacement_vals.basis_values[a].grad.row(q) * displacement_vals.jac_it[q];
					for (int c = 0; c < dim_; ++c)
						F.row(c) += local_displacement(a * dim_ + c) * grad;
				}
				const double J = F.determinant();
				assert(J > 0);
				const Eigen::MatrixXd F_inv = F.inverse();
				const double reference_weight = pressure_vals.det(q) * quadrature.weights(q);
				volume += reference_weight * J;

				for (int a = 0; a < int(displacement_vals.basis_values.size()); ++a)
				{
					const Eigen::RowVectorXd spatial_grad =
						displacement_vals.basis_values[a].grad.row(q) * displacement_vals.jac_it[q] * F_inv;
					for (int c = 0; c < dim_; ++c)
					{
						const double local_dJ = J * spatial_grad(c);
						for (const auto &displacement_global : displacement_vals.basis_values[a].global)
						{
							const int displacement_dof = displacement_global.index * dim_ + c;
							const double dJ = displacement_global.val * local_dJ;
							volume_derivative(displacement_dof) += reference_weight * dJ;
							for (int i = 0; i < int(pressure_vals.basis_values.size()); ++i)
								for (const auto &pressure_global : pressure_vals.basis_values[i].global)
									weight_derivative(pressure_global.index, displacement_dof) +=
										pressure_global.val * pressure_vals.basis_values[i].val(q)
										* reference_weight * dJ;
						}
					}
				}

				for (int i = 0; i < int(pressure_vals.basis_values.size()); ++i)
					for (const auto &pressure_global : pressure_vals.basis_values[i].global)
						weights(pressure_global.index) += pressure_global.val
														  * pressure_vals.basis_values[i].val(q) * reference_weight * J;
			}
		}

		assert(volume > 0);
		weight_derivative =
			(weight_derivative * volume - weights * volume_derivative.transpose()) / (volume * volume);
		weights /= volume;
	}

	void NavierStokesFSIAveragePressureForm::first_derivative_unweighted(
		const Eigen::VectorXd &x, Eigen::VectorXd &residual) const
	{
		Eigen::VectorXd weights;
		Eigen::MatrixXd weight_derivative;
		compute_constraint(x, weights, weight_derivative);
		const Eigen::VectorXd pressure = x.segment(pressure_offset_, n_pressure_bases_);
		const double multiplier = x(multiplier_offset_);
		residual = Eigen::VectorXd::Zero(total_size_);
		residual.segment(pressure_offset_, n_pressure_bases_) = multiplier * weights;
		residual(multiplier_offset_) = weights.dot(pressure);
	}

	void NavierStokesFSIAveragePressureForm::second_derivative_unweighted(
		const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const
	{
		Eigen::VectorXd weights;
		Eigen::MatrixXd weight_derivative;
		compute_constraint(x, weights, weight_derivative);
		const Eigen::VectorXd pressure = x.segment(pressure_offset_, n_pressure_bases_);
		const double multiplier = x(multiplier_offset_);
		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(2 * n_pressure_bases_
						+ n_pressure_bases_ * n_mesh_displacement_bases_ * dim_
						+ n_mesh_displacement_bases_ * dim_);
		for (int i = 0; i < n_pressure_bases_; ++i)
		{
			entries.emplace_back(pressure_offset_ + i, multiplier_offset_, weights(i));
			entries.emplace_back(multiplier_offset_, pressure_offset_ + i, weights(i));
			for (int j = 0; j < n_mesh_displacement_bases_ * dim_; ++j)
			{
				const double value = weight_derivative(i, j);
				if (value != 0)
					entries.emplace_back(pressure_offset_ + i, mesh_displacement_offset_ + j, multiplier * value);
			}
		}
		for (int j = 0; j < n_mesh_displacement_bases_ * dim_; ++j)
		{
			const double value = pressure.dot(weight_derivative.col(j));
			if (value != 0)
				entries.emplace_back(multiplier_offset_, mesh_displacement_offset_ + j, value);
		}
		jacobian.resize(total_size_, total_size_);
		jacobian.setFromTriplets(entries.begin(), entries.end());
		jacobian.makeCompressed();
	}

	NavierStokesFSIForm::NavierStokesFSIForm(
		const int total_size,
		const int n_velocity_bases,
		const int n_pressure_bases,
		const int n_mesh_displacement_bases,
		const std::vector<basis::ElementBases> &velocity_bases,
		const std::vector<basis::ElementBases> &pressure_bases,
		const std::vector<basis::ElementBases> &mesh_displacement_bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::AssemblyValsCache &velocity_cache,
		const assembler::AssemblyValsCache &pressure_cache,
		const assembler::AssemblyValsCache &mesh_displacement_cache,
		std::vector<std::shared_ptr<assembler::MultiSpacesNLAssembler>> assemblers,
		const time_integrator::ImplicitTimeIntegrator *velocity_time_integrator,
		const time_integrator::ImplicitTimeIntegrator *mesh_displacement_time_integrator,
		const double t,
		const double dt,
		const bool is_volume,
		BodyForceEvaluator body_force_evaluator)
		: total_size_(total_size),
		  dim_(assemblers.empty() ? -1 : assemblers.front()->size()),
		  n_bases_({{n_velocity_bases, n_pressure_bases, n_mesh_displacement_bases}}),
		  components_({{dim_, 1, dim_}}),
		  global_offsets_({{0, n_velocity_bases * dim_, n_velocity_bases * dim_ + n_pressure_bases}}),
		  global_sizes_({{n_velocity_bases * dim_, n_pressure_bases, n_mesh_displacement_bases * dim_}}),
		  bases_({{std::cref(velocity_bases), std::cref(pressure_bases), std::cref(mesh_displacement_bases)}}),
		  geom_bases_(geom_bases),
		  caches_({{std::cref(velocity_cache), std::cref(pressure_cache), std::cref(mesh_displacement_cache)}}),
		  assemblers_(std::move(assemblers)),
		  velocity_time_integrator_(velocity_time_integrator),
		  mesh_displacement_time_integrator_(mesh_displacement_time_integrator),
		  t_(t),
		  dt_(dt),
		  is_volume_(is_volume),
		  body_force_evaluator_(std::move(body_force_evaluator))
	{
		assert(dim_ == 2 || dim_ == 3);
		assert(!assemblers_.empty());
		assert(velocity_bases.size() == pressure_bases.size());
		assert(velocity_bases.size() == mesh_displacement_bases.size());
		assert(velocity_bases.size() == geom_bases.size());
		assert(total_size_ >= global_offsets_[2] + global_sizes_[2]);
		x_prev_ = Eigen::VectorXd::Zero(total_size_);
	}

	void NavierStokesFSIForm::compute_element_values(
		const int element, SpaceValues &vals, QuadratureVector &da) const
	{
		for (int s = 0; s < 3; ++s)
			caches_[s].get().compute(element, is_volume_, bases_[s].get()[element], geom_bases_[element], vals[s]);

		int selected = 0;
		for (int s = 1; s < 3; ++s)
			if (vals[s].quadrature.weights.size() > vals[selected].quadrature.weights.size())
				selected = s;
		const quadrature::Quadrature quadrature = vals[selected].quadrature;
		for (int s = 0; s < 3; ++s)
		{
			if (!same_quadrature(vals[s].quadrature, quadrature))
			{
				vals[s].compute(element, is_volume_, quadrature.points, bases_[s].get()[element], geom_bases_[element]);
				vals[s].quadrature = quadrature;
			}
		}
		da = vals[0].det.array() * quadrature.weights.array();
	}

	Eigen::VectorXd NavierStokesFSIForm::gather(
		const Eigen::VectorXd &x,
		const assembler::ElementAssemblyValues &vals,
		const int components,
		const int global_offset) const
	{
		Eigen::VectorXd local = Eigen::VectorXd::Zero(int(vals.basis_values.size()) * components);
		for (int i = 0; i < int(vals.basis_values.size()); ++i)
			for (int c = 0; c < components; ++c)
				for (const auto &global : vals.basis_values[i].global)
					local(i * components + c) += global.val * x(global_offset + global.index * components + c);
		return local;
	}

	assembler::NavierStokesFSIAssemblerData NavierStokesFSIForm::make_data(
		const SpaceValues &vals,
		const LocalCoefficients &x,
		const LocalCoefficients &x_prev,
		const QuadratureVector &da,
		const Eigen::VectorXd &velocity_tilde,
		const Eigen::VectorXd &mesh_velocity) const
	{
		using Data = assembler::NavierStokesFSIAssemblerData;
		Data::Values value_refs = {std::cref(vals[0]), std::cref(vals[1]), std::cref(vals[2])};
		Data::Coefficients x_refs = {std::cref(x[0]), std::cref(x[1]), std::cref(x[2])};
		Data::Coefficients prev_refs = {std::cref(x_prev[0]), std::cref(x_prev[1]), std::cref(x_prev[2])};
		return Data(
			std::move(value_refs), std::move(x_refs), std::move(prev_refs),
			t_, dt_, da, velocity_tilde, mesh_velocity,
			mesh_displacement_time_integrator_ ? mesh_displacement_time_integrator_->dv_dx() : 0,
			velocity_time_integrator_ ? velocity_time_integrator_->acceleration_scaling() : 1,
			velocity_time_integrator_ != nullptr,
			project_to_psd_, body_force_evaluator_);
	}

	void NavierStokesFSIForm::scatter_local_residual(
		const SpaceValues &vals, const Eigen::VectorXd &local, Eigen::VectorXd &global) const
	{
		int local_offset = 0;
		for (int s = 0; s < 3; ++s)
		{
			for (int i = 0; i < int(vals[s].basis_values.size()); ++i)
				for (int c = 0; c < components_[s]; ++c)
					for (const auto &mapping : vals[s].basis_values[i].global)
						global(global_offsets_[s] + mapping.index * components_[s] + c) += mapping.val * local(local_offset + i * components_[s] + c);
			local_offset += int(vals[s].basis_values.size()) * components_[s];
		}
	}

	void NavierStokesFSIForm::scatter_local_block(
		const SpaceValues &vals,
		const int row_space,
		const int col_space,
		const Eigen::MatrixXd &local,
		std::vector<Eigen::Triplet<double>> &entries) const
	{
		const int row_components = components_[row_space];
		const int col_components = components_[col_space];
		assert(local.rows() == int(vals[row_space].basis_values.size()) * row_components);
		assert(local.cols() == int(vals[col_space].basis_values.size()) * col_components);
		for (int i = 0; i < int(vals[row_space].basis_values.size()); ++i)
			for (int rc = 0; rc < row_components; ++rc)
				for (int j = 0; j < int(vals[col_space].basis_values.size()); ++j)
					for (int cc = 0; cc < col_components; ++cc)
					{
						const double value = local(i * row_components + rc, j * col_components + cc);
						if (value == 0)
							continue;
						for (const auto &row : vals[row_space].basis_values[i].global)
							for (const auto &col : vals[col_space].basis_values[j].global)
								entries.emplace_back(
									global_offsets_[row_space] + row.index * row_components + rc,
									global_offsets_[col_space] + col.index * col_components + cc,
									row.val * col.val * value);
					}
	}

	void NavierStokesFSIForm::first_derivative_unweighted(
		const Eigen::VectorXd &x, Eigen::VectorXd &residual) const
	{
		assert(x.size() == total_size_);
		residual = Eigen::VectorXd::Zero(total_size_);

		Eigen::VectorXd velocity_tilde = Eigen::VectorXd::Zero(velocity_ndof());
		if (velocity_time_integrator_)
		{
			velocity_tilde = velocity_time_integrator_->x_tilde();
			if (velocity_tilde_updater_)
				velocity_tilde_updater_(t_, x.segment(global_offsets_[0], global_sizes_[0]), velocity_tilde);
		}
		Eigen::VectorXd mesh_velocity = Eigen::VectorXd::Zero(mesh_displacement_ndof());
		if (mesh_displacement_time_integrator_)
			mesh_velocity = mesh_displacement_time_integrator_->compute_velocity(
				x.segment(global_offsets_[2], global_sizes_[2]));

		for (int e = 0; e < int(geom_bases_.size()); ++e)
		{
			SpaceValues vals;
			QuadratureVector da;
			compute_element_values(e, vals, da);
			LocalCoefficients local_x, local_prev;
			for (int s = 0; s < 3; ++s)
			{
				local_x[s] = gather(x, vals[s], components_[s], global_offsets_[s]);
				local_prev[s] = gather(x_prev_, vals[s], components_[s], global_offsets_[s]);
			}
			const Eigen::VectorXd local_velocity_tilde = gather(velocity_tilde, vals[0], dim_, 0);
			const Eigen::VectorXd local_mesh_velocity = gather(mesh_velocity, vals[2], dim_, 0);
			const auto data = make_data(vals, local_x, local_prev, da, local_velocity_tilde, local_mesh_velocity);
			Eigen::VectorXd local_residual = Eigen::VectorXd::Zero(
				local_x[0].size() + local_x[1].size() + local_x[2].size());
			for (const auto &assembler : assemblers_)
				local_residual += assembler->assemble_gradient(data);
			scatter_local_residual(vals, local_residual, residual);
		}
	}

	double NavierStokesFSIForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd residual;
		first_derivative_unweighted(x, residual);
		return residual.squaredNorm();
	}

	void NavierStokesFSIForm::second_derivative_unweighted(
		const Eigen::VectorXd &x, StiffnessMatrix &jacobian) const
	{
		assert(x.size() == total_size_);
		std::vector<Eigen::Triplet<double>> entries;

		Eigen::VectorXd velocity_tilde = Eigen::VectorXd::Zero(velocity_ndof());
		if (velocity_time_integrator_)
		{
			velocity_tilde = velocity_time_integrator_->x_tilde();
			if (velocity_tilde_updater_)
				velocity_tilde_updater_(t_, x.segment(global_offsets_[0], global_sizes_[0]), velocity_tilde);
		}
		Eigen::VectorXd mesh_velocity = Eigen::VectorXd::Zero(mesh_displacement_ndof());
		if (mesh_displacement_time_integrator_)
			mesh_velocity = mesh_displacement_time_integrator_->compute_velocity(
				x.segment(global_offsets_[2], global_sizes_[2]));

		for (int e = 0; e < int(geom_bases_.size()); ++e)
		{
			SpaceValues vals;
			QuadratureVector da;
			compute_element_values(e, vals, da);
			LocalCoefficients local_x, local_prev;
			for (int s = 0; s < 3; ++s)
			{
				local_x[s] = gather(x, vals[s], components_[s], global_offsets_[s]);
				local_prev[s] = gather(x_prev_, vals[s], components_[s], global_offsets_[s]);
			}
			const Eigen::VectorXd local_velocity_tilde = gather(velocity_tilde, vals[0], dim_, 0);
			const Eigen::VectorXd local_mesh_velocity = gather(mesh_velocity, vals[2], dim_, 0);
			const auto data = make_data(vals, local_x, local_prev, da, local_velocity_tilde, local_mesh_velocity);

			for (const int row_space : {0, 1})
				for (const int col_space : {0, 1, 2})
				{
					Eigen::MatrixXd block = Eigen::MatrixXd::Zero(local_x[row_space].size(), local_x[col_space].size());
					for (const auto &assembler : assemblers_)
						block += assembler->assemble_hessian(data, row_space, col_space);
					scatter_local_block(vals, row_space, col_space, block, entries);
				}
		}

		jacobian.resize(total_size_, total_size_);
		jacobian.setFromTriplets(entries.begin(), entries.end());
		jacobian.makeCompressed();
	}

	void NavierStokesFSIForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		t_ = t;
		if (x.size() == total_size_)
			x_prev_ = x;
	}

	bool NavierStokesFSIForm::has_valid_ale_mapping(const Eigen::VectorXd &x) const
	{
		for (int e = 0; e < int(geom_bases_.size()); ++e)
		{
			SpaceValues vals;
			QuadratureVector da;
			compute_element_values(e, vals, da);
			const Eigen::VectorXd local_d = gather(x, vals[2], dim_, global_offsets_[2]);
			for (int q = 0; q < da.size(); ++q)
			{
				Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dim_, dim_);
				for (int a = 0; a < int(vals[2].basis_values.size()); ++a)
				{
					const Eigen::RowVectorXd grad = vals[2].basis_values[a].grad.row(q) * vals[2].jac_it[q];
					for (int c = 0; c < dim_; ++c)
						F.row(c) += local_d(a * dim_ + c) * grad;
				}
				if (!(F.determinant() > 1e-8))
					return false;
			}
		}
		return true;
	}

	bool NavierStokesFSIForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const
	{
		return has_valid_ale_mapping(x1);
	}
} // namespace polyfem::solver
