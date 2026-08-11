#include "NavierStokesFSI.hpp"

#include <polyfem/utils/Logger.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace polyfem::assembler
{
	namespace
	{
		using FSIData = NavierStokesFSIAssemblerData;

		const FSIData &fsi_data(const MultiSpacesNLAssemblerData &data)
		{
#ifndef NDEBUG
			assert(dynamic_cast<const FSIData *>(&data) != nullptr);
#endif
			return static_cast<const FSIData &>(data);
		}

		int components(const FSIData &data, const int space)
		{
			return space == FSIData::Pressure ? 1 : int(data.vals(space).val.cols());
		}

		int local_size(const FSIData &data, const int space)
		{
			return int(data.vals(space).basis_values.size()) * components(data, space);
		}

		int local_total_size(const FSIData &data)
		{
			return local_size(data, FSIData::Velocity)
				   + local_size(data, FSIData::Pressure)
				   + local_size(data, FSIData::MeshDisplacement);
		}

		int local_offset(const FSIData &data, const int space)
		{
			if (space == FSIData::Velocity)
				return 0;
			if (space == FSIData::Pressure)
				return local_size(data, FSIData::Velocity);
			assert(space == FSIData::MeshDisplacement);
			return local_size(data, FSIData::Velocity) + local_size(data, FSIData::Pressure);
		}

		Eigen::RowVectorXd reference_gradient(
			const ElementAssemblyValues &vals, const int basis, const int q)
		{
			return vals.basis_values[basis].grad.row(q) * vals.jac_it[q];
		}

		struct ALEPoint
		{
			Eigen::MatrixXd F;
			Eigen::MatrixXd F_inv;
			double J = 1;
			Eigen::RowVectorXd point;
		};

		ALEPoint ale_point(const FSIData &data, const int q)
		{
			const auto &dvals = data.vals(FSIData::MeshDisplacement);
			const int dim = components(data, FSIData::MeshDisplacement);
			ALEPoint out;
			out.F = Eigen::MatrixXd::Identity(dim, dim);
			out.point = data.vals(FSIData::Velocity).val.row(q);

			for (int a = 0; a < int(dvals.basis_values.size()); ++a)
			{
				const Eigen::RowVectorXd grad = reference_gradient(dvals, a, q);
				for (int c = 0; c < dim; ++c)
				{
					const double d = data.mesh_displacement()(a * dim + c);
					out.F.row(c) += d * grad;
					out.point(c) += dvals.basis_values[a].val(q) * d;
				}
			}

			out.J = out.F.determinant();
			out.F_inv = out.F.inverse();
			return out;
		}

		Eigen::RowVectorXd spatial_gradient(
			const FSIData &data, const int space, const int basis, const int q,
			const Eigen::MatrixXd &F_inv)
		{
			return reference_gradient(data.vals(space), basis, q) * F_inv;
		}

		Eigen::VectorXd interpolate_vector(
			const ElementAssemblyValues &vals,
			const Eigen::VectorXd &coefficients,
			const int dim,
			const int q)
		{
			Eigen::VectorXd result = Eigen::VectorXd::Zero(dim);
			for (int a = 0; a < int(vals.basis_values.size()); ++a)
				for (int c = 0; c < dim; ++c)
					result(c) += vals.basis_values[a].val(q) * coefficients(a * dim + c);
			return result;
		}

		Eigen::MatrixXd velocity_gradient(
			const FSIData &data, const int q, const Eigen::MatrixXd &F_inv)
		{
			const int dim = components(data, FSIData::Velocity);
			const auto &vals = data.vals(FSIData::Velocity);
			Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dim, dim);
			for (int a = 0; a < int(vals.basis_values.size()); ++a)
			{
				const Eigen::RowVectorXd grad = spatial_gradient(data, FSIData::Velocity, a, q, F_inv);
				for (int c = 0; c < dim; ++c)
					result.row(c) += data.velocity()(a * dim + c) * grad;
			}
			return result;
		}

		double interpolate_scalar(
			const ElementAssemblyValues &vals,
			const Eigen::VectorXd &coefficients,
			const int q)
		{
			double result = 0;
			for (int a = 0; a < int(vals.basis_values.size()); ++a)
				result += vals.basis_values[a].val(q) * coefficients(a);
			return result;
		}

		template <typename Function>
		double coordinate_derivative(
			const Eigen::RowVectorXd &point, const int coordinate, Function function)
		{
			const double h = 1e-7 * std::max(1.0, std::abs(point(coordinate)));
			Eigen::RowVectorXd plus = point;
			Eigen::RowVectorXd minus = point;
			plus(coordinate) += h;
			minus(coordinate) -= h;
			return (function(plus) - function(minus)) / (2 * h);
		}

		void evaluate_body_force(
			const FSIData &data,
			const std::vector<ALEPoint> &ale,
			Eigen::MatrixXd &body_force,
			std::vector<Eigen::MatrixXd> *body_force_gradient = nullptr)
		{
			const int dim = components(data, FSIData::Velocity);
			const int n_q = int(ale.size());
			body_force = Eigen::MatrixXd::Zero(n_q, dim);
			if (body_force_gradient != nullptr)
				body_force_gradient->assign(dim, Eigen::MatrixXd::Zero(n_q, dim));
			if (!data.body_force_evaluator)
				return;

			Eigen::MatrixXd points(n_q, dim);
			for (int q = 0; q < n_q; ++q)
				points.row(q) = ale[q].point;
			data.body_force_evaluator(
				data.vals(FSIData::Velocity).element_id, points, data.t, body_force);

			if (body_force_gradient == nullptr)
				return;
			for (int n = 0; n < dim; ++n)
			{
				Eigen::MatrixXd plus_points = points;
				Eigen::MatrixXd minus_points = points;
				Eigen::VectorXd steps(n_q);
				for (int q = 0; q < n_q; ++q)
				{
					steps(q) = 1e-7 * std::max(1.0, std::abs(points(q, n)));
					plus_points(q, n) += steps(q);
					minus_points(q, n) -= steps(q);
				}
				Eigen::MatrixXd plus, minus;
				data.body_force_evaluator(
					data.vals(FSIData::Velocity).element_id, plus_points, data.t, plus);
				data.body_force_evaluator(
					data.vals(FSIData::Velocity).element_id, minus_points, data.t, minus);
				for (int q = 0; q < n_q; ++q)
					body_force_gradient->at(n).row(q) = (plus.row(q) - minus.row(q)) / (2 * steps(q));
			}
		}
	} // namespace

	NavierStokesFSIVelocity::NavierStokesFSIVelocity()
		: viscosity_("viscosity")
	{
	}

	void NavierStokesFSIVelocity::add_multimaterial(
		const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);
		viscosity_.add_multimaterial(index, params, units.viscosity(), root_path);
		density_.add_multimaterial(index, params, units.density(), root_path);
	}

	double NavierStokesFSIVelocity::compute_energy(const MultiSpacesNLAssemblerData &) const
	{
		log_and_throw_error("NavierStokesFSIVelocity is residual based and has no energy");
	}

	Eigen::VectorXd NavierStokesFSIVelocity::assemble_gradient(const MultiSpacesNLAssemblerData &base_data) const
	{
		const FSIData &data = fsi_data(base_data);
		const auto &vvals = data.vals(FSIData::Velocity);
		const int dim = components(data, FSIData::Velocity);
		Eigen::VectorXd residual = Eigen::VectorXd::Zero(local_total_size(data));

		Eigen::MatrixXd body_force;
		if (data.body_force_evaluator)
		{
			Eigen::MatrixXd points(data.da.size(), dim);
			for (int q = 0; q < data.da.size(); ++q)
				points.row(q) = ale_point(data, q).point;
			data.body_force_evaluator(vvals.element_id, points, data.t, body_force);
		}
		if (body_force.size() == 0)
			body_force = Eigen::MatrixXd::Zero(data.da.size(), dim);

		for (int q = 0; q < data.da.size(); ++q)
		{
			const ALEPoint ale = ale_point(data, q);
			const Eigen::VectorXd velocity = interpolate_vector(vvals, data.velocity(), dim, q);
			const Eigen::VectorXd mesh_velocity = interpolate_vector(
				data.vals(FSIData::MeshDisplacement), data.mesh_velocity, dim, q);
			const Eigen::MatrixXd grad_velocity = velocity_gradient(data, q, ale.F_inv);
			const double viscosity = viscosity_(ale.point, data.t, vvals.element_id);
			const double rho = density_(vvals.quadrature.points.row(q), ale.point, data.t, vvals.element_id);
			const double weight = data.spatial_weight * ale.J * data.da(q);

			for (int i = 0; i < int(vvals.basis_values.size()); ++i)
			{
				const double phi_i = vvals.basis_values[i].val(q);
				const Eigen::RowVectorXd grad_i = spatial_gradient(data, FSIData::Velocity, i, q, ale.F_inv);
				for (int m = 0; m < dim; ++m)
				{
					const double viscous = viscosity * grad_velocity.row(m).dot(grad_i);
					const double convection = rho * (velocity - mesh_velocity).dot(grad_velocity.row(m)) * phi_i;
					const double force = -rho * body_force(q, m) * phi_i;
					residual(i * dim + m) += weight * (viscous + convection + force);
				}
			}
		}
		return residual;
	}

	Eigen::MatrixXd NavierStokesFSIVelocity::assemble_hessian(
		const MultiSpacesNLAssemblerData &base_data, const int row_space, const int col_space) const
	{
		const FSIData &data = fsi_data(base_data);
		if (row_space != FSIData::Velocity)
			return Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
		if (col_space == FSIData::MeshDisplacement)
		{
			const auto &vvals = data.vals(FSIData::Velocity);
			const auto &dvals = data.vals(FSIData::MeshDisplacement);
			const int dim = components(data, FSIData::Velocity);
			Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
			std::vector<ALEPoint> ale(data.da.size());
			for (int q = 0; q < data.da.size(); ++q)
				ale[q] = ale_point(data, q);
			Eigen::MatrixXd body_force;
			std::vector<Eigen::MatrixXd> body_force_gradient;
			evaluate_body_force(data, ale, body_force, &body_force_gradient);

			for (int q = 0; q < data.da.size(); ++q)
			{
				const Eigen::VectorXd velocity = interpolate_vector(vvals, data.velocity(), dim, q);
				const Eigen::VectorXd mesh_velocity = interpolate_vector(dvals, data.mesh_velocity, dim, q);
				const Eigen::VectorXd relative_velocity = velocity - mesh_velocity;
				const Eigen::MatrixXd grad_velocity = velocity_gradient(data, q, ale[q].F_inv);
				const double viscosity = viscosity_(ale[q].point, data.t, vvals.element_id);
				const double rho = density_(vvals.quadrature.points.row(q), ale[q].point, data.t, vvals.element_id);
				Eigen::VectorXd viscosity_gradient(dim), density_gradient(dim);
				for (int n = 0; n < dim; ++n)
				{
					viscosity_gradient(n) = coordinate_derivative(
						ale[q].point, n,
						[&](const Eigen::RowVectorXd &point) { return viscosity_(point, data.t, vvals.element_id); });
					density_gradient(n) = coordinate_derivative(
						ale[q].point, n,
						[&](const Eigen::RowVectorXd &point) {
							return density_(vvals.quadrature.points.row(q), point, data.t, vvals.element_id);
						});
				}
				const double weight = data.spatial_weight * ale[q].J * data.da(q);

				for (int i = 0; i < int(vvals.basis_values.size()); ++i)
				{
					const double phi_i = vvals.basis_values[i].val(q);
					const Eigen::RowVectorXd grad_i = spatial_gradient(data, FSIData::Velocity, i, q, ale[q].F_inv);
					for (int j = 0; j < int(dvals.basis_values.size()); ++j)
					{
						const double phi_j = dvals.basis_values[j].val(q);
						const Eigen::RowVectorXd grad_j = spatial_gradient(data, FSIData::MeshDisplacement, j, q, ale[q].F_inv);
						for (int n = 0; n < dim; ++n)
						{
							const double theta = grad_j(n);
							const Eigen::VectorXd dmesh_velocity =
								phi_j * data.dmesh_velocity_dmesh_displacement * Eigen::VectorXd::Unit(dim, n);
							const double dviscosity = phi_j * viscosity_gradient(n);
							const double drho = phi_j * density_gradient(n);
							for (int m = 0; m < dim; ++m)
							{
								const Eigen::RowVectorXd dgrad_velocity = -grad_velocity(m, n) * grad_j;
								const Eigen::RowVectorXd dgrad_i = -grad_i(n) * grad_j;
								const double viscous = viscosity * grad_velocity.row(m).dot(grad_i);
								const double convection = rho * relative_velocity.dot(grad_velocity.row(m)) * phi_i;
								const double force = -rho * body_force(q, m) * phi_i;
								double derivative = theta * (viscous + convection + force);
								derivative += dviscosity * grad_velocity.row(m).dot(grad_i)
											  + viscosity * (dgrad_velocity.dot(grad_i) + grad_velocity.row(m).dot(dgrad_i));
								derivative += drho * relative_velocity.dot(grad_velocity.row(m)) * phi_i
											  + rho * (-dmesh_velocity.dot(grad_velocity.row(m)) + relative_velocity.dot(dgrad_velocity))
													* phi_i;
								derivative -= (drho * body_force(q, m)
											   + rho * phi_j * body_force_gradient[n](q, m))
											  * phi_i;
								jacobian(i * dim + m, j * dim + n) += weight * derivative;
							}
						}
					}
				}
			}
			return jacobian;
		}
		if (col_space != FSIData::Velocity)
			return Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));

		const auto &vvals = data.vals(FSIData::Velocity);
		const int dim = components(data, FSIData::Velocity);
		Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
		for (int q = 0; q < data.da.size(); ++q)
		{
			const ALEPoint ale = ale_point(data, q);
			const Eigen::VectorXd velocity = interpolate_vector(vvals, data.velocity(), dim, q);
			const Eigen::VectorXd mesh_velocity = interpolate_vector(
				data.vals(FSIData::MeshDisplacement), data.mesh_velocity, dim, q);
			const Eigen::MatrixXd grad_velocity = velocity_gradient(data, q, ale.F_inv);
			const double viscosity = viscosity_(ale.point, data.t, vvals.element_id);
			const double rho = density_(vvals.quadrature.points.row(q), ale.point, data.t, vvals.element_id);
			const double weight = data.spatial_weight * ale.J * data.da(q);

			for (int i = 0; i < int(vvals.basis_values.size()); ++i)
			{
				const double phi_i = vvals.basis_values[i].val(q);
				const Eigen::RowVectorXd grad_i = spatial_gradient(data, FSIData::Velocity, i, q, ale.F_inv);
				for (int j = 0; j < int(vvals.basis_values.size()); ++j)
				{
					const double phi_j = vvals.basis_values[j].val(q);
					const Eigen::RowVectorXd grad_j = spatial_gradient(data, FSIData::Velocity, j, q, ale.F_inv);
					for (int m = 0; m < dim; ++m)
						for (int n = 0; n < dim; ++n)
						{
							double value = 0;
							if (m == n)
								value += viscosity * grad_i.dot(grad_j)
										 + rho * (velocity - mesh_velocity).dot(grad_j) * phi_i;
							if (!data.picard)
								value += rho * phi_j * grad_velocity(m, n) * phi_i;
							jacobian(i * dim + m, j * dim + n) += weight * value;
						}
				}
			}
		}
		return jacobian;
	}

	std::map<std::string, Assembler::ParamFunc> NavierStokesFSIVelocity::parameters() const
	{
		return {
			{"viscosity", [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) { return viscosity_(p, t, e); }},
			{"rho", [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return density_(uv, p, t, e); }}};
	}

	double NavierStokesFSIMixed::compute_energy(const MultiSpacesNLAssemblerData &) const
	{
		log_and_throw_error("NavierStokesFSIMixed is residual based and has no energy");
	}

	Eigen::VectorXd NavierStokesFSIMixed::assemble_gradient(const MultiSpacesNLAssemblerData &base_data) const
	{
		const FSIData &data = fsi_data(base_data);
		const auto &vvals = data.vals(FSIData::Velocity);
		const auto &pvals = data.vals(FSIData::Pressure);
		const int dim = components(data, FSIData::Velocity);
		Eigen::VectorXd residual = Eigen::VectorXd::Zero(local_total_size(data));
		const int pressure_offset = local_offset(data, FSIData::Pressure);

		for (int q = 0; q < data.da.size(); ++q)
		{
			const ALEPoint ale = ale_point(data, q);
			const Eigen::MatrixXd grad_velocity = velocity_gradient(data, q, ale.F_inv);
			const double pressure = interpolate_scalar(pvals, data.pressure(), q);
			const double weight = data.spatial_weight * ale.J * data.da(q);

			for (int i = 0; i < int(vvals.basis_values.size()); ++i)
			{
				const Eigen::RowVectorXd grad_i = spatial_gradient(data, FSIData::Velocity, i, q, ale.F_inv);
				for (int m = 0; m < dim; ++m)
					residual(i * dim + m) -= weight * pressure * grad_i(m);
			}
			for (int i = 0; i < int(pvals.basis_values.size()); ++i)
				residual(pressure_offset + i) -= weight * pvals.basis_values[i].val(q) * grad_velocity.trace();
		}
		return residual;
	}

	Eigen::MatrixXd NavierStokesFSIMixed::assemble_hessian(
		const MultiSpacesNLAssemblerData &base_data, const int row_space, const int col_space) const
	{
		const FSIData &data = fsi_data(base_data);
		if (col_space == FSIData::MeshDisplacement
			&& (row_space == FSIData::Velocity || row_space == FSIData::Pressure))
		{
			const auto &vvals = data.vals(FSIData::Velocity);
			const auto &pvals = data.vals(FSIData::Pressure);
			const auto &dvals = data.vals(FSIData::MeshDisplacement);
			const int dim = components(data, FSIData::Velocity);
			Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
			for (int q = 0; q < data.da.size(); ++q)
			{
				const ALEPoint ale = ale_point(data, q);
				const Eigen::MatrixXd grad_velocity = velocity_gradient(data, q, ale.F_inv);
				const double pressure = interpolate_scalar(pvals, data.pressure(), q);
				const double weight = data.spatial_weight * ale.J * data.da(q);
				for (int j = 0; j < int(dvals.basis_values.size()); ++j)
				{
					const Eigen::RowVectorXd grad_j = spatial_gradient(data, FSIData::MeshDisplacement, j, q, ale.F_inv);
					for (int n = 0; n < dim; ++n)
					{
						const double theta = grad_j(n);
						if (row_space == FSIData::Velocity)
						{
							for (int i = 0; i < int(vvals.basis_values.size()); ++i)
							{
								const Eigen::RowVectorXd grad_i = spatial_gradient(data, FSIData::Velocity, i, q, ale.F_inv);
								const Eigen::RowVectorXd dgrad_i = -grad_i(n) * grad_j;
								for (int m = 0; m < dim; ++m)
									jacobian(i * dim + m, j * dim + n) -= weight * pressure * (theta * grad_i(m) + dgrad_i(m));
							}
						}
						else
						{
							double ddivergence = 0;
							for (int c = 0; c < dim; ++c)
								ddivergence -= grad_velocity(c, n) * grad_j(c);
							for (int i = 0; i < int(pvals.basis_values.size()); ++i)
								jacobian(i, j * dim + n) -= weight * pvals.basis_values[i].val(q)
															* (theta * grad_velocity.trace() + ddivergence);
						}
					}
				}
			}
			return jacobian;
		}

		Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
		if (!((row_space == FSIData::Velocity && col_space == FSIData::Pressure)
			  || (row_space == FSIData::Pressure && col_space == FSIData::Velocity)))
			return jacobian;

		const auto &vvals = data.vals(FSIData::Velocity);
		const auto &pvals = data.vals(FSIData::Pressure);
		const int dim = components(data, FSIData::Velocity);
		for (int q = 0; q < data.da.size(); ++q)
		{
			const ALEPoint ale = ale_point(data, q);
			const double weight = data.spatial_weight * ale.J * data.da(q);
			for (int i = 0; i < int(vvals.basis_values.size()); ++i)
			{
				const Eigen::RowVectorXd grad_i = spatial_gradient(data, FSIData::Velocity, i, q, ale.F_inv);
				for (int j = 0; j < int(pvals.basis_values.size()); ++j)
					for (int m = 0; m < dim; ++m)
					{
						const double value = -weight * pvals.basis_values[j].val(q) * grad_i(m);
						if (row_space == FSIData::Velocity)
							jacobian(i * dim + m, j) += value;
						else
							jacobian(j, i * dim + m) += value;
					}
			}
		}
		return jacobian;
	}

	Eigen::VectorXd NavierStokesFSIPressure::assemble_gradient(const MultiSpacesNLAssemblerData &base_data) const
	{
		const FSIData &data = fsi_data(base_data);
		return Eigen::VectorXd::Zero(local_total_size(data));
	}

	Eigen::MatrixXd NavierStokesFSIPressure::assemble_hessian(
		const MultiSpacesNLAssemblerData &base_data, const int row_space, const int col_space) const
	{
		const FSIData &data = fsi_data(base_data);
		return Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
	}

	void NavierStokesFSIInertia::add_multimaterial(
		const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);
		density_.add_multimaterial(index, params, units.density(), root_path);
	}

	double NavierStokesFSIInertia::compute_energy(const MultiSpacesNLAssemblerData &) const
	{
		log_and_throw_error("NavierStokesFSIInertia is residual based and has no energy");
	}

	Eigen::VectorXd NavierStokesFSIInertia::assemble_gradient(const MultiSpacesNLAssemblerData &base_data) const
	{
		const FSIData &data = fsi_data(base_data);
		Eigen::VectorXd residual = Eigen::VectorXd::Zero(local_total_size(data));
		if (!data.include_inertia)
			return residual;

		const auto &vvals = data.vals(FSIData::Velocity);
		const int dim = components(data, FSIData::Velocity);
		for (int q = 0; q < data.da.size(); ++q)
		{
			const ALEPoint ale = ale_point(data, q);
			const Eigen::VectorXd velocity = interpolate_vector(vvals, data.velocity(), dim, q);
			const Eigen::VectorXd velocity_tilde = interpolate_vector(vvals, data.velocity_tilde, dim, q);
			const double rho = density_(vvals.quadrature.points.row(q), ale.point, data.t, vvals.element_id);
			const double weight = rho * ale.J * data.da(q);
			for (int i = 0; i < int(vvals.basis_values.size()); ++i)
				for (int m = 0; m < dim; ++m)
					residual(i * dim + m) += weight * vvals.basis_values[i].val(q) * (velocity(m) - velocity_tilde(m));
		}
		return residual;
	}

	Eigen::MatrixXd NavierStokesFSIInertia::assemble_hessian(
		const MultiSpacesNLAssemblerData &base_data, const int row_space, const int col_space) const
	{
		const FSIData &data = fsi_data(base_data);
		Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(local_size(data, row_space), local_size(data, col_space));
		if (!data.include_inertia || row_space != FSIData::Velocity)
			return jacobian;
		if (col_space == FSIData::MeshDisplacement)
		{
			const auto &vvals = data.vals(FSIData::Velocity);
			const auto &dvals = data.vals(FSIData::MeshDisplacement);
			const int dim = components(data, FSIData::Velocity);
			for (int q = 0; q < data.da.size(); ++q)
			{
				const ALEPoint ale = ale_point(data, q);
				const Eigen::VectorXd velocity = interpolate_vector(vvals, data.velocity(), dim, q);
				const Eigen::VectorXd velocity_tilde = interpolate_vector(vvals, data.velocity_tilde, dim, q);
				const double rho = density_(vvals.quadrature.points.row(q), ale.point, data.t, vvals.element_id);
				Eigen::VectorXd density_gradient(dim);
				for (int n = 0; n < dim; ++n)
					density_gradient(n) = coordinate_derivative(
						ale.point, n,
						[&](const Eigen::RowVectorXd &point) {
							return density_(vvals.quadrature.points.row(q), point, data.t, vvals.element_id);
						});
				const double weight = ale.J * data.da(q);
				for (int i = 0; i < int(vvals.basis_values.size()); ++i)
					for (int j = 0; j < int(dvals.basis_values.size()); ++j)
					{
						const double phi_j = dvals.basis_values[j].val(q);
						const Eigen::RowVectorXd grad_j = spatial_gradient(data, FSIData::MeshDisplacement, j, q, ale.F_inv);
						for (int m = 0; m < dim; ++m)
							for (int n = 0; n < dim; ++n)
								jacobian(i * dim + m, j * dim + n) +=
									weight * vvals.basis_values[i].val(q)
									* (rho * grad_j(n) + phi_j * density_gradient(n))
									* (velocity(m) - velocity_tilde(m));
					}
			}
			return jacobian;
		}
		if (col_space != FSIData::Velocity)
			return jacobian;

		const auto &vvals = data.vals(FSIData::Velocity);
		const int dim = components(data, FSIData::Velocity);
		for (int q = 0; q < data.da.size(); ++q)
		{
			const ALEPoint ale = ale_point(data, q);
			const double rho = density_(vvals.quadrature.points.row(q), ale.point, data.t, vvals.element_id);
			const double weight = rho * ale.J * data.da(q);
			for (int i = 0; i < int(vvals.basis_values.size()); ++i)
				for (int j = 0; j < int(vvals.basis_values.size()); ++j)
					for (int m = 0; m < dim; ++m)
						jacobian(i * dim + m, j * dim + m) += weight * vvals.basis_values[i].val(q) * vvals.basis_values[j].val(q);
		}
		return jacobian;
	}

	std::map<std::string, Assembler::ParamFunc> NavierStokesFSIInertia::parameters() const
	{
		return {{"rho", [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return density_(uv, p, t, e); }}};
	}
} // namespace polyfem::assembler
