#include "ThermoElasticity.hpp"

#include <polyfem/utils/AutodiffTypes.hpp>

#include <cmath>

namespace polyfem::assembler
{
	namespace
	{
		template <typename T>
		void get_local_state(
			const MixedNonLinearAssemblerData &data,
			const int dim,
			Eigen::Matrix<T, Eigen::Dynamic, 1> &local_state)
		{
			const int n_phi_bases = int(data.phi_vals.basis_values.size());
			const int n_psi_bases = int(data.psi_vals.basis_values.size());
			const int phi_local_size = n_phi_bases * dim;
			const int local_size = phi_local_size + n_psi_bases;

			Eigen::VectorXd values = Eigen::VectorXd::Zero(local_size);
			for (int i = 0; i < n_phi_bases; ++i)
			{
				const auto &bs = data.phi_vals.basis_values[i];
				for (const auto &global : bs.global)
				{
					for (int d = 0; d < dim; ++d)
						values(i * dim + d) += global.val * data.x_phi(global.index * dim + d);
				}
			}

			for (int i = 0; i < n_psi_bases; ++i)
			{
				const auto &bs = data.psi_vals.basis_values[i];
				for (const auto &global : bs.global)
					values(phi_local_size + i) += global.val * data.x_psi(global.index);
			}

			DiffScalarBase::setVariableCount(local_size);
			local_state.resize(local_size);

			const AutoDiffAllocator<T> allocate_auto_diff_scalar;
			for (int i = 0; i < local_size; ++i)
				local_state(i) = allocate_auto_diff_scalar(i, values(i));
		}

		template <typename T>
		void displacement_gradient_at_quad(
			const MixedNonLinearAssemblerData &data,
			const Eigen::Matrix<T, Eigen::Dynamic, 1> &local_state,
			const int p,
			const int dim,
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &grad_u)
		{
			grad_u.resize(dim, dim);
			for (int k = 0; k < grad_u.size(); ++k)
				grad_u(k) = T(0);

			for (int i = 0; i < data.phi_vals.basis_values.size(); ++i)
			{
				const auto &bs = data.phi_vals.basis_values[i];
				const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
				assert(grad.size() == dim);

				for (int d = 0; d < dim; ++d)
				{
					for (int c = 0; c < dim; ++c)
						grad_u(d, c) += grad(c) * local_state(i * dim + d);
				}
			}

			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> jac_it(dim, dim);
			for (int k = 0; k < jac_it.size(); ++k)
				jac_it(k) = T(data.phi_vals.jac_it[p](k));
			grad_u = grad_u * jac_it;
		}

		template <typename T>
		T temperature_at_quad(
			const MixedNonLinearAssemblerData &data,
			const Eigen::Matrix<T, Eigen::Dynamic, 1> &local_state,
			const int p,
			const int dim)
		{
			const int phi_local_size = int(data.phi_vals.basis_values.size()) * dim;
			T temperature = T(0);
			for (int i = 0; i < data.psi_vals.basis_values.size(); ++i)
				temperature += data.psi_vals.basis_values[i].val(p) * local_state(phi_local_size + i);
			return temperature;
		}
	} // namespace

	void ThermoElasticity::add_multimaterial(const int index, const json &params, const Units &, const std::string &root_path)
	{
		for (int i = int(alpha_.size()); i <= index; ++i)
		{
			alpha_.emplace_back();
			alpha_.back().init(0.0);
			T0_.emplace_back();
			T0_.back().init(0.0);
		}

		if (params.contains("alpha"))
			alpha_[index].init(params["alpha"], root_path);
		if (params.contains("T0"))
			T0_[index].init(params["T0"], root_path);
	}

	std::map<std::string, Assembler::ParamFunc> ThermoElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["alpha"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return alpha(uv, p, t, e); };
		res["T0"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return T0(uv, p, t, e); };
		res["E"] = [this](const RowVectorNd &, const RowVectorNd &, double, int) { return young_; };
		res["nu"] = [this](const RowVectorNd &, const RowVectorNd &, double, int) { return nu_; };
		return res;
	}

	double ThermoElasticity::compute_energy(const MixedNonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	Eigen::VectorXd ThermoElasticity::compute_gradient(const MixedNonLinearAssemblerData &data) const
	{
		const auto energy = compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(data);
		return energy.getGradient();
	}

	Eigen::MatrixXd ThermoElasticity::compute_hessian(const MixedNonLinearAssemblerData &data) const
	{
		const auto energy = compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(data);
		return energy.getHessian();
	}

	template <typename T>
	T ThermoElasticity::compute_energy_aux(const MixedNonLinearAssemblerData &data) const
	{
		assert(size() == 2 || size() == 3);
		assert(cols() == 1);
		assert(data.phi_vals.basis_values.size() > 0);
		assert(data.psi_vals.basis_values.size() > 0);
		assert(data.phi_vals.quadrature.weights.size() == data.psi_vals.quadrature.weights.size());

		const int dim = size();
		const double lambda = this->lambda();
		const double mu = this->mu();

		Eigen::Matrix<T, Eigen::Dynamic, 1> local_state;
		get_local_state(data, dim, local_state);

		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u(dim, dim);
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> F(dim, dim);

		T energy = T(0);
		for (int p = 0; p < data.da.size(); ++p)
		{
			displacement_gradient_at_quad(data, local_state, p, dim, grad_u);

			F = grad_u;
			for (int d = 0; d < dim; ++d)
				F(d, d) += T(1);

			const T temperature = temperature_at_quad(data, local_state, p, dim);
			const double alpha = this->alpha(
				data.phi_vals.quadrature.points.row(p), data.phi_vals.val.row(p), data.t, data.phi_vals.element_id);
			const double T0 = this->T0(
				data.phi_vals.quadrature.points.row(p), data.phi_vals.val.row(p), data.t, data.phi_vals.element_id);

			using std::exp;
			const T theta = exp(T(alpha) * (temperature - T(T0)));
			const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> Fe = F / theta;
			energy += (elastic_energy(Fe, lambda, mu) - elastic_energy(F, lambda, mu)) * data.da(p);
		}

		return energy;
	}

	template <typename T>
	T ThermoElasticity::elastic_energy(
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &Fe,
		const double lambda,
		const double mu) const
	{
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> I(Fe.rows(), Fe.cols());
		for (int k = 0; k < I.size(); ++k)
			I(k) = T(0);
		for (int d = 0; d < Fe.rows(); ++d)
			I(d, d) = T(1);

		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> strain = (Fe + Fe.transpose()) / T(2) - I;
		return T(mu) * (strain.transpose() * strain).trace() + T(lambda / 2.0) * strain.trace() * strain.trace();
	}

	double ThermoElasticity::alpha(const RowVectorNd &, const RowVectorNd &p, const double t, const int element_id) const
	{
		return eval_param(alpha_, 0.0, p, t, element_id);
	}

	double ThermoElasticity::T0(const RowVectorNd &, const RowVectorNd &p, const double t, const int element_id) const
	{
		return eval_param(T0_, 0.0, p, t, element_id);
	}

	double ThermoElasticity::lambda() const
	{
		assert(size() == 2 || size() == 3);
		if (size() == 3)
			return (young_ * nu_) / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
		return (nu_ * young_) / (1.0 - nu_ * nu_);
	}

	double ThermoElasticity::mu() const
	{
		return young_ / (2.0 * (1.0 + nu_));
	}

	double ThermoElasticity::eval_param(
		const std::vector<utils::ExpressionValue> &params,
		const double default_value,
		const RowVectorNd &p,
		const double t,
		const int element_id)
	{
		if (params.empty())
			return default_value;

		const auto &param = params.size() == 1 ? params.front() : params.at(element_id);
		return param(p(0), p(1), p.size() == 3 ? p(2) : 0.0, t, element_id);
	}
} // namespace polyfem::assembler
