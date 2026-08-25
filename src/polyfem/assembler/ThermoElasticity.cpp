#include "ThermoElasticity.hpp"

#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Jacobian.hpp>

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

	namespace detail
	{
		class ThermoElasticityModel
		{
		public:
			virtual ~ThermoElasticityModel() = default;

			virtual std::string elastic_name() const = 0;
			virtual std::map<std::string, Assembler::ParamFunc> parameters() const = 0;
			virtual void set_size(const int size) = 0;
			virtual void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) = 0;

			virtual double compute_energy(const MixedNonLinearAssemblerData &data) const = 0;
			virtual Eigen::VectorXd compute_gradient(const MixedNonLinearAssemblerData &data) const = 0;
			virtual Eigen::MatrixXd compute_hessian(const MixedNonLinearAssemblerData &data) const = 0;
		};

		template <typename Elasticity>
		class ThermoElasticityModelImpl : public ThermoElasticityModel
		{
		public:
			ThermoElasticityModelImpl();

			std::string elastic_name() const override { return elastic_.name(); }
			std::map<std::string, Assembler::ParamFunc> parameters() const override;

			void set_size(const int size) override;
			void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

			double compute_energy(const MixedNonLinearAssemblerData &data) const override;
			Eigen::VectorXd compute_gradient(const MixedNonLinearAssemblerData &data) const override;
			Eigen::MatrixXd compute_hessian(const MixedNonLinearAssemblerData &data) const override;

		private:
			int size() const { return size_; }
			int cols() const { return 1; }

			template <typename T>
			T compute_energy_aux(const MixedNonLinearAssemblerData &data) const;

			double alpha(const RowVectorNd &uv, const RowVectorNd &p, const double t, const int element_id) const;
			double T0(const RowVectorNd &uv, const RowVectorNd &p, const double t, const int element_id) const;

			int size_ = -1;
			Elasticity elastic_;
			GenericMatParam alpha_;
			GenericMatParam T0_;
		};

		std::unique_ptr<ThermoElasticityModel> make_thermo_elasticity_model(const std::string &elastic_formulation)
		{
			if (elastic_formulation == "NeoHookean")
				return std::make_unique<ThermoElasticityModelImpl<NeoHookeanElasticity>>();

			log_and_throw_error("ThermoElasticity currently supports only NeoHookean elastic_material, got '{}'.", elastic_formulation);
		}
	} // namespace detail

	template <typename Elasticity>
	detail::ThermoElasticityModelImpl<Elasticity>::ThermoElasticityModelImpl()
		: alpha_("alpha"), T0_("T0")
	{
	}

	template <typename Elasticity>
	void detail::ThermoElasticityModelImpl<Elasticity>::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		alpha_.add_multimaterial(index, params, units.one_over_temperature(), root_path);
		T0_.add_multimaterial(index, params, units.temperature(), root_path);

		if (!params.contains("elastic_material") || !params["elastic_material"].is_object())
			log_and_throw_error("ThermoElasticity requires elastic_material to be an elastic material object.");

		json elastic_params = params["elastic_material"];
		const std::string type = elastic_params.value("type", "");
		if (type != elastic_.name())
			log_and_throw_error("ThermoElasticity<{}> requires elastic_material '{}', got '{}'.", elastic_.name(), elastic_.name(), type);

		if (params.contains("id"))
			elastic_params["id"] = params["id"];
		if (params.contains("rho"))
			elastic_params["rho"] = params["rho"];

		elastic_.add_multimaterial(index, elastic_params, units, root_path);
	}

	template <typename Elasticity>
	void detail::ThermoElasticityModelImpl<Elasticity>::set_size(const int size)
	{
		size_ = size;
		elastic_.set_size(size);
	}

	template <typename Elasticity>
	std::map<std::string, Assembler::ParamFunc> detail::ThermoElasticityModelImpl<Elasticity>::parameters() const
	{
		std::map<std::string, Assembler::ParamFunc> res;
		res["alpha"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return alpha(uv, p, t, e); };
		res["T0"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return T0(uv, p, t, e); };
		return res;
	}

	template <typename Elasticity>
	double detail::ThermoElasticityModelImpl<Elasticity>::compute_energy(const MixedNonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	template <typename Elasticity>
	Eigen::VectorXd detail::ThermoElasticityModelImpl<Elasticity>::compute_gradient(const MixedNonLinearAssemblerData &data) const
	{
		const auto energy = compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(data);
		return energy.getGradient();
	}

	template <typename Elasticity>
	Eigen::MatrixXd detail::ThermoElasticityModelImpl<Elasticity>::compute_hessian(const MixedNonLinearAssemblerData &data) const
	{
		const auto energy = compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(data);
		return energy.getHessian();
	}

	template <typename Elasticity>
	template <typename T>
	T detail::ThermoElasticityModelImpl<Elasticity>::compute_energy_aux(const MixedNonLinearAssemblerData &data) const
	{
		assert(size() == 2 || size() == 3);
		assert(cols() == 1);
		assert(data.phi_vals.basis_values.size() > 0);
		assert(data.psi_vals.basis_values.size() > 0);
		assert(data.phi_vals.quadrature.weights.size() == data.psi_vals.quadrature.weights.size());

		const int dim = size();

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
			energy += (elastic_.elastic_energy_density(
						   data.phi_vals.quadrature.points.row(p), data.phi_vals.val.row(p), data.t, data.phi_vals.element_id, Fe)
					   - elastic_.elastic_energy_density(
						   data.phi_vals.quadrature.points.row(p), data.phi_vals.val.row(p), data.t, data.phi_vals.element_id, F))
					  * data.da(p);
		}

		return energy;
	}

	template <typename Elasticity>
	double detail::ThermoElasticityModelImpl<Elasticity>::alpha(const RowVectorNd &, const RowVectorNd &p, const double t, const int element_id) const
	{
		return alpha_(p, t, element_id);
	}

	template <typename Elasticity>
	double detail::ThermoElasticityModelImpl<Elasticity>::T0(const RowVectorNd &, const RowVectorNd &p, const double t, const int element_id) const
	{
		return T0_(p, t, element_id);
	}

	ThermoElasticity::ThermoElasticity() = default;

	ThermoElasticity::~ThermoElasticity() = default;

	std::map<std::string, Assembler::ParamFunc> ThermoElasticity::parameters() const
	{
		if (!model_)
			return {};

		return model_->parameters();
	}

	void ThermoElasticity::set_size(const int size)
	{
		Assembler::set_size(size);
		if (model_)
			model_->set_size(size);
	}

	void ThermoElasticity::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		if (!params.contains("elastic_material") || !params["elastic_material"].is_object())
			log_and_throw_error("ThermoElasticity requires elastic_material to be an elastic material object.");

		const std::string type = params["elastic_material"].value("type", "");
		if (!model_)
		{
			model_ = detail::make_thermo_elasticity_model(type);
			elastic_formulation_ = type;
			model_->set_size(size());
		}
		else if (type != elastic_formulation_)
		{
			log_and_throw_error(
				"ThermoElasticity requires all elastic_material entries to have the same type, got '{}' and '{}'.",
				elastic_formulation_, type);
		}

		model_->add_multimaterial(index, params, units, root_path);
	}

	double ThermoElasticity::compute_energy(const MixedNonLinearAssemblerData &data) const
	{
		return model().compute_energy(data);
	}

	Eigen::VectorXd ThermoElasticity::compute_gradient(const MixedNonLinearAssemblerData &data) const
	{
		return model().compute_gradient(data);
	}

	Eigen::MatrixXd ThermoElasticity::compute_hessian(const MixedNonLinearAssemblerData &data) const
	{
		return model().compute_hessian(data);
	}

	detail::ThermoElasticityModel &ThermoElasticity::model()
	{
		if (!model_)
			log_and_throw_error("ThermoElasticity material model was used before materials were initialized.");

		return *model_;
	}

	const detail::ThermoElasticityModel &ThermoElasticity::model() const
	{
		if (!model_)
			log_and_throw_error("ThermoElasticity material model was used before materials were initialized.");

		return *model_;
	}

	template class detail::ThermoElasticityModelImpl<NeoHookeanElasticity>;

} // namespace polyfem::assembler
