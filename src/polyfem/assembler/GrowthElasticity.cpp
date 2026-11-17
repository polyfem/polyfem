#include "GrowthElasticity.hpp"

#include <polyfem/assembler/HGODispersion.hpp>
#include <polyfem/assembler/IsochoricNeoHookean.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/VolumePenalty.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <cmath>
#include <fstream>
#include <sstream>

namespace polyfem::assembler
{
	namespace
	{
		// ------------------------------------------------------------------
		// Local-state helpers: identical to ThermoElasticity's (same
		// MixedNonLinearAssemblerData layout: phi = displacement,
		// psi = scalar field; here the scalar field is the growth theta).
		// ------------------------------------------------------------------
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
		T growth_at_quad(
			const MixedNonLinearAssemblerData &data,
			const Eigen::Matrix<T, Eigen::Dynamic, 1> &local_state,
			const int p,
			const int dim)
		{
			const int phi_local_size = int(data.phi_vals.basis_values.size()) * dim;
			T theta = T(0);
			for (int i = 0; i < data.psi_vals.basis_values.size(); ++i)
				theta += data.psi_vals.basis_values[i].val(p) * local_state(phi_local_size + i);
			return theta;
		}

		inline double value_of(const double x) { return x; }
		template <typename T>
		double value_of(const T &x) { return x.getValue(); }

		// ------------------------------------------------------------------
		// Per-element vector/scalar cell data.
		//
		// Minimal reader for the legacy ASCII VTK written by the fiber/normal
		// pipeline (add_wall_normals.py): CELL_DATA section, arrays either as
		// "VECTORS <name> ..." or inside "FIELD FieldData" as
		// "<name> <ncomp> <ntuples> <dtype>".
		//
		// TODO(consolidation): route through the same loader GenericFiber
		// uses for fiber_direction, so growth normals and fibers share one
		// code path.
		// ------------------------------------------------------------------
		Eigen::MatrixXd read_vtk_cell_array(
			const std::string &path, const std::string &field, const int ncomp_expected)
		{
			std::ifstream in(path);
			if (!in)
				log_and_throw_error("GrowthElasticity: cannot open '{}'", path);

			std::string line;
			bool in_cell_data = false;
			long n_cells = -1;

			while (std::getline(in, line))
			{
				std::istringstream ls(line);
				std::string tok;
				ls >> tok;

				if (tok == "CELL_DATA")
				{
					in_cell_data = true;
					ls >> n_cells;
					continue;
				}
				if (tok == "POINT_DATA")
				{
					in_cell_data = false;
					continue;
				}
				if (!in_cell_data)
					continue;

				long ntuples = n_cells;
				int ncomp = -1;
				std::string name;

				if (tok == "VECTORS")
				{
					ls >> name;
					ncomp = 3;
				}
				else if (tok == "FIELD")
				{
					continue; // array headers follow on their own lines
				}
				else if (tok == "SCALARS")
				{
					ls >> name;
					ncomp = 1;
					std::getline(in, line); // LOOKUP_TABLE line
				}
				else
				{
					// possible FIELD array header: <name> <ncomp> <ntuples> <dtype>
					name = tok;
					if (!(ls >> ncomp >> ntuples))
						continue;
				}

				if (name != field)
				{
					// skip this array's payload
					long remaining = ntuples * std::max(ncomp, 1);
					double dummy;
					while (remaining > 0 && in >> dummy)
						--remaining;
					continue;
				}

				if (ncomp != ncomp_expected)
					log_and_throw_error(
						"GrowthElasticity: field '{}' in '{}' has {} components, expected {}",
						field, path, ncomp, ncomp_expected);

				Eigen::MatrixXd values(ntuples, ncomp);
				for (long i = 0; i < ntuples; ++i)
					for (int c = 0; c < ncomp; ++c)
						if (!(in >> values(i, c)))
							log_and_throw_error(
								"GrowthElasticity: truncated field '{}' in '{}'", field, path);
				return values;
			}

			log_and_throw_error(
				"GrowthElasticity: cell field '{}' not found in '{}'", field, path);
			return Eigen::MatrixXd(); // unreachable
		}
	} // namespace

	namespace detail
	{
		class GrowthElasticityModel
		{
		public:
			void set_size(const int size)
			{
				size_ = size;
				for (auto &m : iso_)
					m->set_size(size);
				for (auto &m : fiber_)
					m->set_size(size);
				for (auto &m : vol_)
					m->set_size(size);
			}

			void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
			{
				// ---- elastic material: the fitted MaterialSum composition
				if (!params.contains("elastic_material") || !params["elastic_material"].is_object())
					log_and_throw_error("GrowthElasticity requires elastic_material to be an elastic material object.");

				const json &em = params["elastic_material"];
				if (em.value("type", "") != "MaterialSum")
					log_and_throw_error(
						"GrowthElasticity requires elastic_material of type 'MaterialSum', got '{}'.",
						em.value("type", ""));

				for (const json &child_in : em.at("models"))
				{
					json child = child_in;
					// forward parent identifiers, exactly as ThermoElasticity does
					if (params.contains("id"))
						child["id"] = params["id"];
					if (params.contains(MATERIAL_ELEMENT_INDEX))
						child[MATERIAL_ELEMENT_INDEX] = params[MATERIAL_ELEMENT_INDEX];

					const std::string type = child.value("type", "");
					if (type == "IsochoricNeoHookean")
						add_child(iso_, index, child, units, root_path);
					else if (type == "HGODispersion")
						add_child(fiber_, index, child, units, root_path);
					else if (type == "VolumePenalty")
						add_child(vol_, index, child, units, root_path);
					else
						log_and_throw_error(
							"GrowthElasticity elastic_material supports IsochoricNeoHookean, "
							"HGODispersion, VolumePenalty; got '{}'.",
							type);
				}

				// ---- reference wall normal n0 (required): per-element data or constant
				if (!params.contains("normal_direction"))
					log_and_throw_error("GrowthElasticity requires normal_direction.");
				const json &nd = params["normal_direction"];
				if (nd.is_array())
				{
					constant_n0_ = Eigen::Vector3d(
									   nd[0].get<double>(), nd[1].get<double>(), nd[2].get<double>())
									   .normalized();
				}
				else if (nd.is_object() && nd.value("type", "") == "per_element_file")
				{
					normals_ = read_vtk_cell_array(
						utils::resolve_path(nd.at("path").get<std::string>(), root_path),
						nd.value("field", "N0"), 3);
					normals_.rowwise().normalize();
				}
				else
				{
					log_and_throw_error(
						"GrowthElasticity normal_direction must be a constant [x,y,z] "
						"or {type: per_element_file, path, field}.");
				}

				// ---- prescribed normal growth vn (optional; default 1 = pure
				// in-plane row; the candidate amended map supplies it per element)
				if (params.contains("normal_growth"))
				{
					const json &ng = params["normal_growth"];
					if (ng.is_number())
						constant_vn_ = ng.get<double>();
					else if (ng.is_object() && ng.value("type", "") == "per_element_file")
						vn_ = read_vtk_cell_array(
							utils::resolve_path(ng.at("path").get<std::string>(), root_path),
							ng.value("field", "VN"), 1);
					else
						log_and_throw_error(
							"GrowthElasticity normal_growth must be a number "
							"or {type: per_element_file, path, field}.");
				}
			}

			std::map<std::string, Assembler::ParamFunc> parameters() const
			{
				std::map<std::string, Assembler::ParamFunc> res;
				res["normal_growth"] = [this](const RowVectorNd &, const RowVectorNd &, double, int e) {
					return vn(e);
				};
				return res;
			}

			template <typename T>
			T compute_energy_aux(const MixedNonLinearAssemblerData &data) const
			{
				if (size_ != 3)
					log_and_throw_error("GrowthElasticity supports 3D only (in-plane growth needs a wall normal).");
				assert(data.phi_vals.basis_values.size() > 0);
				assert(data.psi_vals.basis_values.size() > 0);
				assert(data.phi_vals.quadrature.weights.size() == data.psi_vals.quadrature.weights.size());

				const int dim = size_;
				const int el_id = data.phi_vals.element_id;

				Eigen::Matrix<T, Eigen::Dynamic, 1> local_state;
				get_local_state(data, dim, local_state);

				const Eigen::Vector3d n0v = n0(el_id);
				const double vn_e = vn(el_id);
				if (!(vn_e > 0))
					log_and_throw_error("GrowthElasticity: normal_growth must be positive, got {} on element {}.", vn_e, el_id);

				Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad_u(dim, dim);
				Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> F(dim, dim), Fgi(dim, dim), Fe(dim, dim);

				T energy = T(0);
				for (int p = 0; p < data.da.size(); ++p)
				{
					displacement_gradient_at_quad(data, local_state, p, dim, grad_u);

					F = grad_u;
					for (int d = 0; d < dim; ++d)
						F(d, d) += T(1);

					const T theta = growth_at_quad(data, local_state, p, dim);
					if (!(value_of(theta) > 0))
						log_and_throw_error(
							"GrowthElasticity: growth theta must stay positive; got {} on element {} "
							"(the growth-block feasibility filter is responsible for keeping iterates admissible).",
							value_of(theta), el_id);

					// Fg^-1 = I + (theta^{-1/2} - 1)(I - n0 x n0) + (vn^{-1} - 1) n0 x n0.
					// Identity-anchored form: at theta == 1, vn == 1 both coefficients
					// are exactly zero, so Fgi is the BITWISE identity for any normal
					// and the correction vanishes exactly (the V1 regression identity).
					using std::sqrt;
					const T c_ip = T(1) / sqrt(theta) - T(1);
					const double c_n = 1.0 / vn_e - 1.0;
					for (int a = 0; a < dim; ++a)
						for (int b = 0; b < dim; ++b)
						{
							const double id = (a == b) ? 1.0 : 0.0;
							const double nn = n0v(a) * n0v(b);
							Fgi(a, b) = T(id) + c_ip * T(id - nn) + T(c_n * nn);
						}

					Fe = F * Fgi;
					const T Jg = theta * T(vn_e);

					// per-intermediate-volume convention (Q5): J^g * psi_e(Fe),
					// minus psi_e(F) so this form is the additive correction and
					// vanishes identically at theta = 1, vn = 1.
					const RowVectorNd pos = data.phi_vals.val.row(p);
					energy += (Jg * sum_energy<T>(pos, data.t, el_id, Fe)
							   - sum_energy<T>(pos, data.t, el_id, F))
							  * data.da(p);
				}

				return energy;
			}

		private:
			template <typename Child>
			void add_child(
				std::vector<std::unique_ptr<Child>> &list,
				const int index, const json &child, const Units &units, const std::string &root_path)
			{
				if (list.empty())
				{
					list.emplace_back(std::make_unique<Child>());
					if (size_ > 0) // robust to set_size/add_multimaterial call order
						list.front()->set_size(size_);
				}
				// multimaterial entries accumulate on the same child instances,
				// mirroring how SumModel forwards add_multimaterial per index
				list.front()->add_multimaterial(index, child, units, root_path);
			}

			template <typename T>
			T sum_energy(const RowVectorNd &p, const double t, const int el_id,
						 const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
			{
				T e = T(0);
				for (const auto &m : iso_)
					e += m->elastic_energy(p, t, el_id, F);
				for (const auto &m : fiber_)
					e += m->elastic_energy(p, t, el_id, F);
				for (const auto &m : vol_)
					e += m->elastic_energy(p, t, el_id, F);
				return e;
			}

			Eigen::Vector3d n0(const int el_id) const
			{
				if (normals_.rows() > 0)
				{
					if (el_id < 0 || el_id >= normals_.rows())
						log_and_throw_error(
							"GrowthElasticity: element {} out of range of normal_direction data ({} rows).",
							el_id, normals_.rows());
					return normals_.row(el_id).transpose();
				}
				return constant_n0_;
			}

			double vn(const int el_id) const
			{
				if (vn_.rows() > 0)
				{
					if (el_id < 0 || el_id >= vn_.rows())
						log_and_throw_error(
							"GrowthElasticity: element {} out of range of normal_growth data ({} rows).",
							el_id, vn_.rows());
					return vn_(el_id, 0);
				}
				return constant_vn_;
			}

			int size_ = -1;

			// concrete children (the fitted MaterialSum re-composed with the
			// SAME templated elastic_energy bodies the plain elastic path runs)
			std::vector<std::unique_ptr<IsochoricNeoHookean>> iso_;
			std::vector<std::unique_ptr<HGODispersion>> fiber_;
			std::vector<std::unique_ptr<VolumePenalty>> vol_;

			Eigen::MatrixXd normals_;                    // per-element n0, or:
			Eigen::Vector3d constant_n0_{0.0, 0.0, 1.0}; // constant fallback (tests)
			Eigen::MatrixXd vn_;                         // per-element vn, or:
			double constant_vn_ = 1.0;                   // default: pure in-plane row
		};
	} // namespace detail

	GrowthElasticity::GrowthElasticity()
		: model_(std::make_unique<detail::GrowthElasticityModel>())
	{
	}

	GrowthElasticity::~GrowthElasticity() = default;

	std::map<std::string, Assembler::ParamFunc> GrowthElasticity::parameters() const
	{
		return model().parameters();
	}

	void GrowthElasticity::set_size(const int size)
	{
		Assembler::set_size(size);
		model().set_size(size);
	}

	void GrowthElasticity::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		model().add_multimaterial(index, params, units, root_path);
	}

	double GrowthElasticity::compute_energy(const MixedNonLinearAssemblerData &data) const
	{
		return model().compute_energy_aux<double>(data);
	}

	Eigen::VectorXd GrowthElasticity::compute_gradient(const MixedNonLinearAssemblerData &data) const
	{
		const auto energy = model().compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(data);
		return energy.getGradient();
	}

	Eigen::MatrixXd GrowthElasticity::compute_hessian(const MixedNonLinearAssemblerData &data) const
	{
		const auto energy = model().compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(data);
		return energy.getHessian();
	}

	detail::GrowthElasticityModel &GrowthElasticity::model()
	{
		if (!model_)
			log_and_throw_error("GrowthElasticity used before initialization.");
		return *model_;
	}

	const detail::GrowthElasticityModel &GrowthElasticity::model() const
	{
		if (!model_)
			log_and_throw_error("GrowthElasticity used before initialization.");
		return *model_;
	}
} // namespace polyfem::assembler
