#include "Mass.hpp"

#include <utility>

#include <polyfem/assembler/Assemble.hpp>
#include <polyfem/assembler/ComputeSparsityPattern.hpp>

namespace polyfem::assembler
{
	namespace
	{

		template <int value_dim, int dim>
		struct MassMatrixKernel
		{
			using Material = material::Density<double>;
			static constexpr int VALUE_DIM = value_dim;
			static constexpr int DIM = dim;
			static constexpr bool SUPPORT_DEVICE_EVAL = true;

			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;
			POLYFEM_BOTH Mat eval_matrix(
				int element_id,
				int quad_id,
				int bi,
				int bj,
				const AssemblyDataView &data,
				const ElementAssemblyCacheView &cache,
				const Material &material,
				Span<const double> unknown) const
			{
				(void)element_id;
				(void)quad_id;
				(void)data;
				(void)unknown;

				double value = material.rho * cache.get_basis_value(bi, quad_id) * cache.get_basis_value(bj, quad_id);
				return value * Mat::Identity();
			}
		};

	} // namespace

	Mass::Mass()
		: density_(std::make_shared<Density>())
	{
	}

	Mass::Mass(std::shared_ptr<Density> density)
		: density_(std::move(density))
	{
		assert(density_);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> Mass::assemble(const LinearAssemblerData &data) const
	{
		double tmp = 0;

		// loop over quadrature points
		for (int q = 0; q < data.da.size(); ++q)
		{
			const double rho = density()(data.vals.quadrature.points.row(q), data.vals.val.row(q), data.t, data.vals.element_id);
			// phi_i * phi_j weighted by quadrature weights
			tmp += rho * data.vals.basis_values[data.i].val(q) * data.vals.basis_values[data.j].val(q) * data.da(q);
		}

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size(), 1);
		res.setZero();
		for (int i = 0; i < size(); ++i)
			res(i * size() + i) = tmp;

		return res;
	}

	std::optional<BSRSparsityPattern> Mass::hessian_sparsity_pattern_ng(
		const int n_basis,
		const AssemblyData &data) const
	{
		return compute_sparsity_pattern(data.view(), n_basis, size());
	}

	void Mass::assemble_hessian_ng(
		const bool is_volume,
		const int n_basis,
		const AssemblyData &data,
		const AssemblyData &geom_data,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &materials,
		Span<const double> x,
		Span<const double> x_prev,
		const double t,
		const double dt,
		BSRMatrix &hessian,
		const bool project_to_psd,
		const double scale,
		ExecutionPolicy policy) const
	{
		(void)n_basis;
		(void)x;
		(void)x_prev;
		(void)dt;
		(void)project_to_psd;

		assert(has_ng_assembly_support());
		assert(hessian.rows() == size() * n_basis);
		assert(hessian.cols() == size() * n_basis);

		// TODO: better way to get space dimension.
		int dim = data.view().element_desc.front().basis_desc.dim;
		switch (dim)
		{
		case 1:
			switch (size())
			{
			case 1:
				assemble_matrix(MassMatrixKernel<1, 1>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			case 2:
				assemble_matrix(MassMatrixKernel<2, 1>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			case 3:
				assemble_matrix(MassMatrixKernel<3, 1>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			default:
				log_and_throw_error("Unsupported NG mass value dimension {}.", size());
			}
			break;
		case 2:
			switch (size())
			{
			case 1:
				assemble_matrix(MassMatrixKernel<1, 2>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			case 2:
				assemble_matrix(MassMatrixKernel<2, 2>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			case 3:
				assemble_matrix(MassMatrixKernel<3, 2>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			default:
				log_and_throw_error("Unsupported NG mass value dimension {}.", size());
			}
			break;
		case 3:
			switch (size())
			{
			case 1:
				assemble_matrix(MassMatrixKernel<1, 3>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			case 2:
				assemble_matrix(MassMatrixKernel<2, 3>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			case 3:
				assemble_matrix(MassMatrixKernel<3, 3>{}, data, geom_data, cache, materials, {}, hessian, false, t, scale, true, policy);
				break;
			default:
				log_and_throw_error("Unsupported NG mass value dimension {}.", size());
			}
			break;
		default:
			log_and_throw_error("Unsupported NG mass geometric dimension {}.", dim);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> Mass::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(false);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> result;

		return result;
	}

	void Mass::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size_ == 1 || size_ == 2 || size_ == 3);

		if (auto thermal_density = std::dynamic_pointer_cast<ThermalMassDensity>(density_))
			thermal_density->add_multimaterial(index, params, units.density(), units.specific_heat_capacity(), root_path);
		else
			density_->add_multimaterial(index, params, units.density(), root_path);
	}

	std::map<std::string, Assembler::ParamFunc> Mass::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		if (auto thermal_density = std::dynamic_pointer_cast<ThermalMassDensity>(density_))
		{
			res["rho"] = [thermal_density](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return thermal_density->rho(p, t, e);
			};
			res["heat_capacity"] = [thermal_density](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
				return thermal_density->heat_capacity(p, t, e);
			};
			res["rho_heat_capacity"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
				return this->density()(uv, p, t, e);
			};
		}
		else
		{
			res["rho"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
				return this->density()(uv, p, t, e);
			};
		}

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> HRZMass::assemble(const LinearAssemblerData &data) const
	{
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> res(size() * size(), 1);
		res.setZero();

		if (data.i != data.j)
			return res;

		double sum_all_entries = 0;
		double sum_all_diag_entries = 0;
		double sum_target_diag_entries = 0;

		for (int i = 0; i < data.vals.basis_values.size(); ++i)
		{
			for (int j = 0; j < data.vals.basis_values.size(); ++j)
			{
				double entry = 0;
				for (int q = 0; q < data.da.size(); ++q)
				{
					entry += data.vals.basis_values[i].val(q) * data.vals.basis_values[j].val(q) * data.da(q);
				}
				sum_all_entries += entry;
				if (i == j)
				{
					sum_all_diag_entries += entry;
					if (i == data.i)
					{
						sum_target_diag_entries += entry;
					}
				}
			}
		}

		for (int i = 0; i < size(); ++i)
			res(i * size() + i) = sum_all_entries / sum_all_diag_entries * sum_target_diag_entries;

		return res;
	}

} // namespace polyfem::assembler
