#include "Mass.hpp"

#include <utility>

namespace polyfem::assembler
{
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
