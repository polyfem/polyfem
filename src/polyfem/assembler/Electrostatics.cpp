#include "Electrostatics.hpp"

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}
	} // namespace

	void Electrostatics::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		epsilon_.add_multimaterial(index, params, units.permittivity());
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> Electrostatics::assemble(const LinearAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.vals.basis_values[data.j].grad_t_m;
		// return ((gradi.array() * gradj.array()).rowwise().sum().array() * da.array()).colwise().sum();

		double res = 0;
		assert(gradi.rows() == data.da.size());
		for (int k = 0; k < gradi.rows(); ++k)
		{
			double epsilon = epsilon_(data.vals.val.row(k), data.t, data.vals.element_id);
			// compute grad(phi_i) dot grad(phi_j) weighted by quadrature weights
			res += epsilon * gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	void Electrostatics::compute_stress_grad_multiply_mat(const OptAssemblerData &data,
														  const Eigen::MatrixXd &mat,
														  Eigen::MatrixXd &stress,
														  Eigen::MatrixXd &result) const
	{
		stress = data.grad_u_i;
		result = mat;
	}

	void Electrostatics::compute_stiffness_value(const double t,
												 const assembler::ElementAssemblyValues &vals,
												 const Eigen::MatrixXd &local_pts,
												 const Eigen::MatrixXd &displacement,
												 Eigen::MatrixXd &tensor) const
	{
		const int dim = local_pts.cols();
		tensor.resize(local_pts.rows(), dim * dim);
		assert(displacement.cols() == 1);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			for (int i = 0, idx = 0; i < dim; i++)
				for (int j = 0; j < dim; j++)
				{
					tensor(p, idx) = delta(i, j);
					idx++;
				}
		}
	}

	std::map<std::string, Assembler::ParamFunc> Electrostatics::parameters() const
	{
		std::map<std::string, ParamFunc> res;

		res["epsilon"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return epsilon_(p, t, e);
		};

		return res;
	}

	double Electrostatics::compute_stored_energy(
		const bool is_volume,
		const int n_basis,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double t,
		const Eigen::MatrixXd &solution)
	{
		StiffnessMatrix K;
		assemble(is_volume, n_basis, bases, gbases, cache, t, K);

		Eigen::MatrixXd energy = 0.5 * solution.transpose() * (K * solution);
		assert(energy.size() == 1);
		return energy(0);
	}

} // namespace polyfem::assembler