#include "GenericElastic.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	GenericElastic::GenericElastic()
	{
	}

	void GenericElastic::assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, const ElasticityTensorType &type, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		Eigen::MatrixXd deformation_grad(size(), size());
		Eigen::MatrixXd stress_tensor(size(), size());

		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>> Diff;

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);
		DiffScalarBase::setVariableCount(deformation_grad.size());

		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, deformation_grad);

			// Id + grad d
			for (int d = 0; d < size(); ++d)
				deformation_grad(d, d) += 1;

			if (type == ElasticityTensorType::F)
			{
				all.row(p) = fun(deformation_grad);
				continue;
			}

			for (int d1 = 0; d1 < size(); ++d1)
			{
				for (int d2 = 0; d2 < size(); ++d2)
					def_grad(d1, d2) = Diff(d1 * size() + d2, deformation_grad(d1, d2));
			}

			const auto val = elastic_energy(local_pts.row(p), vals.element_id, def_grad);

			for (int d1 = 0; d1 < size(); ++d1)
			{
				for (int d2 = 0; d2 < size(); ++d2)
					stress_tensor(d1, d2) = val.getGradient()(d1 * size() + d2);
			}

			stress_tensor = 1.0 / deformation_grad.determinant() * stress_tensor * deformation_grad.transpose();

			if (type == ElasticityTensorType::PK1)
				stress_tensor = pk1_from_cauchy(stress_tensor, deformation_grad);
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = pk2_from_cauchy(stress_tensor, deformation_grad);

			all.row(p) = fun(stress_tensor);
		}
	}

	double GenericElastic::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	Eigen::VectorXd GenericElastic::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		const int n_bases = data.vals.basis_values.size();
		return polyfem::gradient_from_energy(
			size(), n_bases, data,
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 60, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(data); });
	}

	Eigen::MatrixXd GenericElastic::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const int n_bases = data.vals.basis_values.size();
		return polyfem::hessian_from_energy(
			size(), n_bases, data,
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>>(data); },
			[&](const NonLinearAssemblerData &data) { return compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(data); });
	}
} // namespace polyfem::assembler