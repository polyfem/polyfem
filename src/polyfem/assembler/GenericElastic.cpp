#include "GenericElastic.hpp"

#include <polyfem/assembler/MooneyRivlinElasticity.hpp>
#include <polyfem/assembler/OgdenElasticity.hpp>
#include <polyfem/assembler/NeoHookeanElasticityAutodiff.hpp>
#include <polyfem/assembler/AMIPSEnergy.hpp>

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	template <typename Derived>
	GenericElastic<Derived>::GenericElastic()
	{
	}

	template <typename Derived>
	void GenericElastic<Derived>::assign_stress_tensor(
		const int el_id,
		const basis::ElementBases &bs,
		const basis::ElementBases &gbs,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &displacement,
		const int all_size,
		const ElasticityTensorType &type,
		Eigen::MatrixXd &all,
		const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
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

			const auto val = derived().elastic_energy(local_pts.row(p), vals.element_id, def_grad);

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

	template <typename Derived>
	double GenericElastic<Derived>::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	template <typename Derived>
	Eigen::VectorXd GenericElastic<Derived>::assemble_gradient(const NonLinearAssemblerData &data) const
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

	template <typename Derived>
	Eigen::MatrixXd GenericElastic<Derived>::assemble_hessian(const NonLinearAssemblerData &data) const
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

	template <typename Derived>
	void GenericElastic<Derived>::compute_stress_grad_multiply_mat(
		const int el_id,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &global_pts,
		const Eigen::MatrixXd &grad_u_i,
		const Eigen::MatrixXd &mat,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9>> Diff;

		DiffScalarBase::setVariableCount(size() * size());
		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < size(); ++d)
			F(d, d) += 1.;

		assert(local_pts.rows() == 1);
		for (int i = 0; i < size(); ++i)
			for (int j = 0; j < size(); ++j)
				def_grad(i, j) = Diff(i + j * size(), F(i, j));

		auto energy = derived().elastic_energy(global_pts, el_id, def_grad);

		// Grad is ∂W(F)/∂F_ij
		Eigen::MatrixXd grad = energy.getGradient().reshaped(size(), size());
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		Eigen::MatrixXd hess = energy.getHessian();

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		// Compute ∂S_ij/∂F_kl * M_kl, same as M_ij * ∂S_ij/∂F_kl since the hessian is symmetric
		result = (hess * mat.reshaped(size() * size(), 1)).reshaped(size(), size());
	}

	template <typename Derived>
	void GenericElastic<Derived>::compute_stress_grad_multiply_stress(
		const int el_id,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &global_pts,
		const Eigen::MatrixXd &grad_u_i,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9>> Diff;

		DiffScalarBase::setVariableCount(size() * size());
		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < size(); ++d)
			F(d, d) += 1.;

		assert(local_pts.rows() == 1);
		for (int i = 0; i < size(); ++i)
			for (int j = 0; j < size(); ++j)
				def_grad(i, j) = Diff(i + j * size(), F(i, j));

		auto energy = derived().elastic_energy(global_pts, el_id, def_grad);

		// Grad is ∂W(F)/∂F_ij
		Eigen::MatrixXd grad = energy.getGradient().reshaped(size(), size());
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		Eigen::MatrixXd hess = energy.getHessian();

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		// Compute ∂S_ij/∂F_kl * S_kl, same as S_ij * ∂S_ij/∂F_kl since the hessian is symmetric
		result = (hess * stress.reshaped(size() * size(), 1)).reshaped(size(), size());
	}

	template <typename Derived>
	void GenericElastic<Derived>::compute_stress_grad_multiply_vect(
		const int el_id,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &global_pts,
		const Eigen::MatrixXd &grad_u_i,
		const Eigen::MatrixXd &vect,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9>> Diff;

		DiffScalarBase::setVariableCount(size() * size());
		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < size(); ++d)
			F(d, d) += 1.;

		assert(local_pts.rows() == 1);
		for (int i = 0; i < size(); ++i)
			for (int j = 0; j < size(); ++j)
				def_grad(i, j) = Diff(i + j * size(), F(i, j));

		auto energy = derived().elastic_energy(global_pts, el_id, def_grad);

		// Grad is ∂W(F)/∂F_ij
		Eigen::MatrixXd grad = energy.getGradient().reshaped(size(), size());
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		Eigen::MatrixXd hess = energy.getHessian();

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		result.resize(hess.rows(), vect.size());
		for (int i = 0; i < hess.rows(); ++i)
			if (vect.rows() == 1)
				// Compute ∂S_ij/∂F_kl * v_k, same as ∂S_ij/∂F_kl * v_i since the hessian is symmetric
				result.row(i) = vect * hess.row(i).reshaped(size(), size());
			else
				// Compute ∂S_ij/∂F_kl * v_l, same as ∂S_ij/∂F_kl * v_j since the hessian is symmetric
				result.row(i) = hess.row(i).reshaped(size(), size()) * vect;
	}
	template class GenericElastic<MooneyRivlinElasticity>;
	template class GenericElastic<AMIPSEnergy>;
	template class GenericElastic<UnconstrainedOgdenElasticity>;
	template class GenericElastic<IncompressibleOgdenElasticity>;
	template class GenericElastic<NeoHookeanAutodiff>;
} // namespace polyfem::assembler