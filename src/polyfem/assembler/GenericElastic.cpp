#include "GenericElastic.hpp"

#include "MooneyRivlinElasticity.hpp"
#include "OgdenElasticity.hpp"
#include "NeoHookeanElasticityAutodiff.hpp"

#include <polyfem/basis/Basis.hpp>

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	template <typename ElasticFormulation>
	GenericElastic<ElasticFormulation>::GenericElastic()
	{
	}

	template <typename ElasticFormulation>
	void GenericElastic<ElasticFormulation>::set_size(const int size)
	{
		size_ = size;
	}

	template <typename ElasticFormulation>
	void GenericElastic<ElasticFormulation>::add_multimaterial(const int index, const json &params)
	{
		assert(size_ == 2 || size_ == 3);

		formulation_.add_multimaterial(index, params, size());
	}

	template <typename ElasticFormulation>
	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	GenericElastic<ElasticFormulation>::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());

		log_and_throw_error("Fabricated solution not supported!");

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;
		return res;
	}

	template <typename ElasticFormulation>
	void GenericElastic<ElasticFormulation>::compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, size() * size(), stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::MatrixXd tmp = stress;
			auto a = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, size() * size());
			return Eigen::MatrixXd(a);
		});
	}

	template <typename ElasticFormulation>
	void GenericElastic<ElasticFormulation>::compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const
	{
		assign_stress_tensor(el_id, bs, gbs, local_pts, displacement, 1, stresses, [&](const Eigen::MatrixXd &stress) {
			Eigen::Matrix<double, 1, 1> res;
			res.setConstant(von_mises_stress_for_stress_tensor(stress));
			return res;
		});
	}

	template <typename ElasticFormulation>
	void GenericElastic<ElasticFormulation>::assign_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		Eigen::MatrixXd displacement_grad(size(), size());
		Eigen::MatrixXd stress_tensor(size(), size());

		typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>> Diff;

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);
		DiffScalarBase::setVariableCount(displacement_grad.size());

		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> disp_grad(size(), size());

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			for (int d1 = 0; d1 < size(); ++d1)
			{
				for (int d2 = 0; d2 < size(); ++d2)
					disp_grad(d1, d2) = Diff(d1 * size() + d2, displacement_grad(d1, d2));
			}

			const auto val = formulation_.elastic_energy(size(), local_pts.row(p), vals.element_id, disp_grad);

			for (int d1 = 0; d1 < size(); ++d1)
			{
				for (int d2 = 0; d2 < size(); ++d2)
					stress_tensor(d1, d2) = val.getGradient()(d1 * size() + d2);
			}

			all.row(p) = fun(stress_tensor);
		}
	}

	template <typename ElasticFormulation>
	Eigen::VectorXd GenericElastic<ElasticFormulation>::assemble_grad(const NonLinearAssemblerData &data) const
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

	template <typename ElasticFormulation>
	Eigen::MatrixXd GenericElastic<ElasticFormulation>::assemble_hessian(const NonLinearAssemblerData &data) const
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

	template <typename ElasticFormulation>
	double GenericElastic<ElasticFormulation>::compute_energy(const NonLinearAssemblerData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	template class GenericElastic<MooneyRivlinElasticity>;
	template class GenericElastic<OgdenElasticity>;
	// for testing
	template class GenericElastic<NeoHookeanAutodiff>;
} // namespace polyfem::assembler