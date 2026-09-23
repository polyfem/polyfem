#pragma once

#include <polyfem/assembler/Assembler.hpp>

#include <memory>

namespace polyfem::assembler
{
	namespace detail
	{
		class GrowthElasticityModel;
	}

	/// Mixed (displacement, growth) assembler implementing the growth
	/// correction energy of the uterine growth formulation:
	///
	///     W = J^g * psi_e(Fe) - psi_e(F),
	///     Fe  = F * inv(Fg),
	///     Fg  = sqrt(theta) * (I - n0 x n0) + vn * n0 x n0,
	///     J^g = theta * vn,
	///
	/// with theta the scalar growth field (the psi-space DOF, interpolated at
	/// quadrature points), n0 the reference wall normal (per-element data),
	/// and vn the prescribed normal growth component (default 1: the pure
	/// in-plane row; the candidate amended map supplies vn per element).
	///
	/// The subtraction of psi_e(F) makes this an additive correction: the
	/// standard elastic form on the displacement block stays registered
	/// unchanged, and at theta == 1, vn == 1 the correction is identically
	/// zero (bitwise: both energy calls see the same def_grad), which is the
	/// V1 regression identity.
	///
	/// psi_e is the fitted MaterialSum composition, re-composed here from
	/// concrete child materials (see GrowthElasticity.cpp) because SumModel's
	/// type-erased children expose no templated energy; the children's
	/// templated elastic_energy bodies are reused verbatim.
	///
	/// Follows the structure of ThermoElasticity (the in-tree precedent for a
	/// displacement/scalar mixed correction assembler).
	class GrowthElasticity : public MixedNLAssembler
	{
	public:
		GrowthElasticity();
		~GrowthElasticity() override;

		std::string name() const override { return "GrowthElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void set_size(const int size) override;
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

	protected:
		int rows() const override { return size(); }
		int cols() const override { return 1; }

		double compute_energy(const MixedNonLinearAssemblerData &data) const override;
		Eigen::VectorXd compute_gradient(const MixedNonLinearAssemblerData &data) const override;
		Eigen::MatrixXd compute_hessian(const MixedNonLinearAssemblerData &data) const override;

	private:
		detail::GrowthElasticityModel &model();
		const detail::GrowthElasticityModel &model() const;

		std::unique_ptr<detail::GrowthElasticityModel> model_;
	};
} // namespace polyfem::assembler
