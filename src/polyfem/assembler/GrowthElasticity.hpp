#pragma once

#include <polyfem/assembler/Assembler.hpp>

#include <memory>

namespace polyfem::assembler
{
	namespace detail
	{
		class GrowthElasticityModel;
	}

	/// Mixed (displacement, growth) assembler implementing the correction energy
	/// of finite areal growth of a thin wall:
	///
	///     W = J^g * psi_e(Fe) - psi_e(F),
	///     Fe  = F * inv(Fg),
	///     Fg  = sqrt(theta) * (I - n0 x n0) + vn * n0 x n0,
	///     J^g = theta * vn,
	///
	/// with theta the scalar growth field (the psi-space DOF, interpolated at
	/// quadrature points), n0 the reference wall normal (per-element data),
	/// and vn the prescribed normal growth component (default 1: pure in-plane
	/// growth; a per-element field prescribes thickening or thinning).
	///
	/// The subtraction of psi_e(F) makes this an additive correction: the
	/// standard elastic form on the displacement block stays registered
	/// unchanged, and at theta == 1, vn == 1 the correction is identically
	/// zero (bitwise: both energy calls see the same def_grad), so an ungrown
	/// body reproduces the plain elastic material exactly.
	///
	/// psi_e is the fitted MaterialSum composition, re-composed here from
	/// concrete child materials (see GrowthElasticity.cpp) because SumModel's
	/// type-erased children expose no templated energy; the children's
	/// templated elastic_energy bodies are reused verbatim.
	///
	/// Follows the structure of ThermoElasticity (the in-tree precedent for a
	/// displacement/scalar mixed correction assembler).
	/// Read one CELL_DATA array of a legacy ASCII VTK (VECTORS, SCALARS or
	/// FIELD layout) as an (n_cells x ncomp_expected) matrix. Shared by the
	/// assembler (normal_direction, normal_growth) and the growth VarForm
	/// (growth_law threshold), so per-element growth data has one reader.
	Eigen::MatrixXd read_growth_cell_array(
		const io::ResourceIO &resources, const std::string &path, const std::string &field, const int ncomp_expected);

	class GrowthElasticity : public MixedNLAssembler
	{
	public:
		GrowthElasticity();
		~GrowthElasticity() override;

		std::string name() const override { return "GrowthElasticity"; }
		std::map<std::string, ParamFunc> parameters() const override;

		void set_size(const int size) override;
		void add_multimaterial(const int index, const json &params, const Units &units, const io::ResourceIO &resources) override;

		/// Growth driving force f_I = -(1/vn) d(J^g psi_e)/d theta as a function of
		/// the elastic right Cauchy-Green tensor alone (energy per intermediate
		/// volume): with Fe = Ce^{1/2},
		/// f_I(Ce) = -d/dtheta [theta psi_e(Fe Fg^{-1}(theta))] at theta = 1, vn = 1.
		/// Ce is in global coordinates; el_id, pos and t select the material
		/// parameters (per-element fibers) exactly as in the energy.
		double driving_force(const int el_id, const RowVectorNd &pos, const double t, const Eigen::Matrix3d &Ce) const;
		/// Reference wall normal n0 of element el_id (unit).
		Eigen::Vector3d wall_normal(const int el_id) const;

		/// Stresses of the grown material, total energy density J^g psi_e(F Fg^{-1}),
		/// at the total deformation gradient F and growth theta (vn per element):
		///     P     = J^g Pe(Fe) Fg^{-T},   Pe = d psi_e / d Fe,
		///     sigma = P F^T / J = Pe(Fe) Fe^T / J^e,
		///     S     = F^{-1} P.
		/// At theta == 1, vn == 1 these are the plain elastic stresses at F. All
		/// tensors are 3x3 in global coordinates; pos is the global position, as
		/// in the energy.
		void grown_stresses(
			const int el_id, const RowVectorNd &pos, const double t,
			const Eigen::Matrix3d &F, const double theta,
			Eigen::Matrix3d &cauchy, Eigen::Matrix3d &pk1, Eigen::Matrix3d &pk2) const;

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
