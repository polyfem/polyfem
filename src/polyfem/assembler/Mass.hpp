#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	class Mass : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		/// computes and returns local stiffness matrix (1x1) for
		/// bases i,j (where i,j is passed in through data)
		/// ie integral of phi_i * phi_j on the given element
		FlatMatrixNd
		assemble(const LinearAssemblerData &data) const override;

		/// uses autodiff to compute the rhs for a fabricated solution
		/// in this case it just return pt.getHessian().trace()
		/// pt is the evaluation of the solution at a point
		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		/// inialize material parameter
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		/// class that stores and compute density per point
		const Density &density() const { return density_; }

		std::string name() const override { return "Mass"; }
		virtual std::map<std::string, ParamFunc> parameters() const override;

	private:
		// class that stores and compute density per point
		Density density_;
	};
} // namespace polyfem::assembler
