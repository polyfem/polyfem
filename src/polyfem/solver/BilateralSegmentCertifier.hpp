#pragma once

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace ipc
{
	class CollisionMesh;
}

namespace polyfem::solver
{
	class Form;
	class NLProblem;
	class BarrierContactForm;

	/// @brief Shared narrow phase of the NonsmoothNewton contact term (performance path).
	///
	/// Same result as ContactForm::max_step_size on the cached swept candidates (same candidates, order, tmax pruning and
	/// Tight Inclusion object), with a displacement certificate in front of TI: a face-vertex or edge-edge pair whose AABB
	/// gap or start distance D(a) exceeds the translation-removed displacement bound U_tr enough to keep
	/// D >= 0.2 D(a) + 0.001 dhat on the whole segment cannot make TI stop it, so TI is skipped. With dedupe, pairs that
	/// provably stay farther than dhat on the whole segment are also removed from the cached candidates (order preserved),
	/// so collision sets rebuilt at trial points of the segment touch only pairs that can become active.
	/// The step bounds of the other forms are combined as in FullNLProblem::max_step_size.
	class BilateralSegmentCertifier
	{
	public:
		BilateralSegmentCertifier(NLProblem &problem,
								  const std::shared_ptr<BarrierContactForm> &contact,
								  const std::vector<std::shared_ptr<Form>> &other_forms,
								  const ipc::CollisionMesh &collision_mesh,
								  const bool dedupe);

		/// @brief Largest collision-free step fraction on the reduced segment x0 -> x1 (cached candidates must be installed)
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1);

		long n_certified = 0; ///< pairs skipped by the certificate
		long n_ti = 0;        ///< pairs passed to TI
		long n_dropped = 0;   ///< candidates removed by dedupe
		long n_kept = 0;      ///< candidates kept by dedupe

	private:
		NLProblem &problem_;
		std::shared_ptr<BarrierContactForm> contact_;
		std::vector<std::shared_ptr<Form>> other_;
		const ipc::CollisionMesh &mesh_;
		const bool dedupe_;
	};
} // namespace polyfem::solver
