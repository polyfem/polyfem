#include "BilateralSegmentCertifier.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <ipc/candidates/candidates.hpp>
#include <ipc/candidates/edge_edge.hpp>
#include <ipc/candidates/face_vertex.hpp>
#include <ipc/ccd/tight_inclusion_ccd.hpp>
#include <ipc/collision_mesh.hpp>

#include <algorithm>
#include <cmath>

namespace polyfem::solver
{
	namespace
	{
		/// Translation-removed displacement bound U_tr and AABB gap at the segment start.
		void bounds(const ipc::VectorMax12d &xa, const ipc::VectorMax12d &xb, int na, double &utr, double &gap)
		{
			const int nv = int(xa.size() / 3);
			Eigen::Vector3d mean = Eigen::Vector3d::Zero();
			for (int i = 0; i < nv; ++i)
				mean += xb.segment<3>(3 * i) - xa.segment<3>(3 * i);
			mean /= nv;
			double ua = 0, ub = 0;
			for (int i = 0; i < nv; ++i)
			{
				const double e = (xb.segment<3>(3 * i) - xa.segment<3>(3 * i) - mean).norm();
				(i < na ? ua : ub) = std::max(i < na ? ua : ub, e);
			}
			utr = ua + ub;
			Eigen::Vector3d amin = Eigen::Vector3d::Constant(INFINITY), amax = -amin, bmin = amin, bmax = -amin;
			for (int i = 0; i < nv; ++i)
			{
				const Eigen::Vector3d p = xa.segment<3>(3 * i);
				if (i < na)
				{
					amin = amin.cwiseMin(p);
					amax = amax.cwiseMax(p);
				}
				else
				{
					bmin = bmin.cwiseMin(p);
					bmax = bmax.cwiseMax(p);
				}
			}
			double g2 = 0;
			for (int k = 0; k < 3; ++k)
			{
				const double g = std::max({bmin[k] - amax[k], amin[k] - bmax[k], 0.0});
				g2 += g * g;
			}
			gap = std::sqrt(g2);
		}
	} // namespace

	BilateralSegmentCertifier::BilateralSegmentCertifier(
		NLProblem &problem,
		const std::shared_ptr<BarrierContactForm> &contact,
		const std::vector<std::shared_ptr<Form>> &other_forms,
		const ipc::CollisionMesh &collision_mesh,
		const bool dedupe)
		: problem_(problem), contact_(contact), other_(other_forms), mesh_(collision_mesh), dedupe_(dedupe)
	{
	}

	double BilateralSegmentCertifier::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		// certificate factor for the stock separation rule (TI stops at 0.2 D(a), i.e. conservative rescaling 0.8)
		constexpr double cert_c = 0.8;
		const Eigen::VectorXd f0 = problem_.reduced_to_full(x0), f1 = problem_.reduced_to_full(x1);
		double step = 1;
		for (auto &f : other_)
			if (f->enabled())
				step = std::min(step, f->max_step_size(f0, f1));
		ipc::Candidates &C = contact_->cached_candidates();
		const ipc::TightInclusionCCD &ti = contact_->tight_inclusion_ccd();
		std::vector<char> keep(dedupe_ ? C.size() : 0, 1);
		const double far_thr = 1.001 * contact_->dhat();
		const auto V0 = contact_->compute_displaced_surface(f0);
		const auto V1 = contact_->compute_displaced_surface(f1);
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double margin = 1e-3 * contact_->dhat();
		double earliest = 1.0;
		// candidate order of ipc::Candidates::operator[]: vv, ev, ee, fv, pv. Vertex slot groups of the two primitives:
		// EE [a0 a1 | b0 b1], FV [v | f0 f1 f2]; vv / ev / pv candidates are not certified (always CCD).
		const size_t begin_ee = C.vv_candidates.size() + C.ev_candidates.size(), begin_fv = begin_ee + C.ee_candidates.size(),
					 end_fv = begin_fv + C.fv_candidates.size();
		// the certificate and the dedupe test of a candidate do not depend on the other candidates or on the earliest time
		// of impact: in parallel, flagging the candidates left for Tight Inclusion
		const size_t nC = C.size();
		std::vector<char> to_ti(nC, 0);
		utils::maybe_parallel_for(int(nC), [&](int start, int end, int /*thread_id*/) {
			for (size_t i = start; i < size_t(end); ++i)
			{
				if (i >= begin_ee && i < end_fv)
				{
					const ipc::CollisionStencil &s = C[i];
					const ipc::VectorMax12d xa = s.dof(V0, E, Fc), xb = s.dof(V1, E, Fc);
					const int na = i >= begin_fv ? 1 : 2;
					double utr, gap;
					bounds(xa, xb, na, utr, gap);
					if (gap >= (utr + margin) / cert_c)
					{
						if (dedupe_ && gap - utr >= far_thr)
							keep[i] = 0;
						continue;
					}
					const double Da = std::sqrt(s.compute_distance(xa));
					if (dedupe_ && Da - utr >= far_thr)
						keep[i] = 0; // farther than dhat on the whole segment: never active, never colliding
					if (utr <= cert_c * Da - margin)
						continue;
				}
				to_ti[i] = 1;
			}
		});
		// the certified candidates: the edge-edge and face-vertex candidates not left for Tight Inclusion
		n_certified += long(std::count(to_ti.begin() + begin_ee, to_ti.begin() + end_fv, char(0)));
		// Tight Inclusion in candidate order, each with the earliest time of impact so far as its tmax
		for (size_t i = 0; i < nC; ++i)
		{
			if (!to_ti[i])
				continue;
			const ipc::CollisionStencil &s = C[i];
			const ipc::VectorMax12d xa = s.dof(V0, E, Fc), xb = s.dof(V1, E, Fc);
			double toi = 1;
			++n_ti;
			if (s.ccd(xa, xb, toi, 0.0, earliest, ti))
				earliest = std::min(earliest, toi);
		}
		if (dedupe_)
		{
			// drop provably-far pairs from the cached candidates (order preserved)
			const size_t off_ee = begin_ee, off_fv = begin_fv;
			auto filt = [&](auto &vec, size_t off) {
				size_t w = 0;
				for (size_t i = 0; i < vec.size(); ++i)
					if (keep[off + i])
						vec[w++] = vec[i];
				n_dropped += long(vec.size() - w);
				n_kept += long(w);
				vec.erase(vec.begin() + w, vec.end());
			};
			filt(C.ee_candidates, off_ee);
			filt(C.fv_candidates, off_fv);
		}
		return std::min(step, earliest);
	}
} // namespace polyfem::solver
