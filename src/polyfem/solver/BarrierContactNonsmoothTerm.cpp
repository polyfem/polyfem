#include "BarrierContactNonsmoothTerm.hpp"

#include "BilateralSegmentCertifier.hpp"
#include "FastTetValidity.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/utils/Logger.hpp>

#include <ipc/candidates/candidates.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/potentials/barrier_potential.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace polyfem::solver
{
	namespace
	{
		using Vec = Eigen::VectorXd;
		constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

		/// null control on the candidate order: 1 = reverse, s >= 2 = shuffle seeded by s
		void candidate_permute(ipc::Candidates &C, int s)
		{
			if (s == 1)
			{
				std::reverse(C.ee_candidates.begin(), C.ee_candidates.end());
				std::reverse(C.fv_candidates.begin(), C.fv_candidates.end());
				return;
			}
			std::mt19937_64 g{static_cast<uint64_t>(s)};
			std::shuffle(C.ee_candidates.begin(), C.ee_candidates.end(), g);
			std::shuffle(C.fv_candidates.begin(), C.fv_candidates.end(), g);
		}

		template <typename T>
		T value_or(const json &j, const char *key, const T &def)
		{
			return j.is_object() && j.contains(key) ? j[key].get<T>() : def;
		}
	} // namespace

	bool selects_nonsmooth_newton(const json &nonlinear_params)
	{
		if (!nonlinear_params.is_object() || !nonlinear_params.contains("solver"))
			return false;
		const json &s = nonlinear_params["solver"];
		if (s.is_string())
			return s.get<std::string>() == "NonsmoothNewton";
		if (s.is_array())
			for (const auto &e : s)
				if (e.is_object() && e.contains("type") && e["type"] == "NonsmoothNewton")
					return true;
		return false;
	}

	BarrierContactNonsmoothTerm::Options BarrierContactNonsmoothTerm::Options::from_json(const json &nonlinear_params)
	{
		Options o;
		if (!nonlinear_params.is_object() || !nonlinear_params.contains("NonsmoothNewton"))
			return o;
		const json &p = nonlinear_params["NonsmoothNewton"];
		if (!p.is_object() || !p.contains("debug"))
			return o;
		const json &d = p["debug"];
		o.certificate = value_or(d, "certificate", o.certificate);
		o.dedupe = value_or(d, "dedupe", o.dedupe);
		o.local_blocks = value_or(d, "local_blocks", o.local_blocks);
		o.fast_validity = value_or(d, "fast_validity", o.fast_validity);
		o.candidate_permutation = value_or(d, "candidate_permutation", o.candidate_permutation);
		return o;
	}

	struct BarrierContactNonsmoothTerm::AnchorHolder
	{
		FastTetValidity::Anchor anchor;
	};

	BarrierContactNonsmoothTerm::BarrierContactNonsmoothTerm(NLProblem &problem,
															 const std::vector<std::shared_ptr<Form>> &forms,
															 const Eigen::VectorXd &mass_full,
															 const ipc::CollisionMesh &collision_mesh,
															 const Eigen::SparseMatrix<double> *displacement_map,
															 const std::shared_ptr<const FastTetValidity> &fast_validity,
															 const Options &opts)
		: problem_(problem), mesh_(collision_mesh), displacement_map_(displacement_map), fast_validity_(fast_validity),
		  opts_(opts), mass_full_(mass_full), anchor_(std::make_unique<AnchorHolder>())
	{
		for (const auto &f : forms)
		{
			if (auto c = std::dynamic_pointer_cast<BarrierContactForm>(f))
			{
				if (contact_)
					log_and_throw_error("NonsmoothNewton: more than one barrier contact form");
				contact_ = c;
				continue;
			}
			other_.push_back(f);
		}
		if (!contact_)
			log_and_throw_error("NonsmoothNewton: the problem has no barrier contact form");
		certifier_ = std::make_unique<BilateralSegmentCertifier>(problem_, contact_, other_, mesh_, opts_.dedupe);
		const auto &Vr = mesh_.rest_positions();
		const auto &Ed = mesh_.edges();
		double sum = 0;
		for (int e = 0; e < Ed.rows(); ++e)
			sum += (Vr.row(Ed(e, 1)) - Vr.row(Ed(e, 0))).norm();
		h_edge_ = Ed.rows() ? sum / Ed.rows() : NaN;
		if (opts_.local_blocks)
			init_local_map();
	}

	BarrierContactNonsmoothTerm::~BarrierContactNonsmoothTerm() = default;

	// ------------------------------------------------------------------------------------------------------ dof maps
	// The term works in the reduced space (cf. NLProblem::gradient / hessian). Without constraint forms the reduced space
	// is the full space: the reduction matrices are not set up and full-size quantities are already reduced.
	Vec BarrierContactNonsmoothTerm::to_reduced(const Vec &full) const
	{
		return problem_.full_size() != problem_.reduced_size() ? problem_.full_to_reduced_grad(full) : full;
	}

	void BarrierContactNonsmoothTerm::to_reduced(StiffnessMatrix &hessian) const
	{
		if (problem_.full_size() != problem_.reduced_size())
			problem_.full_hessian_to_reduced_hessian(hessian);
	}

	void BarrierContactNonsmoothTerm::init_local_map()
	{
		local_ok_ = false;
		const Vec x0 = Vec::Zero(problem_.reduced_size());
		const int nf = int(problem_.reduced_to_full(x0).size());
		bool ok = problem_.reduction_is_selection(red_of_full_) && int(red_of_full_.size()) == nf;
		if (ok)
			ok = init_node_of_vertex(nf);
		local_ok_ = ok;
		if (!local_ok_)
			logger().warn("[NonsmoothNewton] local contact blocks disabled: dof map is not a plain selection");
	}

	bool BarrierContactNonsmoothTerm::init_node_of_vertex(const int nf)
	{
		// collision vertex v follows the full vertex fv = to_full_vertex_id(v) of the collision mesh, which follows the FE
		// nodes through the displacement map (row fv); the local path needs exactly one FE node with weight 1 per row
		const int dim = 3;
		if (nf % dim != 0)
			return false;
		const int n_nodes = nf / dim;
		const int nv = int(mesh_.num_vertices());
		const int n_full = int(mesh_.full_num_vertices());
		std::vector<int> node_of_full(n_full, -1);
		if (displacement_map_ == nullptr || displacement_map_->size() == 0)
		{
			if (n_full > n_nodes)
				return false;
			for (int fv = 0; fv < n_full; ++fv)
				node_of_full[fv] = fv;
		}
		else
		{
			const Eigen::SparseMatrix<double> &T = *displacement_map_;
			if (T.rows() != n_full || T.cols() != n_nodes)
				return false;
			std::vector<int> count(n_full, 0);
			for (int col = 0; col < T.outerSize(); ++col)
				for (Eigen::SparseMatrix<double>::InnerIterator it(T, col); it; ++it)
				{
					if (it.value() == 0)
						continue;
					const int row = int(it.row());
					if (++count[row] > 1 || it.value() != 1)
						node_of_full[row] = -2; // not a single unit weight
					else
						node_of_full[row] = int(it.col());
				}
		}
		node_of_vertex_.assign(nv, -1);
		for (int v = 0; v < nv; ++v)
		{
			const int fv = int(mesh_.to_full_vertex_id(v));
			if (fv < 0 || fv >= n_full || node_of_full[fv] < 0 || node_of_full[fv] >= n_nodes)
				return false;
			node_of_vertex_[v] = node_of_full[fv];
		}
		return true;
	}

	// ---------------------------------------------------------------------------------------------------- smooth part
	void BarrierContactNonsmoothTerm::smooth_gradient(const Vec &x, Vec &grad)
	{
		const Vec xfull = problem_.reduced_to_full(x);
		Vec gn = Vec::Zero(xfull.size());
		for (auto &f : other_)
		{
			if (!f->enabled())
				continue;
			Vec gt;
			f->first_derivative(xfull, gt);
			gn += gt;
		}
		grad = to_reduced(gn);
	}

	void BarrierContactNonsmoothTerm::smooth_hessian(const Vec &x, const bool project_to_psd, StiffnessMatrix &hessian)
	{
		const Vec xfull = problem_.reduced_to_full(x);
		for (auto &f : other_)
			f->set_project_to_psd(project_to_psd);
		StiffnessMatrix h(xfull.size(), xfull.size());
		for (auto &f : other_)
		{
			if (!f->enabled())
				continue;
			StiffnessMatrix tmp;
			f->second_derivative(xfull, tmp);
			h += tmp;
		}
		to_reduced(h);
		hessian = std::move(h);
	}

	void BarrierContactNonsmoothTerm::ensure_anchor(const Vec &x)
	{
		auto &a = anchor_->anchor;
		if (a.x.size() == x.size() && a.x == x)
			return;
		fast_validity_->anchor(x, problem_.reduced_to_full(x), a);
	}

	double BarrierContactNonsmoothTerm::smooth_valid_fraction(const Vec &x, const Vec &a, const Vec &b)
	{
		if (!(opts_.fast_validity && fast_validity_ != nullptr))
			return valid_fraction_native(a, b);
		ensure_anchor(x);
		return fast_validity_->valid_fraction(anchor_->anchor, a, b, [this](const Vec &y) { return problem_.reduced_to_full(y); });
	}

	/// Native validity shortening (cf. polysolve compute_nan_free_step_size): halve until the step is valid and the
	/// smooth energy is finite. Returns the fraction.
	double BarrierContactNonsmoothTerm::valid_fraction_native(const Vec &a, const Vec &b)
	{
		double alpha = 1;
		for (int i = 0; i < 40; ++i, alpha *= 0.5)
		{
			const Vec y = a + alpha * (b - a);
			if (!problem_.is_step_valid(a, y))
				continue;
			const Vec yf = problem_.reduced_to_full(y);
			double e = 0;
			for (auto &f : other_)
				if (f->enabled())
					e += f->value(yf);
			if (std::isfinite(e))
				return alpha;
		}
		return 0;
	}

	void BarrierContactNonsmoothTerm::metric_diagonal(Vec &diag) const
	{
		diag = problem_.full_to_reduced_diag(mass_full_);
	}

	// ---------------------------------------------------------------------------------------------- nonsmooth term
	void BarrierContactNonsmoothTerm::update(const Vec &y)
	{
		contact_->solution_changed(problem_.reduced_to_full(y));
	}

	double BarrierContactNonsmoothTerm::value(const Vec &y)
	{
		return contact_->value(problem_.reduced_to_full(y));
	}

	void BarrierContactNonsmoothTerm::gradient(const Vec &y, Vec &grad)
	{
		Vec g;
		contact_->first_derivative(problem_.reduced_to_full(y), g);
		grad = to_reduced(g);
	}

	void BarrierContactNonsmoothTerm::hessian(const Vec &y, const bool project_to_psd, StiffnessMatrix &hessian)
	{
		contact_->set_project_to_psd(project_to_psd);
		StiffnessMatrix K;
		contact_->second_derivative(problem_.reduced_to_full(y), K);
		to_reduced(K);
		hessian = std::move(K);
	}

	bool BarrierContactNonsmoothTerm::hessian_blocks(const Vec &y, Vec &grad, std::vector<polysolve::nonlinear::HessianBlock> &blocks)
	{
		if (!(opts_.local_blocks && local_ok_))
			return false;
		const auto V = contact_->compute_displaced_surface(problem_.reduced_to_full(y));
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight();
		grad = Vec::Zero(y.size());
		blocks.clear();
		blocks.reserve(cs.size());
		for (size_t i = 0; i < cs.size(); ++i)
		{
			const auto &col = cs[i];
			const ipc::VectorMax12d xc = col.dof(V, E, Fc);
			const auto ids = col.vertex_ids(E, Fc);
			polysolve::nonlinear::HessianBlock &b = blocks.emplace_back();
			b.size = int(xc.size());
			const ipc::VectorMax12d g = w * bp.gradient(col, xc);
			b.H.topLeftCorner(b.size, b.size) = w * bp.hessian(col, xc, ipc::PSDProjectionMethod::NONE);
			for (int a = 0; a < b.size / 3; ++a)
				for (int d = 0; d < 3; ++d)
				{
					const int r = ids[a] >= 0 ? red_of_full_[3 * node_of_vertex_[ids[a]] + d] : -1;
					b.indices[3 * a + d] = r;
					if (r >= 0)
						grad[r] += g[3 * a + d];
				}
		}
		return true;
	}

	int BarrierContactNonsmoothTerm::active_size() const
	{
		return int(contact_->collision_set().size());
	}

	void BarrierContactNonsmoothTerm::active_distances(const Vec &a, const Vec &b, Vec &distance, Vec &rate)
	{
		const auto Va = contact_->compute_displaced_surface(problem_.reduced_to_full(a));
		const auto Vb = contact_->compute_displaced_surface(problem_.reduced_to_full(b));
		const auto &cs = contact_->collision_set();
		const auto &Em = mesh_.edges();
		const auto &Fm = mesh_.faces();
		std::vector<double> dv, rv;
		dv.reserve(cs.size());
		rv.reserve(cs.size());
		for (size_t i = 0; i < cs.size(); ++i)
		{
			const auto &c = cs[i];
			const Vec xa = c.dof(Va, Em, Fm), xb = c.dof(Vb, Em, Fm);
			const double d2 = c.compute_distance(xa);
			if (!(d2 > 0))
				continue;
			const double dist = std::sqrt(d2);
			dv.push_back(dist);
			rv.push_back(c.compute_distance_gradient(xa).dot(xb - xa) / (2 * dist)); // d'(0) along the segment
		}
		distance = Eigen::Map<const Vec>(dv.data(), Eigen::Index(dv.size()));
		rate = Eigen::Map<const Vec>(rv.data(), Eigen::Index(rv.size()));
	}

	double BarrierContactNonsmoothTerm::step_length(const Vec &a, const Vec &b)
	{
		const auto Va = contact_->compute_displaced_surface(problem_.reduced_to_full(a));
		const auto Vb = contact_->compute_displaced_surface(problem_.reduced_to_full(b));
		return (Vb - Va).rowwise().norm().maxCoeff();
	}

	double BarrierContactNonsmoothTerm::activation_distance() const
	{
		return contact_->dhat();
	}

	void BarrierContactNonsmoothTerm::segment_begin(const Vec &a, const Vec &b)
	{
		problem_.line_search_begin(a, b);
		if (opts_.candidate_permutation > 0)
			candidate_permute(contact_->cached_candidates(), opts_.candidate_permutation);
	}

	bool BarrierContactNonsmoothTerm::certified_max_step_size() const
	{
		return opts_.certificate && !problem_.is_residual() && contact_->uses_cached_candidates();
	}

	double BarrierContactNonsmoothTerm::max_step_size(const Vec &x0, const Vec &x1)
	{
		return certifier_->max_step_size(x0, x1);
	}

	json BarrierContactNonsmoothTerm::stats() const
	{
		return {{"certified_pairs", certifier_->n_certified},
				{"ti_pairs", certifier_->n_ti},
				{"local_contact_blocks", local_ok_}};
	}
} // namespace polyfem::solver
