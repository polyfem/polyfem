#include <polyfem/mesh/remesh/PhysicsRemesher.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/assembler/MassMatrixAssembler.hpp>
#include <polyfem/assembler/Mass.hpp>

#include <wmtk/ExecutionScheduler.hpp>
#include <wmtk/utils/TriQualityUtils.hpp>
#include <wmtk/utils/TetraQualityUtils.hpp>
#include <wmtk/utils/TupleUtils.hpp>
#include <wmtk/utils/AMIPS.h>
#include <wmtk/utils/AMIPS2D.h>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	double get_quality(const WildRemesher<WMTKMesh> &m, const typename WMTKMesh::Tuple &t)
	{
		// Global ids of the vertices of the triangle
		auto vids = m.element_vids(t);

		// Temporary variable to store the stacked coordinates of the triangle
		constexpr int NDOF = WildRemesher<WMTKMesh>::DIM * WildRemesher<WMTKMesh>::VERTICES_PER_ELEMENT;
		std::array<double, NDOF> T;
		for (auto i = 0; i < WildRemesher<WMTKMesh>::VERTICES_PER_ELEMENT; i++)
			for (auto j = 0; j < WildRemesher<WMTKMesh>::DIM; j++)
				T[i * WildRemesher<WMTKMesh>::DIM + j] =
					m.vertex_attrs[vids[i]].rest_position[j];

		// Energy evaluation
		double energy;
		if constexpr (std::is_same_v<wmtk::TriMesh, WMTKMesh>)
			energy = wmtk::AMIPS2D_energy(T);
		else
			energy = wmtk::AMIPS_energy(T);

		// Filter for numerical issues
		if (!std::isfinite(energy))
			return std::numeric_limits<double>::max();

		return energy;
	}

	template <class WMTKMesh>
	void project_local_quantities(
		WildRemesher<WMTKMesh> &m,
		LocalMesh<WildRemesher<WMTKMesh>> &old_local_mesh,
		LocalMesh<WildRemesher<WMTKMesh>> &new_local_mesh)
	{
		POLYFEM_REMESHER_SCOPED_TIMER("Project local quantities");

		const std::vector<basis::ElementBases> from_bases =
			old_local_mesh.build_bases(m.state.formulation());
		const std::vector<basis::ElementBases> to_bases =
			new_local_mesh.build_bases(m.state.formulation());

		const Eigen::MatrixXd &from_projection_quantities = old_local_mesh.projection_quantities();
		const Eigen::MatrixXd &to_projection_quantities = new_local_mesh.projection_quantities();

		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			assembler::MassMatrixAssembler assembler;
			assembler::Mass mass_matrix_assembler;
			assembler::AssemblyValsCache cache;

			mass_matrix_assembler.assemble(
				m.is_volume(), new_local_mesh.num_vertices(),
				to_bases, to_bases, cache, 0, M, true);
			assert(M.rows() == new_local_mesh.num_vertices() * m.dim());

			assembler.assemble_cross(
				m.is_volume(), m.dim(),
				old_local_mesh.num_vertices(), from_bases, from_bases,
				new_local_mesh.num_vertices(), to_bases, to_bases,
				cache, A);
			assert(A.rows() == new_local_mesh.num_vertices() * m.dim());
			assert(A.cols() == old_local_mesh.num_vertices() * m.dim());
		}

		// --------------------------------------------------------------------

		// NOTE: This assumes the boundary is fixed.
		ipc::CollisionMesh collision_mesh;
		// ipc::CollisionMesh collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
		// 	local_mesh.rest_positions(), local_mesh.boundary_edges(), local_mesh.boundary_faces());
		// // Ignore all collisions between fixed elements.
		// std::vector<bool> is_vertex_fixed(local_mesh.num_vertices(), false);
		// for (const int vi : local_mesh.fixed_vertices())
		// 	is_vertex_fixed[vi] = true;
		// collision_mesh.can_collide = [is_vertex_fixed, &collision_mesh](size_t vi, size_t vj) {
		// 	return !is_vertex_fixed[collision_mesh.to_full_vertex_id(vi)]
		// 		   || !is_vertex_fixed[collision_mesh.to_full_vertex_id(vj)];
		// };

		// --------------------------------------------------------------------

		std::vector<int> boundary_nodes;
		for (int d = 0; d < m.dim(); ++d)
		{
			// Internal vertices that are on the boundary of the local mesh
			for (const int fv : new_local_mesh.fixed_vertices())
				boundary_nodes.push_back(fv * m.dim() + d);
			// Boundary vertices that are on the boundary of the local mesh
			for (const int fv : new_local_mesh.boundary_facets().reshaped())
				boundary_nodes.push_back(fv * m.dim() + d);
		}
		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto new_end = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.erase(new_end, boundary_nodes.end());

		// --------------------------------------------------------------------

		Eigen::MatrixXd projected_quantities = to_projection_quantities;
		const int n_constrained_quantaties = projected_quantities.cols() / 3;
		const int n_unconstrained_quantaties = projected_quantities.cols() - n_constrained_quantaties;

		auto nl_solver = m.state.make_nl_solver(/*for_al=*/false);
		for (int i = 0; i < n_constrained_quantaties; ++i)
		{
			const auto level_before = logger().level();
			logger().set_level(spdlog::level::warn);
			projected_quantities.col(i) = constrained_L2_projection(
				nl_solver,
				// L2 projection form
				M, A, /*y=*/from_projection_quantities.col(i),
				// Inversion-free form
				new_local_mesh.rest_positions(), new_local_mesh.elements(), m.dim(),
				// Contact form
				collision_mesh, m.state.args["contact"]["dhat"],
				m.state.solve_data.contact_form
					? m.state.solve_data.contact_form->barrier_stiffness()
					: 1.0,
				m.state.args["contact"]["use_convergent_formulation"],
				m.state.args["solver"]["contact"]["CCD"]["broad_phase"],
				m.state.args["solver"]["contact"]["CCD"]["tolerance"],
				m.state.args["solver"]["contact"]["CCD"]["max_iterations"],
				// Augmented lagrangian form
				boundary_nodes, /*obstacle_ndof=*/0, to_projection_quantities.col(i),
				// Initial guess
				to_projection_quantities.col(i));
			logger().set_level(level_before);
		}

		// Minimize the L2 norm with the boundary fixed.
		reduced_L2_projection(
			M, A, from_projection_quantities.rightCols(n_unconstrained_quantaties),
			boundary_nodes, projected_quantities.rightCols(n_unconstrained_quantaties));

		// --------------------------------------------------------------------

		assert(projected_quantities.rows() == m.dim() * new_local_mesh.num_vertices());
		for (int i = 0; i < new_local_mesh.num_vertices(); i++)
		{
			m.vertex_attrs[new_local_mesh.local_to_global()[i]].projection_quantities =
				projected_quantities.middleRows(m.dim() * i, m.dim());
		}
	}

	// =========================================================================

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::smooth_before(const Tuple &v)
	{
		if (!WMTKMesh::smooth_before(v))
			return false;
		return !vertex_attrs[v.vid(*this)].fixed;
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::smooth_before(const Tuple &v)
	{
		if (!Super::smooth_before(v))
			return false;

		if (this->op_cache == nullptr)
		{
			if constexpr (std::is_same_v<WMTKMesh, wmtk::TriMesh>)
				this->op_cache = std::make_shared<TriOperationCache>();
			else
				this->op_cache = std::make_shared<TetOperationCache>();
		}

		this->op_cache->local_energy = local_mesh_energy(
			vertex_attrs[v.vid(*this)].rest_position);

		return true;
	}

	// -------------------------------------------------------------------------

	template <class WMTKMesh>
	bool WildRemesher<WMTKMesh>::smooth_after(const Tuple &v)
	{
		if (!WMTKMesh::smooth_before(v))
			return false;

		const size_t vid = v.vid(*this);

		const std::vector<Tuple> one_ring = this->get_one_ring_elements_for_vertex(v);
		assert(one_ring.size() > 0);

		// ---------------------------------------------------------------------
		// 1. update rest position of new vertex

		// Computes the maximal error around the one ring
		// that is needed to ensure the operation will decrease the error measure
		double max_quality = 0;
		for (const Tuple &t : get_one_ring_elements_for_vertex(v))
			max_quality = std::max(max_quality, get_quality(*this, t));
		assert(max_quality > 0); // If max quality is zero it is likely that the triangles are flipped

		// Collects the coordinate of all vertices in the 1-ring
		constexpr int NDOF = DIM * VERTICES_PER_ELEMENT;
		std::vector<std::array<double, NDOF>> assembles(one_ring.size());

		// For each triangle, make a reordered copy of the vertices so that
		// the vertex to optimize is always the first
		for (int i = 0; i < one_ring.size(); i++)
		{
			const Tuple &t = one_ring[i];
			auto t_id = t.fid(*this);

			assert(!is_inverted(t));
			const auto local_verts = orient_preserve_element_reorder(element_vids(t), vid);

			for (auto j = 0; j < VERTICES_PER_ELEMENT; j++)
				for (auto d = 0; d < DIM; d++)
					assembles[i][j * DIM + d] = vertex_attrs[local_verts[j]].rest_position[d];
		}

		// Make a backup of the current configuration
		LocalMesh<This> old_local_mesh(*this, one_ring, false);
		const VectorNd old_rest_position = vertex_attrs[vid].rest_position;

		// Minimize distortion using newton's method
		const auto log_lvl = wmtk::logger().level();
		wmtk::logger().set_level(spdlog::level::warn);
		if constexpr (DIM == 2)
			vertex_attrs[vid].rest_position = wmtk::newton_method_from_stack_2d(
				assembles, wmtk::AMIPS2D_energy, wmtk::AMIPS2D_jacobian, wmtk::AMIPS2D_hessian);
		else
			vertex_attrs[vid].rest_position = wmtk::newton_method_from_stack(
				assembles, wmtk::AMIPS_energy, wmtk::AMIPS_jacobian, wmtk::AMIPS_hessian);
		wmtk::logger().set_level(log_lvl);

		// The AMIPS energy should have prevented inversions
		if (std::any_of(one_ring.begin(), one_ring.end(), [this](const Tuple &t) {
				return this->is_rest_inverted(t);
			}))
		{
			assert(false);
			return false;
		}

		// ---------------------------------------------------------------------
		// 2. project quantities so to minimize the L2 error

		// Adjust the previous displacements so they results in the same previous positions
		vertex_attrs[vid].projection_quantities.leftCols(n_quantities() / 3) +=
			old_rest_position - vertex_attrs[vid].rest_position;

		LocalMesh<This> new_local_mesh(*this, one_ring, false);

		project_local_quantities(*this, old_local_mesh, new_local_mesh);

		// The Constrained L2 Projection should have prevented inversions
		if (std::any_of(one_ring.begin(), one_ring.end(), [this](const Tuple &t) {
				return this->is_inverted(t);
			}))
		{
			assert(false);
			return false;
		}

		return true;
	}

	template <class WMTKMesh>
	bool PhysicsRemesher<WMTKMesh>::smooth_after(const Tuple &v)
	{
		utils::Timer timer(this->timings["Smooth vertex after"]);
		timer.start();
		if (!Super::smooth_after(v))
			return false;
		// local relaxation has its own timers
		timer.stop();

		// ---------------------------------------------------------------------
		// 3. perform a local relaxation of the n-ring to get an estimate of the
		//    energy decrease.
		const std::vector<Tuple> one_ring = this->get_one_ring_elements_for_vertex(v);
		return local_relaxation(one_ring, args["smooth"]["acceptance_tolerance"]);
	}

	template <class WMTKMesh>
	void WildRemesher<WMTKMesh>::smooth_vertices()
	{
		executor.priority = [](const WildRemesher<WMTKMesh> &m, std::string op, const Tuple &v) -> double {
			double max_quality = 0;
			for (const Tuple &t : m.get_one_ring_elements_for_vertex(v))
				max_quality = std::max(max_quality, get_quality(m, t));
			assert(max_quality > 0); // If max quality is zero it is likely that the triangles are flipped
			return max_quality;
		};
		executor.should_renew = [](auto) { return true; };
		executor.renew_neighbor_tuples = [](auto &, auto, auto &) -> Operations { return {}; };

		const int max_iters = args["smooth"]["max_iters"];
		for (int i = 0; i < max_iters; i++)
		{
			Operations smooths;
			for (auto &v : WMTKMesh::get_vertices())
				smooths.emplace_back("vertex_smooth", v);
			executor(*this, smooths);
			if (executor.cnt_success() == 0)
				break;
		}
	}

	// ------------------------------------------------------------------------
	// Template specializations

	template class WildRemesher<wmtk::TriMesh>;
	template class WildRemesher<wmtk::TetMesh>;
	template class PhysicsRemesher<wmtk::TriMesh>;
	template class PhysicsRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh