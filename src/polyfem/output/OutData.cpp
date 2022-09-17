#include "OutData.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::output
{

	void OutGeometryData::init_sampler()
	{
		ref_element_sampler.init(mesh->is_volume(), mesh->n_elements(), args["output"]["paraview"]["vismesh_rel_area"]);
	}

	void OutGeometryData::build_grid()
	{
		const double spacing = args["output"]["advanced"]["sol_on_grid"];
		RowVectorNd min, max;
		mesh->bounding_box(min, max);
		const RowVectorNd delta = max - min;
		const int nx = delta[0] / spacing + 1;
		const int ny = delta[1] / spacing + 1;
		const int nz = delta.cols() >= 3 ? (delta[2] / spacing + 1) : 1;
		const int n = nx * ny * nz;

		grid_points.resize(n, delta.cols());
		int index = 0;
		for (int i = 0; i < nx; ++i)
		{
			const double x = (delta[0] / (nx - 1)) * i + min[0];

			for (int j = 0; j < ny; ++j)
			{
				const double y = (delta[1] / (ny - 1)) * j + min[1];

				if (delta.cols() <= 2)
				{
					grid_points.row(index++) << x, y;
				}
				else
				{
					for (int k = 0; k < nz; ++k)
					{
						const double z = (delta[2] / (nz - 1)) * k + min[2];
						grid_points.row(index++) << x, y, z;
					}
				}
			}
		}

		assert(index == n);

		std::vector<std::array<Eigen::Vector3d, 2>> boxes;
		mesh->elements_boxes(boxes);

		BVH::BVH bvh;
		bvh.init(boxes);

		const double eps = 1e-6;

		grid_points_to_elements.resize(grid_points.rows(), 1);
		grid_points_to_elements.setConstant(-1);

		grid_points_bc.resize(grid_points.rows(), mesh->is_volume() ? 4 : 3);

		for (int i = 0; i < grid_points.rows(); ++i)
		{
			const Eigen::Vector3d min(
				grid_points(i, 0) - eps,
				grid_points(i, 1) - eps,
				(mesh->is_volume() ? grid_points(i, 2) : 0) - eps);

			const Eigen::Vector3d max(
				grid_points(i, 0) + eps,
				grid_points(i, 1) + eps,
				(mesh->is_volume() ? grid_points(i, 2) : 0) + eps);

			std::vector<unsigned int> candidates;

			bvh.intersect_box(min, max, candidates);

			for (const auto cand : candidates)
			{
				if (!mesh->is_simplex(cand))
				{
					logger().warn("Element {} is not simplex, skipping", cand);
					continue;
				}

				Eigen::MatrixXd coords;
				mesh->barycentric_coords(grid_points.row(i), cand, coords);

				for (int d = 0; d < coords.size(); ++d)
				{
					if (fabs(coords(d)) < 1e-8)
						coords(d) = 0;
					else if (fabs(coords(d) - 1) < 1e-8)
						coords(d) = 1;
				}

				if (coords.array().minCoeff() >= 0 && coords.array().maxCoeff() <= 1)
				{
					grid_points_to_elements(i) = cand;
					grid_points_bc.row(i) = coords;
					break;
				}
			}
		}
	}

	void OutStatsData::compute_mesh_size(const Mesh &mesh_in, const std::vector<ElementBases> &bases_in, const int n_samples)
	{
		Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1, p;

		mesh_size = 0;
		average_edge_length = 0;
		min_edge_length = std::numeric_limits<double>::max();

		if (!args["output"]["advanced"]["curved_mesh_size"])
		{
			mesh_in.get_edges(p0, p1);
			p = p0 - p1;
			min_edge_length = p.rowwise().norm().minCoeff();
			average_edge_length = p.rowwise().norm().mean();
			mesh_size = p.rowwise().norm().maxCoeff();

			logger().info("hmin: {}", min_edge_length);
			logger().info("hmax: {}", mesh_size);
			logger().info("havg: {}", average_edge_length);

			return;
		}

		if (mesh_in.is_volume())
		{
			EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_3d_cube(n_samples, samples_cube);
		}
		else
		{
			EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
			EdgeSampler::sample_2d_cube(n_samples, samples_cube);
		}

		int n = 0;
		for (size_t i = 0; i < bases_in.size(); ++i)
		{
			if (mesh_in.is_polytope(i))
				continue;
			int n_edges;

			if (mesh_in.is_simplex(i))
			{
				n_edges = mesh_in.is_volume() ? 6 : 3;
				bases_in[i].eval_geom_mapping(samples_simplex, mapped);
			}
			else
			{
				n_edges = mesh_in.is_volume() ? 12 : 4;
				bases_in[i].eval_geom_mapping(samples_cube, mapped);
			}

			for (int j = 0; j < n_edges; ++j)
			{
				double current_edge = 0;
				for (int k = 0; k < n_samples - 1; ++k)
				{
					p0 = mapped.row(j * n_samples + k);
					p1 = mapped.row(j * n_samples + k + 1);
					p = p0 - p1;

					current_edge += p.norm();
				}

				mesh_size = std::max(current_edge, mesh_size);
				min_edge_length = std::min(current_edge, min_edge_length);
				average_edge_length += current_edge;
				++n;
			}
		}

		average_edge_length /= n;

		logger().info("hmin: {}", min_edge_length);
		logger().info("hmax: {}", mesh_size);
		logger().info("havg: {}", average_edge_length);
	}

	void OutStatsData::reset()
	{
		sigma_avg = 0;
		sigma_max = 0;
		sigma_min = 0;

		n_flipped = 0;
	}

	void OutStatsData::count_flipped_elements()
	{
		logger().info("Counting flipped elements...");
		const auto &els_tag = mesh->elements_tag();

		// flipped_elements.clear();
		for (size_t i = 0; i < gbases.size(); ++i)
		{
			if (mesh->is_polytope(i))
				continue;

			ElementAssemblyValues vals;
			if (!vals.is_geom_mapping_positive(mesh->is_volume(), gbases[i]))
			{
				++n_flipped;

				std::string type = "";
				switch (els_tag[i])
				{
				case ElementType::Simplex:
					type = "Simplex";
					break;
				case ElementType::RegularInteriorCube:
					type = "RegularInteriorCube";
					break;
				case ElementType::RegularBoundaryCube:
					type = "RegularBoundaryCube";
					break;
				case ElementType::SimpleSingularInteriorCube:
					type = "SimpleSingularInteriorCube";
					break;
				case ElementType::MultiSingularInteriorCube:
					type = "MultiSingularInteriorCube";
					break;
				case ElementType::SimpleSingularBoundaryCube:
					type = "SimpleSingularBoundaryCube";
					break;
				case ElementType::InterfaceCube:
					type = "InterfaceCube";
					break;
				case ElementType::MultiSingularBoundaryCube:
					type = "MultiSingularBoundaryCube";
					break;
				case ElementType::BoundaryPolytope:
					type = "BoundaryPolytope";
					break;
				case ElementType::InteriorPolytope:
					type = "InteriorPolytope";
					break;
				case ElementType::Undefined:
					type = "Undefined";
					break;
				}

				logger().error("element {} is flipped, type {}", i, type);
				throw "invalid mesh";
			}
		}

		logger().info(" done");

		// dynamic_cast<Mesh3D *>(mesh.get())->save({56}, 1, "mesh.HYBRID");

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));
	}

	void OutStatsData::compute_errors()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
		if (rhs.size() <= 0)
		{
			logger().error("Assemble the rhs first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		if (!args["output"]["advanced"]["compute_error"])
			return;

		int actual_dim = 1;
		if (!problem->is_scalar())
			actual_dim = mesh->dimension();

		igl::Timer timer;
		timer.start();
		logger().info("Computing errors...");
		using std::max;

		const int n_el = int(bases.size());

		MatrixXd v_exact, v_approx;
		MatrixXd v_exact_grad(0, 0), v_approx_grad;

		l2_err = 0;
		h1_err = 0;
		grad_max_err = 0;
		h1_semi_err = 0;
		linf_err = 0;
		lp_err = 0;
		// double pred_norm = 0;

		static const int p = 8;

		// Eigen::MatrixXd err_per_el(n_el, 5);
		ElementAssemblyValues vals;

		double tend;
		if (!args["time"].is_null())
		{
			tend = args["time"].value("tend", 1.0);
			if (tend <= 0)
				tend = 1;
		}
		else
			tend = 0;

		for (int e = 0; e < n_el; ++e)
		{
			// const auto &vals    = values[e];
			// const auto &gvalues = iso_parametric() ? values[e] : geom_values[e];

			if (iso_parametric())
				vals.compute(e, mesh->is_volume(), bases[e], bases[e]);
			else
				vals.compute(e, mesh->is_volume(), bases[e], geom_bases_[e]);

			if (problem->has_exact_sol())
			{
				problem->exact(vals.val, tend, v_exact);
				problem->exact_grad(vals.val, tend, v_exact_grad);
			}

			v_approx.resize(vals.val.rows(), actual_dim);
			v_approx.setZero();

			v_approx_grad.resize(vals.val.rows(), mesh->dimension() * actual_dim);
			v_approx_grad.setZero();

			const int n_loc_bases = int(vals.basis_values.size());

			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto &val = vals.basis_values[i];

				for (size_t ii = 0; ii < val.global.size(); ++ii)
				{
					for (int d = 0; d < actual_dim; ++d)
					{
						v_approx.col(d) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.val;
						v_approx_grad.block(0, d * val.grad_t_m.cols(), v_approx_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.grad_t_m;
					}
				}
			}

			const auto err = problem->has_exact_sol() ? (v_exact - v_approx).eval().rowwise().norm().eval() : (v_approx).eval().rowwise().norm().eval();
			const auto err_grad = problem->has_exact_sol() ? (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval() : (v_approx_grad).eval().rowwise().norm().eval();

			// for(long i = 0; i < err.size(); ++i)
			// errors.push_back(err(i));

			linf_err = max(linf_err, err.maxCoeff());
			grad_max_err = max(linf_err, err_grad.maxCoeff());

			// {
			// 	const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
			// 	const auto v0 = mesh3d.point(mesh3d.cell_vertex(e, 0));
			// 	const auto v1 = mesh3d.point(mesh3d.cell_vertex(e, 1));
			// 	const auto v2 = mesh3d.point(mesh3d.cell_vertex(e, 2));
			// 	const auto v3 = mesh3d.point(mesh3d.cell_vertex(e, 3));

			// 	Eigen::Matrix<double, 6, 3> ee;
			// 	ee.row(0) = v0 - v1;
			// 	ee.row(1) = v1 - v2;
			// 	ee.row(2) = v2 - v0;

			// 	ee.row(3) = v0 - v3;
			// 	ee.row(4) = v1 - v3;
			// 	ee.row(5) = v2 - v3;

			// 	Eigen::Matrix<double, 6, 1> en = ee.rowwise().norm();

			// 	// Eigen::Matrix<double, 3*4, 1> alpha;
			// 	// alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
			// 	// alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
			// 	// alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
			// 	// alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

			// 	const double S = (ee.row(0).cross(ee.row(1)).norm() + ee.row(0).cross(ee.row(4)).norm() + ee.row(4).cross(ee.row(1)).norm() + ee.row(2).cross(ee.row(5)).norm()) / 2;
			// 	const double V = std::abs(ee.row(3).dot(ee.row(2).cross(-ee.row(0))))/6;
			// 	const double rho = 3 * V / S;
			// 	const double hp = en.maxCoeff();
			// 	const int pp = disc_orders(e);
			// 	const int p_ref = args["space"]["discr_order"];

			// 	err_per_el(e, 0) = err.mean();
			// 	err_per_el(e, 1) = err.maxCoeff();
			// 	err_per_el(e, 2) = std::pow(hp, pp+1)/(rho/hp); // /std::pow(average_edge_length, p_ref+1) * (sqrt(6)/12);
			// 	err_per_el(e, 3) = rho/hp;
			// 	err_per_el(e, 4) = (vals.det.array() * vals.quadrature.weights.array()).sum();

			// 	// pred_norm += (pow(std::pow(hp, pp+1)/(rho/hp),p) * vals.det.array() * vals.quadrature.weights.array()).sum();
			// }

			l2_err += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			h1_err += (err_grad.array() * err_grad.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(p) * vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		h1_semi_err = sqrt(fabs(h1_err));
		h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
		l2_err = sqrt(fabs(l2_err));

		lp_err = pow(fabs(lp_err), 1. / p);

		// pred_norm = pow(fabs(pred_norm), 1./p);

		timer.stop();
		computing_errors_time = timer.getElapsedTime();
		logger().info(" took {}s", computing_errors_time);

		logger().info("-- L2 error: {}", l2_err);
		logger().info("-- Lp error: {}", lp_err);
		logger().info("-- H1 error: {}", h1_err);
		logger().info("-- H1 semi error: {}", h1_semi_err);
		// logger().info("-- Perd norm: {}", pred_norm);

		logger().info("-- Linf error: {}", linf_err);
		logger().info("-- grad max error: {}", grad_max_err);

		logger().info("total time: {}s", (building_basis_time + assembling_stiffness_mat_time + solving_time));

		// {
		// 	std::ofstream out("errs.txt");
		// 	out<<err_per_el;
		// 	out.close();
		// }
	}

	void OutStatsData::compute_mesh_stats()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		bases.clear();
		pressure_bases.clear();
		geom_bases_.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		simplex_count = 0;
		regular_count = 0;
		regular_boundary_count = 0;
		simple_singular_count = 0;
		multi_singular_count = 0;
		boundary_count = 0;
		non_regular_boundary_count = 0;
		non_regular_count = 0;
		undefined_count = 0;
		multi_singular_boundary_count = 0;

		const auto &els_tag = mesh->elements_tag();

		mesh->prepare_mesh();

		for (size_t i = 0; i < els_tag.size(); ++i)
		{
			const ElementType type = els_tag[i];

			switch (type)
			{
			case ElementType::Simplex:
				simplex_count++;
				break;
			case ElementType::RegularInteriorCube:
				regular_count++;
				break;
			case ElementType::RegularBoundaryCube:
				regular_boundary_count++;
				break;
			case ElementType::SimpleSingularInteriorCube:
				simple_singular_count++;
				break;
			case ElementType::MultiSingularInteriorCube:
				multi_singular_count++;
				break;
			case ElementType::SimpleSingularBoundaryCube:
				boundary_count++;
				break;
			case ElementType::InterfaceCube:
			case ElementType::MultiSingularBoundaryCube:
				multi_singular_boundary_count++;
				break;
			case ElementType::BoundaryPolytope:
				non_regular_boundary_count++;
				break;
			case ElementType::InteriorPolytope:
				non_regular_count++;
				break;
			case ElementType::Undefined:
				undefined_count++;
				break;
			}
		}

		logger().info("simplex_count: \t{}", simplex_count);
		logger().info("regular_count: \t{}", regular_count);
		logger().info("regular_boundary_count: \t{}", regular_boundary_count);
		logger().info("simple_singular_count: \t{}", simple_singular_count);
		logger().info("multi_singular_count: \t{}", multi_singular_count);
		logger().info("boundary_count: \t{}", boundary_count);
		logger().info("multi_singular_boundary_count: \t{}", multi_singular_boundary_count);
		logger().info("non_regular_count: \t{}", non_regular_count);
		logger().info("non_regular_boundary_count: \t{}", non_regular_boundary_count);
		logger().info("undefined_count: \t{}", undefined_count);
		logger().info("total count:\t {}", mesh->n_elements());
	}
} // namespace polyfem::output