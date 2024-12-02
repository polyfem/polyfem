#include "Evaluator.hpp"

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/quadrature/HexQuadrature.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>

#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/Jacobian.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/utils/Logger.hpp>

#include <igl/AABB.h>
#include <igl/per_face_normals.h>

namespace polyfem::io
{
	using namespace mesh;
	using namespace assembler;
	using namespace basis;

	namespace
	{
		void flattened_tensor_coeffs(const Eigen::MatrixXd &S, Eigen::MatrixXd &X)
		{
			if (S.cols() == 4)
			{
				X.resize(S.rows(), 3);
				X.col(0) = S.col(0);
				X.col(1) = S.col(3);
				X.col(2) = S.col(1);
			}
			else if (S.cols() == 9)
			{
				// [S11, S22, S33, S12, S13, S23]
				X.resize(S.rows(), 6);
				X.col(0) = S.col(0);
				X.col(1) = S.col(4);
				X.col(2) = S.col(8);
				X.col(3) = S.col(1);
				X.col(4) = S.col(2);
				X.col(5) = S.col(5);
			}
			else
			{
				logger().error("Invalid tensor dimensions.");
			}
		}
	} // namespace

	void Evaluator::interpolate_boundary_function(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::MatrixXd &pts,
		const Eigen::MatrixXi &faces,
		const Eigen::MatrixXd &fun,
		const bool compute_avg,
		Eigen::MatrixXd &result)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}
		assert(mesh.is_volume());

		const Mesh3D &mesh3d = dynamic_cast<const Mesh3D &>(mesh);

		Eigen::MatrixXd points, uv;
		Eigen::VectorXd weights;

		int actual_dim = 1;
		if (!is_problem_scalar)
			actual_dim = 3;

		igl::AABB<Eigen::MatrixXd, 3> tree;
		tree.init(pts, faces);

		result.resize(faces.rows(), actual_dim);
		result.setConstant(std::numeric_limits<double>::quiet_NaN());

		int counter = 0;

		for (int e = 0; e < mesh3d.n_elements(); ++e)
		{
			const basis::ElementBases &gbs = gbases[e];
			const basis::ElementBases &bs = bases[e];

			for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
			{
				const int face_id = mesh3d.cell_face(e, lf);
				if (!mesh3d.is_boundary_face(face_id))
					continue;

				if (mesh3d.is_simplex(e))
					utils::BoundarySampler::quadrature_for_tri_face(lf, 4, face_id, mesh3d, uv, points, weights);
				else if (mesh3d.is_cube(e))
					utils::BoundarySampler::quadrature_for_quad_face(lf, 4, face_id, mesh3d, uv, points, weights);
				else
					assert(false);

				ElementAssemblyValues vals;
				vals.compute(e, true, points, bs, gbs);
				RowVectorNd loc_val(actual_dim);
				loc_val.setZero();

				// UIEvaluator::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

				// const auto nodes = bs.local_nodes_for_primitive(face_id, mesh3d);

				// for(long n = 0; n < nodes.size(); ++n)
				for (size_t j = 0; j < bs.bases.size(); ++j)
				{
					// const auto &b = bs.bases[nodes(n)];
					// const AssemblyValues &v = vals.basis_values[nodes(n)];
					const AssemblyValues &v = vals.basis_values[j];
					for (int d = 0; d < actual_dim; ++d)
					{
						for (size_t g = 0; g < v.global.size(); ++g)
						{
							loc_val(d) += (v.global[g].val * v.val.array() * fun(v.global[g].index * actual_dim + d) * weights.array()).sum();
						}
					}
				}

				int I;
				Eigen::RowVector3d C;
				const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

				const double dist = tree.squared_distance(pts, faces, bary, I, C);
				assert(dist < 1e-16);

				assert(std::isnan(result(I, 0)));
				if (compute_avg)
					result.row(I) = loc_val / weights.sum();
				else
					result.row(I) = loc_val;
				++counter;
			}
		}

		assert(counter == result.rows());
	}

	void Evaluator::average_grad_based_function(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const assembler::Assembler &assembler,
		const utils::RefElementSampler &sampler,
		const double t,
		const int n_points,
		const Eigen::MatrixXd &fun,
		std::vector<assembler::Assembler::NamedMatrix> &result_scalar,
		std::vector<assembler::Assembler::NamedMatrix> &result_tensor,
		const bool use_sampler,
		const bool boundary_only)
	{
		result_scalar.clear();
		result_tensor.clear();

		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}
		if (is_problem_scalar)
		{
			logger().error("Define a tensor problem!");
			return;
		}

		assert(!is_problem_scalar);
		const int actual_dim = mesh.dimension();

		std::vector<Eigen::MatrixXd> avg_scalar;

		Eigen::MatrixXd areas(n_bases, 1);
		areas.setZero();

		std::vector<std::pair<std::string, Eigen::MatrixXd>> tmp_s;
		Eigen::MatrixXd local_val;

		ElementAssemblyValues vals;
		for (int i = 0; i < int(bases.size()); ++i)
		{
			const ElementBases &bs = bases[i];
			const ElementBases &gbs = gbases[i];
			Eigen::MatrixXd local_pts;

			if (mesh.is_simplex(i))
			{
				if (mesh.dimension() == 3)
					autogen::p_nodes_3d(disc_orders(i), local_pts);
				else
					autogen::p_nodes_2d(disc_orders(i), local_pts);
			}
			else if (mesh.is_cube(i))
			{
				if (mesh.dimension() == 3)
					autogen::q_nodes_3d(disc_orders(i), local_pts);
				else
					autogen::q_nodes_2d(disc_orders(i), local_pts);
			}
			else
			{
				// not supported for polys
				continue;
			}

			vals.compute(i, actual_dim == 3, bases[i], gbases[i]);
			const quadrature::Quadrature &quadrature = vals.quadrature;
			const double area = (vals.det.array() * quadrature.weights.array()).sum();

			assembler.compute_scalar_value(OutputData(t, i, bs, gbs, local_pts, fun), tmp_s);

			// assembler.compute_tensor_value(i, bs, gbs, local_pts, fun, local_val);
			// MatrixXd avg_tensor(n_points * actual_dim*actual_dim, 1);

			for (size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];
				if (b.global().size() > 1)
					continue;

				auto &global = b.global().front();
				areas(global.index) += area;
			}

			if (avg_scalar.empty())
			{
				avg_scalar.resize(tmp_s.size());
				for (auto &m : avg_scalar)
				{
					m.resize(n_bases, 1);
					m.setZero();
				}
			}

			for (int k = 0; k < tmp_s.size(); ++k)
			{
				local_val = tmp_s[k].second;

				for (size_t j = 0; j < bs.bases.size(); ++j)
				{
					const Basis &b = bs.bases[j];
					if (b.global().size() > 1)
						continue;

					auto &global = b.global().front();
					avg_scalar[k](global.index) += local_val(j) * area;
				}
			}
		}

		for (auto &m : avg_scalar)
		{
			m.array() /= areas.array();
		}

		result_scalar.resize(tmp_s.size());
		for (int k = 0; k < tmp_s.size(); ++k)
		{
			result_scalar[k].first = tmp_s[k].first;
			interpolate_function(mesh, 1, bases, disc_orders, polys, polys_3d, sampler, n_points,
								 avg_scalar[k], result_scalar[k].second, use_sampler, boundary_only);
		}
		// interpolate_function(n_points, actual_dim*actual_dim, bases, avg_tensor, result_tensor, boundary_only);
	}

	void Evaluator::compute_stress_at_quadrature_points(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXi &disc_orders,
		const assembler::Assembler &assembler,
		const Eigen::MatrixXd &fun,
		const double t,
		Eigen::MatrixXd &result,
		Eigen::VectorXd &von_mises)
	{
		// if (!mesh)
		// {
		// 	logger().error("Load the mesh first!");
		// 	return;
		// }
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}
		if (is_problem_scalar)
		{
			logger().error("Define a tensor problem!");
			return;
		}

		const int actual_dim = mesh.dimension();
		assert(!is_problem_scalar);

		Eigen::MatrixXd local_val, local_stress, local_mises;

		int num_quadr_pts = 0;
		result.resize(disc_orders.sum(), actual_dim == 2 ? 3 : 6);
		result.setZero();
		von_mises.resize(disc_orders.sum(), 1);
		von_mises.setZero();
		for (int e = 0; e < mesh.n_elements(); ++e)
		{
			// Compute quadrature points for element
			quadrature::Quadrature quadr;
			if (mesh.is_simplex(e))
			{
				if (mesh.is_volume())
				{
					quadrature::TetQuadrature f;
					f.get_quadrature(disc_orders(e), quadr);
				}
				else
				{
					quadrature::TriQuadrature f;
					f.get_quadrature(disc_orders(e), quadr);
				}
			}
			else if (mesh.is_cube(e))
			{
				if (mesh.is_volume())
				{
					quadrature::HexQuadrature f;
					f.get_quadrature(disc_orders(e), quadr);
				}
				else
				{
					quadrature::QuadQuadrature f;
					f.get_quadrature(disc_orders(e), quadr);
				}
			}
			else
			{
				continue;
			}

			std::vector<std::pair<std::string, Eigen::MatrixXd>> tmp_s, tmp_t;

			assembler.compute_scalar_value(OutputData(t, e, bases[e], gbases[e], quadr.points, fun), tmp_s);
			assembler.compute_tensor_value(OutputData(t, e, bases[e], gbases[e], quadr.points, fun), tmp_t);

			local_mises = tmp_s[0].second;
			local_val = tmp_t[0].second;

			if (num_quadr_pts + local_val.rows() >= result.rows())
			{
				result.conservativeResize(
					std::max(num_quadr_pts + local_val.rows() + 1, 2 * result.rows()),
					result.cols());
				von_mises.conservativeResize(result.rows(), von_mises.cols());
			}
			flattened_tensor_coeffs(local_val, local_stress);
			result.block(num_quadr_pts, 0, local_stress.rows(), local_stress.cols()) = local_stress;
			von_mises.block(num_quadr_pts, 0, local_mises.rows(), local_mises.cols()) = local_mises;
			num_quadr_pts += local_val.rows();
		}
		result.conservativeResize(num_quadr_pts, result.cols());
		von_mises.conservativeResize(num_quadr_pts, von_mises.cols());
	}

	void Evaluator::interpolate_function(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const utils::RefElementSampler &sampler,
		const int n_points,
		const Eigen::MatrixXd &fun,
		Eigen::MatrixXd &result,
		const bool use_sampler,
		const bool boundary_only)
	{
		int actual_dim = 1;
		if (!is_problem_scalar)
			actual_dim = mesh.dimension();
		interpolate_function(mesh, actual_dim, bases, disc_orders,
							 polys, polys_3d, sampler, n_points,
							 fun, result, use_sampler, boundary_only);
	}

	void Evaluator::mark_flipped_cells(
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &gbasis,
		const std::vector<basis::ElementBases> &basis,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const utils::RefElementSampler &sampler,
		const int n_points,
		const Eigen::MatrixXd &fun,
		Eigen::Vector<bool, -1> &result,
		const bool use_sampler,
		const bool boundary_only)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		result.setZero(n_points);

		int index = 0;

		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;

		const auto invalidList = utils::count_invalid(mesh.dimension(), basis, gbasis, fun);

		for (int i = 0; i < int(basis.size()); ++i)
		{
			const ElementBases &bs = basis[i];
			Eigen::MatrixXd local_pts;

			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			if (use_sampler)
			{
				if (mesh.is_simplex(i))
					local_pts = sampler.simplex_points();
				else if (mesh.is_cube(i))
					local_pts = sampler.cube_points();
				else
				{
					if (mesh.is_volume())
						sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, local_pts, vis_faces_poly, vis_edges_poly);
					else
						sampler.sample_polygon(polys.at(i), local_pts, vis_faces_poly, vis_edges_poly);
				}
			}
			else
			{
				if (mesh.is_volume())
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_3d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_3d(disc_orders(i), local_pts);
					else
						continue;
				}
				else
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_2d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_2d(disc_orders(i), local_pts);
					else
						continue;
				}
			}

			if (std::find(invalidList.begin(), invalidList.end(), i) != invalidList.end())
				result.segment(index, local_pts.rows()).array() = true;
			index += local_pts.rows();
		}
	}

	void Evaluator::interpolate_function(
		const mesh::Mesh &mesh,
		const int actual_dim,
		const std::vector<basis::ElementBases> &basis,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const utils::RefElementSampler &sampler,
		const int n_points,
		const Eigen::MatrixXd &fun,
		Eigen::MatrixXd &result,
		const bool use_sampler,
		const bool boundary_only)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		std::vector<AssemblyValues> tmp;

		result.resize(n_points, actual_dim);

		int index = 0;

		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;

		for (int i = 0; i < int(basis.size()); ++i)
		{
			const ElementBases &bs = basis[i];
			Eigen::MatrixXd local_pts;

			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			if (use_sampler)
			{
				if (mesh.is_simplex(i))
					local_pts = sampler.simplex_points();
				else if (mesh.is_cube(i))
					local_pts = sampler.cube_points();
				else
				{
					if (mesh.is_volume())
						sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, local_pts, vis_faces_poly, vis_edges_poly);
					else
						sampler.sample_polygon(polys.at(i), local_pts, vis_faces_poly, vis_edges_poly);
				}
			}
			else
			{
				if (mesh.is_volume())
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_3d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_3d(disc_orders(i), local_pts);
					else
						continue;
				}
				else
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_2d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_2d(disc_orders(i), local_pts);
					else
						continue;
				}
			}

			Eigen::MatrixXd local_res = Eigen::MatrixXd::Zero(local_pts.rows(), actual_dim);
			bs.evaluate_bases(local_pts, tmp);
			for (size_t j = 0; j < bs.bases.size(); ++j)
			{
				const Basis &b = bs.bases[j];

				for (int d = 0; d < actual_dim; ++d)
				{
					for (size_t ii = 0; ii < b.global().size(); ++ii)
						local_res.col(d) += b.global()[ii].val * tmp[j].val * fun(b.global()[ii].index * actual_dim + d);
				}
			}

			result.block(index, 0, local_res.rows(), actual_dim) = local_res;
			index += local_res.rows();
		}
	}

	void Evaluator::interpolate_at_local_vals(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const int el_index,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &fun,
		Eigen::MatrixXd &result,
		Eigen::MatrixXd &result_grad)
	{
		int actual_dim = 1;
		if (!is_problem_scalar)
			actual_dim = mesh.dimension();
		interpolate_at_local_vals(mesh, actual_dim, bases, gbases, el_index,
								  local_pts, fun, result, result_grad);
	}

	void Evaluator::interpolate_at_local_vals(
		const mesh::Mesh &mesh,
		const int actual_dim,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const int el_index,
		const Eigen::MatrixXd &local_pts,
		const Eigen::MatrixXd &fun,
		Eigen::MatrixXd &result,
		Eigen::MatrixXd &result_grad)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		assert(local_pts.cols() == mesh.dimension());
		assert(fun.cols() == 1);

		const ElementBases &gbs = gbases[el_index];
		const ElementBases &bs = bases[el_index];

		ElementAssemblyValues vals;
		vals.compute(el_index, mesh.is_volume(), local_pts, bs, gbs);

		result.resize(vals.val.rows(), actual_dim);
		result.setZero();

		result_grad.resize(vals.val.rows(), mesh.dimension() * actual_dim);
		result_grad.setZero();

		const int n_loc_bases = int(vals.basis_values.size());

		for (int i = 0; i < n_loc_bases; ++i)
		{
			const auto &val = vals.basis_values[i];

			for (size_t ii = 0; ii < val.global.size(); ++ii)
			{
				for (int d = 0; d < actual_dim; ++d)
				{
					result.col(d) += val.global[ii].val * fun(val.global[ii].index * actual_dim + d) * val.val;
					result_grad.block(0, d * val.grad_t_m.cols(), result_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * fun(val.global[ii].index * actual_dim + d) * val.grad_t_m;
				}
			}
		}
	}

	void Evaluator::interpolate_at_local_vals(const int el_index, const int dim, const int actual_dim, const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, Eigen::MatrixXd &result_grad)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		assert(fun.cols() == 1);

		result.resize(vals.val.rows(), actual_dim);
		result.setZero();

		result_grad.resize(vals.val.rows(), dim * actual_dim);
		result_grad.setZero();

		const int n_loc_bases = int(vals.basis_values.size());

		for (int i = 0; i < n_loc_bases; ++i)
		{
			const auto &val = vals.basis_values[i];

			for (size_t ii = 0; ii < val.global.size(); ++ii)
			{
				for (int d = 0; d < actual_dim; ++d)
				{
					result.col(d) += val.global[ii].val * fun(val.global[ii].index * actual_dim + d) * val.val;
					result_grad.block(0, d * val.grad_t_m.cols(), result_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * fun(val.global[ii].index * actual_dim + d) * val.grad_t_m;
				}
			}
		}
	}

	bool Evaluator::check_scalar_value(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const assembler::Assembler &assembler,
		const utils::RefElementSampler &sampler,
		const Eigen::MatrixXd &fun,
		const double t,
		const bool use_sampler,
		const bool boundary_only)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return true;
		}

		assert(!is_problem_scalar);

		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;

		std::vector<std::pair<std::string, Eigen::MatrixXd>> tmp_s;

		for (int i = 0; i < int(bases.size()); ++i)
		{
			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			const ElementBases &bs = bases[i];
			const ElementBases &gbs = gbases[i];
			Eigen::MatrixXd local_pts;

			if (use_sampler)
			{
				if (mesh.is_simplex(i))
					local_pts = sampler.simplex_points();
				else if (mesh.is_cube(i))
					local_pts = sampler.cube_points();
				else
				{
					if (mesh.is_volume())
						sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, local_pts, vis_faces_poly, vis_edges_poly);
					else
						sampler.sample_polygon(polys.at(i), local_pts, vis_faces_poly, vis_edges_poly);
				}
			}
			else
			{
				if (mesh.is_volume())
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_3d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_3d(disc_orders(i), local_pts);
					else
						continue;
				}
				else
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_2d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_2d(disc_orders(i), local_pts);
					else
						continue;
				}
			}

			assembler.compute_scalar_value(OutputData(t, i, bs, gbs, local_pts, fun), tmp_s);

			for (const auto &s : tmp_s)
				if (std::isnan(s.second.norm()))
					return false;
		}

		return true;
	}

	void Evaluator::compute_scalar_value(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const assembler::Assembler &assembler,
		const utils::RefElementSampler &sampler,
		const int n_points,
		const Eigen::MatrixXd &fun,
		const double t,
		std::vector<assembler::Assembler::NamedMatrix> &result,
		const bool use_sampler,
		const bool boundary_only)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		result.clear();

		assert(!is_problem_scalar);

		int index = 0;

		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;
		std::vector<std::pair<std::string, Eigen::MatrixXd>> tmp_s;

		for (int i = 0; i < int(bases.size()); ++i)
		{
			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			const ElementBases &bs = bases[i];
			const ElementBases &gbs = gbases[i];
			Eigen::MatrixXd local_pts;

			if (use_sampler)
			{
				if (mesh.is_simplex(i))
					local_pts = sampler.simplex_points();
				else if (mesh.is_cube(i))
					local_pts = sampler.cube_points();
				else
				{
					if (mesh.is_volume())
						sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, local_pts, vis_faces_poly, vis_edges_poly);
					else
						sampler.sample_polygon(polys.at(i), local_pts, vis_faces_poly, vis_edges_poly);
				}
			}
			else
			{
				if (mesh.is_volume())
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_3d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_3d(disc_orders(i), local_pts);
					else
						continue;
				}
				else
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_2d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_2d(disc_orders(i), local_pts);
					else
						continue;
				}
			}

			assembler.compute_scalar_value(OutputData(t, i, bs, gbs, local_pts, fun), tmp_s);

			if (result.empty())
			{
				result.resize(tmp_s.size());
				for (int k = 0; k < tmp_s.size(); ++k)
				{
					result[k].first = tmp_s[k].first;
					result[k].second.resize(n_points, 1);
				}
			}

			for (int k = 0; k < tmp_s.size(); ++k)
			{
				assert(local_pts.rows() == tmp_s[k].second.rows());
				result[k].second.block(index, 0, tmp_s[k].second.rows(), 1) = tmp_s[k].second;
			}
			index += local_pts.rows();
		}
	}

	void Evaluator::compute_tensor_value(
		const mesh::Mesh &mesh,
		const bool is_problem_scalar,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::VectorXi &disc_orders,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const assembler::Assembler &assembler,
		const utils::RefElementSampler &sampler,
		const int n_points,
		const Eigen::MatrixXd &fun,
		const double t,
		std::vector<assembler::Assembler::NamedMatrix> &result,
		const bool use_sampler,
		const bool boundary_only)
	{
		if (fun.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		result.clear();

		const int actual_dim = mesh.dimension();
		assert(!is_problem_scalar);

		int index = 0;

		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;
		std::vector<std::pair<std::string, Eigen::MatrixXd>> tmp_t;

		for (int i = 0; i < int(bases.size()); ++i)
		{
			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			const ElementBases &bs = bases[i];
			const ElementBases &gbs = gbases[i];
			Eigen::MatrixXd local_pts;

			if (use_sampler)
			{
				if (mesh.is_simplex(i))
					local_pts = sampler.simplex_points();
				else if (mesh.is_cube(i))
					local_pts = sampler.cube_points();
				else
				{
					if (mesh.is_volume())
						sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, local_pts, vis_faces_poly, vis_edges_poly);
					else
						sampler.sample_polygon(polys.at(i), local_pts, vis_faces_poly, vis_edges_poly);
				}
			}
			else
			{
				if (mesh.is_volume())
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_3d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_3d(disc_orders(i), local_pts);
					else
						continue;
				}
				else
				{
					if (mesh.is_simplex(i))
						autogen::p_nodes_2d(disc_orders(i), local_pts);
					else if (mesh.is_cube(i))
						autogen::q_nodes_2d(disc_orders(i), local_pts);
					else
						continue;
				}
			}

			assembler.compute_tensor_value(OutputData(t, i, bs, gbs, local_pts, fun), tmp_t);

			if (result.empty())
			{
				result.resize(tmp_t.size());
				for (int k = 0; k < tmp_t.size(); ++k)
				{
					result[k].first = tmp_t[k].first;
					result[k].second.resize(n_points, actual_dim * actual_dim);
				}
			}

			for (int k = 0; k < tmp_t.size(); ++k)
			{
				assert(local_pts.rows() == tmp_t[k].second.rows());
				result[k].second.block(index, 0, tmp_t[k].second.rows(), tmp_t[k].second.cols()) = tmp_t[k].second;
			}
			index += local_pts.rows();
		}
	}

	Eigen::MatrixXd Evaluator::get_bases_position(
		const int n_bases,
		const std::shared_ptr<mesh::MeshNodes> mesh_nodes)
	{
		Eigen::MatrixXd func;
		func.setZero(n_bases, mesh_nodes->node_position(0).size());

		for (int i = 0; i < n_bases; i++)
			func.row(i) = mesh_nodes->node_position(i);

		return func;
	}

	Eigen::MatrixXd Evaluator::generate_linear_field(
		const int n_bases,
		const std::shared_ptr<mesh::MeshNodes> mesh_nodes,
		const Eigen::MatrixXd &grad)
	{
		return utils::flatten(get_bases_position(n_bases, mesh_nodes) * grad.transpose());
	}

	Eigen::VectorXd Evaluator::integrate_function(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const Eigen::MatrixXd &fun,
		const int dim,
		const int actual_dim)
	{
		Eigen::VectorXd result;
		result.setZero(actual_dim);
		for (int e = 0; e < bases.size(); ++e)
		{
			ElementAssemblyValues vals;
			vals.compute(e, dim == 3, bases[e], gbases[e]);

			Eigen::MatrixXd u, grad_u;
			io::Evaluator::interpolate_at_local_vals(e, dim, actual_dim, vals, fun, u, grad_u);
			const quadrature::Quadrature &quadrature = vals.quadrature;
			Eigen::VectorXd da = vals.det.array() * quadrature.weights.array();
			result += u.transpose() * da;
		}

		return result;
	}
} // namespace polyfem::io
