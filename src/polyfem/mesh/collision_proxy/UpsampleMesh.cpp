#include "UpsampleMesh.hpp"

#include <polyfem/utils/GeometryUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/remove_duplicate_vertices.h>
#include <igl/barycentric_coordinates.h>
#ifdef POLYFEM_WITH_TRIANGLE
#include <igl/triangle/triangulate.h>
#endif

namespace polyfem::mesh
{
	namespace
	{
		Eigen::MatrixXd sample_triangle(
			const VectorNd &a,
			const VectorNd &b,
			const VectorNd &c,
			const Eigen::MatrixXd &coords)
		{
			// c
			// | \
    		// a--b
			Eigen::MatrixXd V(coords.rows(), a.size());
			for (int i = 0; i < coords.rows(); i++)
			{
				V.row(i) = coords(i, 0) * (b - a) + coords(i, 1) * (c - a) + a;
			}
			return V;
		}
	} // namespace

	void stitch_mesh(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out,
		const double epsilon)
	{
		std::vector<Eigen::Triplet<double>> _, __;
		stitch_mesh(V, F, _, V_out, F_out, __, epsilon);
	}

	void stitch_mesh(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::vector<Eigen::Triplet<double>> &W,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out,
		std::vector<Eigen::Triplet<double>> &W_out,
		const double epsilon)
	{
		/// indices: #V_out by 1 list of indices so V_out = V(indices,:)
		/// inverse: #V by 1 list of indices so V = V_out(inverse,:)
		Eigen::VectorXi indices, inverse;
		igl::remove_duplicate_vertices(
			V, F, epsilon, V_out, indices, inverse, F_out);
		assert(indices.size() == V_out.rows());
		assert(inverse.size() == V.rows());

		// Find indices of vertices that were removed
		std::sort(indices.data(), indices.data() + indices.size());
		std::vector<int> removed_indices;
		removed_indices.reserve(V.rows() - indices.size());
		for (int i = 0, j = 0; i < V.rows(); i++)
		{
			if (j < indices.size() && indices(j) == i)
				j++;
			else
				removed_indices.push_back(i);
		}
		assert(removed_indices.size() == V.rows() - indices.size());
		assert(std::is_sorted(removed_indices.begin(), removed_indices.end()));

		// Filter out the weights that correspond to duplicate vertices
		W_out.clear();
		W_out.reserve(W.size());
		for (const Eigen::Triplet<double> &w : W)
		{
			if (!std::binary_search(removed_indices.begin(), removed_indices.end(), w.row()))
				W_out.emplace_back(inverse(w.row()), w.col(), w.value());
		}
	}

	double max_edge_length(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
	{
		double max_edge_length = 0;
		for (int i = 0; i < F.cols(); i++)
		{
			max_edge_length = std::max(
				max_edge_length,
				(V(F.col(i), Eigen::all) - V(F.col((i + 1) % F.cols()), Eigen::all))
					.rowwise()
					.norm()
					.maxCoeff());
		}
		return max_edge_length;
	}

	// Regular tessellation

	void regular_grid_triangle_barycentric_coordinates(
		const int n, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
	{
		const double delta = 1.0 / (n - 1);
		// map from(i, j) coordinates to vertex id
		Eigen::MatrixXd ij2v = Eigen::MatrixXd::Constant(n, n, -1);
		V.resize(n * (n + 1) / 2, 2);

		int vi = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n - i; j++)
			{
				ij2v(i, j) = vi;
				V.row(vi++) << i * delta, j * delta;
			}
		}
		assert(vi == V.rows());

		// Create triangulated faces
		F.resize((n - 1) * (n - 1), 3);
		int fi = 0;
		Eigen::Vector3i f;
		for (int i = 0; i < n - 1; i++)
		{
			for (int j = 0; j < n - 1; j++)
			{
				f << ij2v(i, j), ij2v(i + 1, j), ij2v(i, j + 1);
				if (f.x() >= 0 && f.y() >= 0 && f.z() >= 0)
				{
					F.row(fi++) = f;
				}

				f << ij2v(i + 1, j), ij2v(i + 1, j + 1), ij2v(i, j + 1);
				if (f.x() >= 0 && f.y() >= 0 && f.z() >= 0)
				{
					F.row(fi++) = f;
				}
			}
		}

		F.conservativeResize(fi, Eigen::NoChange);
	}

	void regular_grid_tessellation(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const double out_max_edge_length,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out)
	{
		// Add one because n is the number of edge vertices not edges segments
		const double in_max_edge_length = max_edge_length(V, F);
		const int n =
			std::max(1, int(std::ceil(in_max_edge_length / out_max_edge_length)))
			+ 1;

		Eigen::MatrixXd coords;
		Eigen::MatrixXi local_F;
		regular_grid_triangle_barycentric_coordinates(n, coords, local_F);

		Eigen::MatrixXd V_tmp(F.rows() * coords.rows(), V.cols());
		Eigen::MatrixXi F_tmp(F.rows() * local_F.rows(), F.cols());
		for (int i = 0; i < F.rows(); i++)
		{
			F_tmp.middleRows(i * local_F.rows(), local_F.rows()) =
				local_F.array() + i * coords.rows();
			V_tmp.middleRows(i * coords.rows(), coords.rows()) = sample_triangle(
				V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), coords);
		}

		stitch_mesh(V_tmp, F_tmp, V_out, F_out);
	}

	// ------------------------------------------------------------------------
	// Irregular tessellation
	// ------------------------------------------------------------------------

	Eigen::MatrixXd
	refine_edge(const VectorNd &a, const VectorNd &b, const double max_edge_length)
	{
		const int n(std::ceil((b - a).norm() / max_edge_length));
		Eigen::MatrixXd V(n + 1, a.size());
		for (int i = 0; i <= n; i++)
		{
			V.row(i) = (b - a) * (i / double(n)) + a;
		}
		return V;
	}

	void refine_triangle_edges(
		const VectorNd &a,
		const VectorNd &b,
		const VectorNd &c,
		const double max_edge_len,
		Eigen::MatrixXd &V,
		Eigen::MatrixXi &E)
	{
		const int Nab = std::ceil((b - a).norm() / max_edge_len);
		const int Nbc = std::ceil((c - b).norm() / max_edge_len);
		const int Nca = std::ceil((a - c).norm() / max_edge_len);
		V.resize(Nab + Nbc + Nca, a.size());
		V.topRows(Nab) = refine_edge(a, b, max_edge_len).topRows(Nab);
		V.middleRows(Nab, Nbc) = refine_edge(b, c, max_edge_len).topRows(Nbc);
		V.bottomRows(Nca) = refine_edge(c, a, max_edge_len).topRows(Nca);

		E.resize(V.rows(), 2);
		for (int i = 0; i < V.rows(); i++)
		{
			E.row(i) << i, (i + 1) % V.rows();
		}
	}

	void irregular_triangle(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c,
		const double max_edge_length,
		Eigen::MatrixXd &V,
		Eigen::MatrixXi &F)
	{
#ifdef POLYFEM_WITH_TRIANGLE
		const double p = 3.0 * max_edge_length / 2.0;
		const double max_area = sqrt(p * std::pow(p - max_edge_length, 3));

		Eigen::MatrixXi E;
		refine_triangle_edges(a, b, c, max_edge_length, V, E);

		//  Compute a rotation that aligns the z axis with the triangle normal
		const Eigen::Matrix3d R =
			Eigen::Quaterniond::FromTwoVectors(
				(b - a).cross(c - a).normalized(), Eigen::Vector3d::UnitZ())
				.toRotationMatrix();

		// Align the triangle with the z axis
		Eigen::MatrixXd V_2D = V * R.transpose();
		const double z = V_2D(0, 2); // Save the z-offset
		assert((abs(V_2D.col(2).array() - z) < 1e-10).all());
		V_2D.conservativeResize(Eigen::NoChange, 2); // Drop the z coordinate

		igl::triangle::triangulate(
			V_2D, E, Eigen::MatrixXi(), fmt::format("Ya{:f}qQ", max_area), V, F);

		V.conservativeResize(V.rows(), 3);
		V.col(2).setConstant(z); // Restore the z-offset
		V = V * R;               // Rotate back to the original orientation

		// TODO: IDK why there are zero-area faces
		int fi = 0;
		for (int i = 0; i < F.rows(); i++)
		{
			if (utils::triangle_area(V(F.row(i), Eigen::all)) > 1e-12)
			{
				F.row(fi++) = F.row(i);
			}
		}
		F.conservativeResize(fi, Eigen::NoChange);
#else
		log_and_throw_error("irregular_triangle(): POLYFEM_WITH_TRIANGLE is not enabled!");
#endif
	}

	void irregular_triangle_barycentric_coordinates(
		const Eigen::Vector3d &a,
		const Eigen::Vector3d &b,
		const Eigen::Vector3d &c,
		const double max_edge_length,
		Eigen::MatrixXd &UV,
		Eigen::MatrixXi &F)
	{
		Eigen::MatrixXd V;
		irregular_triangle(a, b, c, max_edge_length, V, F);

		// Convert the triangle vertices to barycentric coordinates
		UV.resize(V.rows(), 2);
		for (int i = 0; i < UV.rows(); i++)
		{
			Eigen::RowVector3d tmp;
			igl::barycentric_coordinates(
				V.row(i), a.transpose(), b.transpose(), c.transpose(), tmp);
			UV.row(i) = tmp.head<2>();
		}
	}

	void irregular_tessellation(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const double max_edge_length,
		Eigen::MatrixXd &V_out,
		Eigen::MatrixXi &F_out)
	{
		Eigen::MatrixXd V_tmp(0, V.cols());
		Eigen::MatrixXi F_tmp(0, F.cols());
		for (int i = 0; i < F.rows(); i++)
		{
			Eigen::MatrixXd local_V;
			Eigen::MatrixXi local_F;
			irregular_triangle(
				V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), max_edge_length,
				local_V, local_F);

			F_tmp.conservativeResize(F_tmp.rows() + local_F.rows(), Eigen::NoChange);
			F_tmp.bottomRows(local_F.rows()) = local_F.array() + V_tmp.rows();

			V_tmp.conservativeResize(V_tmp.rows() + local_V.rows(), Eigen::NoChange);
			V_tmp.bottomRows(local_V.rows()) = local_V;
		}

		stitch_mesh(V_tmp, F_tmp, V_out, F_out);
	}
} // namespace polyfem::mesh