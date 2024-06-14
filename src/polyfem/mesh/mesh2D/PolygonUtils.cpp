////////////////////////////////////////////////////////////////////////////////
#include "PolygonUtils.hpp"
#include <polyfem/utils/ClipperUtils.hpp>
#include <geogram/numerics/predicates.h>
#include <igl/barycenter.h>
#include <cassert>
////////////////////////////////////////////////////////////////////////////////

namespace
{

	inline GEO::Sign point_is_in_half_plane(
		const Eigen::RowVector2d &p, const Eigen::RowVector2d &q1, const Eigen::RowVector2d &q2)
	{
		return GEO::PCK::orient_2d(q1.data(), q2.data(), p.data());
	}

	inline bool intersect_segments(
		const Eigen::RowVector2d &p1, const Eigen::RowVector2d &p2,
		const Eigen::RowVector2d &q1, const Eigen::RowVector2d &q2,
		Eigen::RowVector2d &result)
	{

		Eigen::RowVector2d Vp = p2 - p1;
		Eigen::RowVector2d Vq = q2 - q1;
		Eigen::RowVector2d pq = q1 - p1;

		double a = Vp(0);
		double b = -Vq(0);
		double c = Vp(1);
		double d = -Vq(1);

		double delta = a * d - b * c;
		if (delta == 0.0)
		{
			return false;
		}

		double tp = (d * pq(0) - b * pq(1)) / delta;

		result << (1.0 - tp) * p1(0) + tp * p2(0),
			(1.0 - tp) * p1(1) + tp * p2(1);

		return true;
	}

} // anonymous namespace

// -----------------------------------------------------------------------------

void polyfem::mesh::clip_polygon_by_half_plane(const Eigen::MatrixXd &P_in,
											   const Eigen::RowVector2d &q1, const Eigen::RowVector2d &q2, Eigen::MatrixXd &P_out)
{
	using namespace GEO;
	assert(P_in.cols() == 2);
	std::vector<Eigen::RowVector2d> result;

	if (P_in.rows() == 0)
	{
		P_out.resize(0, 2);
		return;
	}

	if (P_in.rows() == 1)
	{
		if (point_is_in_half_plane(P_in.row(0), q1, q2))
		{
			P_out.resize(1, 2);
			P_out << P_in.row(0);
		}
		else
		{
			P_out.resize(0, 2);
		}
		return;
	}

	Eigen::RowVector2d prev_p = P_in.row(P_in.rows() - 1);
	Sign prev_status = point_is_in_half_plane(prev_p, q1, q2);

	for (unsigned int i = 0; i < P_in.rows(); ++i)
	{
		Eigen::RowVector2d p = P_in.row(i);
		Sign status = point_is_in_half_plane(p, q1, q2);
		if (status != prev_status && status != ZERO && prev_status != ZERO)
		{
			Eigen::RowVector2d intersect;
			if (intersect_segments(prev_p, p, q1, q2, intersect))
			{
				result.push_back(intersect);
			}
		}

		switch (status)
		{
		case NEGATIVE:
			break;
		case ZERO:
			result.push_back(p);
			break;
		case POSITIVE:
			result.push_back(p);
			break;
		default:
			break;
		}

		prev_p = p;
		prev_status = status;
	}

	P_out.resize((int)result.size(), 2);
	for (size_t i = 0; i < result.size(); ++i)
	{
		P_out.row((int)i) = result[i];
	}
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::compute_visibility_kernel(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV)
{
	assert(IV.cols() == 2 || IV.cols() == 3);

	// 1) Start from the bounding box of the input points
	Eigen::MatrixXd src, dst;
	const auto &minV = IV.colwise().minCoeff().array();
	const auto &maxV = IV.colwise().maxCoeff().array();
	src.resize(4, 2);
	src.row(0) << minV(0), minV(1);
	src.row(1) << maxV(0), minV(1);
	src.row(2) << maxV(0), maxV(1);
	src.row(3) << minV(0), maxV(1);
	// std::cout << IV << std::endl;
	// std::cout << minV << ' ' << maxV << std::endl;

	// 2) Clip by half planes until we are left with the kernel
	for (unsigned int i = 0; i < IV.rows(); ++i)
	{
		unsigned int j = ((i + 1) % (int)IV.rows());
		const Eigen::RowVector2d &q1 = IV.row(i).head<2>();
		const Eigen::RowVector2d &q2 = IV.row(j).head<2>();
		clip_polygon_by_half_plane(src, q1, q2, dst);
		std::swap(src, dst);
	}
	OV = src;
}

////////////////////////////////////////////////////////////////////////////////

bool polyfem::mesh::is_star_shaped(const Eigen::MatrixXd &IV, Eigen::RowVector3d &bary)
{
	Eigen::MatrixXd OV;
	compute_visibility_kernel(IV, OV);
	if (OV.rows() == 0)
	{
		return false;
	}
	else
	{
		Eigen::MatrixXi F(1, OV.rows());
		for (int i = 0; i < OV.rows(); ++i)
		{
			F(0, i) = i;
		}
		Eigen::MatrixXd BC;
		igl::barycenter(OV, F, BC);
		// std::cout << BC.rows() << 'x' << BC.cols() << std::endl;
		// std::cout << BC << std::endl;
		int n = std::min(3, (int)BC.cols());
		bary.setZero();
		bary.head(n) = BC.row(0).head(n);
		return true;
	}
}

// -----------------------------------------------------------------------------

void polyfem::mesh::offset_polygon(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, double eps)
{
#ifdef POLYFEM_WITH_CLIPPER
	using namespace ClipperLib;
	using namespace polyfem::utils;

	// Convert input polygon to integer grid
	ClipperOffset co;
	co.AddPath(PolygonClipping::toClipperPolygon(IV), jtSquare, etClosedPolygon);

	// Compute offset in the integer grid
	Paths solution;
	co.Execute(solution, cInt(eps * DOUBLE_TO_INT_SCALE_FACTOR));
	assert(solution.size() == 1);

	// Convert back to double
	OV = PolygonClipping::fromClipperPolygon(solution.front());
#else
	throw std::runtime_error("Compile with clipper!");
#endif
}

////////////////////////////////////////////////////////////////////////////////

namespace
{

	typedef Eigen::RowVector2d Vec2d;

	double inline det(Vec2d u, Vec2d v)
	{
		return u[0] * v[1] - u[1] * v[0];
	}

	// Return true iff [a,b] intersects [c,d], and store the intersection in ans
	bool intersect_segment(const Vec2d &a, const Vec2d &b, Vec2d c, Vec2d d, Vec2d &ans)
	{
		const double eps = 1e-10; // small epsilon for numerical precision
		double x = det(c - a, d - c);
		double y = det(b - a, a - c);
		double z = det(b - a, d - c);
		if (std::abs(z) < eps || x * z < 0 || x * z > z * z || y * z < 0 || y * z > z * z)
			return false;
		ans = c + (d - c) * y / z;
		return true;
	}

	bool is_point_inside(const Eigen::MatrixXd &poly, const Vec2d &outside, const Vec2d &query)
	{
		int n = poly.rows();
		bool tmp, ans = false;
		for (long i = 0; i < poly.rows(); ++i)
		{
			Vec2d m; // Coordinates of intersection point
			tmp = intersect_segment(query, outside, poly.row(i), poly.row((i + 1) % n), m);
			ans = (ans != tmp);
		}
		return ans;
	}

} // anonymous namespace

// -----------------------------------------------------------------------------

int polyfem::mesh::is_inside(const Eigen::MatrixXd &IV, const Eigen::MatrixXd &Q, std::vector<bool> &inside)
{
	assert(IV.cols() == 2);
	Eigen::RowVector2d minV = IV.colwise().minCoeff().array();
	Eigen::RowVector2d maxV = IV.colwise().maxCoeff().array();
	Eigen::RowVector2d center = 0.5 * (maxV + minV);
	Eigen::RowVector2d outside(2.0 * minV(0) - center(0), center(1));
	inside.resize(Q.rows());
	int num_inside = 0;
	for (long i = 0; i < Q.rows(); ++i)
	{
		inside[i] = is_point_inside(IV, outside, Q.row(i));
		if (inside[i])
		{
			++num_inside;
		}
	}
	return num_inside;
}

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::sample_polygon(const Eigen::MatrixXd &poly, int num_samples, Eigen::MatrixXd &S)
{
	assert(poly.rows() >= 2);
	int n = poly.rows();

	auto length = [&](int i) {
		return (poly.row(i) - poly.row((i + 1) % n)).norm();
	};

	// Step 1: compute starting edge + total length of the polygon
	int i0 = 0;
	double max_length = 0;
	double total_length = 0;
	for (int i = 0; i < n; ++i)
	{
		double len = length(i);
		total_length += len;
		if (len > max_length)
		{
			max_length = len;
			i0 = i;
		}
	}

	// Step 2: place a sample at regular intervals along the polygon,
	// starting from the middle of edge i0
	S.resize(num_samples, poly.cols());
	double spacing = total_length / num_samples; // sampling distance
	double offset = length(i0) / 2.0;            // distance from first sample to vertex i0
	double distance_to_next = length(i0);
	for (int s = 0, i = i0; s < num_samples; ++s)
	{
		double distance_to_sample = s * spacing + offset; // next sample length
		while (distance_to_sample > distance_to_next)
		{
			i = (i + 1) % n;
			distance_to_next += length(i);
		}
		double t = (distance_to_next - distance_to_sample) / length(i);
		S.row(s) = t * poly.row(i) + (1.0 - t) * poly.row((i + 1) % n);
	}
}
