////////////////////////////////////////////////////////////////////////////////
#include "PolygonUtils.hpp"
#include <geogram/numerics/predicates.h>
#include <igl/barycenter.h>
////////////////////////////////////////////////////////////////////////////////

namespace {

inline GEO::Sign point_is_in_half_plane(
	const Eigen::RowVector2d &p, const Eigen::RowVector2d &q1, const Eigen::RowVector2d &q2)
{
	return GEO::PCK::orient_2d(q1.data(), q2.data(), p.data());
}

inline bool intersect_segments(
	const Eigen::RowVector2d & p1, const Eigen::RowVector2d & p2,
	const Eigen::RowVector2d & q1, const Eigen::RowVector2d & q2,
	Eigen::RowVector2d & result)
{

	Eigen::RowVector2d Vp = p2 - p1;
	Eigen::RowVector2d Vq = q2 - q1;
	Eigen::RowVector2d pq = q1 - p1;

	double a =  Vp(0);
	double b = -Vq(0);
	double c =  Vp(1);
	double d = -Vq(1);

	double delta = a*d-b*c;
	if (delta == 0.0) {
		return false;
	}

	double tp = (d * pq(0) -b * pq(1)) / delta;

	result <<
		(1.0 - tp) * p1(0) + tp * p2(0),
		(1.0 - tp) * p1(1) + tp * p2(1);

	return true;
}

} // anonymous namespace

// -----------------------------------------------------------------------------

void poly_fem::clip_polygon_by_half_plane(const Eigen::MatrixXd &P_in,
	const Eigen::RowVector2d &q1, const Eigen::RowVector2d &q2, Eigen::MatrixXd &P_out)
{
	using namespace GEO;
	assert(P_in.cols() == 2);
	std::vector<Eigen::RowVector2d> result;

	if (P_in.rows() == 0) {
		P_out.resize(0, 2);
		return ;
	}

	if (P_in.rows() == 1) {
		if (point_is_in_half_plane(P_in.row(0), q1, q2)) {
			P_out.resize(1, 2);
			P_out << P_in.row(0);
		} else {
			P_out.resize(0, 2);
		}
		return;
	}

	Eigen::RowVector2d prev_p = P_in.row(P_in.rows() - 1);
	Sign prev_status = point_is_in_half_plane(prev_p, q1, q2);

	for (unsigned int i = 0; i < P_in.rows(); ++i) {
		Eigen::RowVector2d p = P_in.row(i);
		Sign status = point_is_in_half_plane(p, q1, q2);
		if (status != prev_status && status != ZERO && prev_status != ZERO) {
			Eigen::RowVector2d intersect;
			if (intersect_segments(prev_p, p, q1, q2, intersect)) {
				result.push_back(intersect) ;
			}
		}

		switch(status) {
		case NEGATIVE:
			break ;
		case ZERO:
			result.push_back(p) ;
			break ;
		case POSITIVE:
			result.push_back(p) ;
			break ;
		default:
			break;
		}

		prev_p = p ;
		prev_status = status ;
	}

	P_out.resize((int) result.size(), 2);
	for (size_t i = 0; i < result.size(); ++i) {
		P_out.row((int) i) = result[i];
	}
}

////////////////////////////////////////////////////////////////////////////////

void poly_fem::compute_visibility_kernel(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV) {
	assert(IV.cols() == 2 || IV.cols() == 3);

	// 1) Start from the bounding box of the input points
	Eigen::MatrixXd src, dst;
	const auto & minV = IV.colwise().minCoeff().array();
	const auto & maxV = IV.colwise().maxCoeff().array();
	src.resize(4, 2);
	src.row(0) << minV(0), minV(1);
	src.row(1) << maxV(0), minV(1);
	src.row(2) << maxV(0), maxV(1);
	src.row(3) << minV(0), maxV(1);
	// std::cout << IV << std::endl;
	// std::cout << minV << ' ' << maxV << std::endl;

	// 2) Clip by half planes until we are left with the kernel
	for (unsigned int i = 0; i < IV.rows(); ++i) {
		unsigned int j = ((i+1) % (int) IV.rows()) ;
		const Eigen::RowVector2d &q1 = IV.row(i).head<2>();
		const Eigen::RowVector2d &q2 = IV.row(j).head<2>();
		clip_polygon_by_half_plane(src, q1, q2, dst);
		std::swap(src, dst);
	}
	OV = src;
}

////////////////////////////////////////////////////////////////////////////////

bool poly_fem::is_star_shaped(const Eigen::MatrixXd &IV, Eigen::RowVector3d &bary) {
	Eigen::MatrixXd OV;
	compute_visibility_kernel(IV, OV);
	if (OV.rows() == 0) {
		return false;
	} else {
		Eigen::MatrixXi F(1, OV.rows());
		for (int  i = 0; i < OV.rows(); ++i) {
			F(0, i) = i;
		}
		Eigen::MatrixXd BC;
		igl::barycenter(OV, F, BC);
		// std::cout << BC.rows() << 'x' << BC.cols() << std::endl;
		// std::cout << BC << std::endl;
		int n = std::min(3, (int) BC.cols());
		bary.setZero();
		bary.head(n) = BC.row(0).head(n);
		return true;
	}
}
