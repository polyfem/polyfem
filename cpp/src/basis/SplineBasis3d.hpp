#ifndef SPLINE_BASIS_3D_HPP
#define SPLINE_BASIS_3D_HPP

#include "Mesh3D.hpp"
#include "ElementBases.hpp"
#include "LocalBoundary.hpp"
#include "InterfaceData.hpp"

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace poly_fem
{
	class SplineBasis3d
	{
	public:
		static int build_bases(
			const Mesh3D &mesh,
			const int quadrature_order,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::vector< int > &bounday_nodes,
			std::map<int, InterfaceData> &poly_face_to_data);

		static void fit_nodes(const Mesh3D &mesh, const int n_bases, std::vector< ElementBases > &gbases);
	};
}

#endif //SPLINE_BASIS_3D_HPP
