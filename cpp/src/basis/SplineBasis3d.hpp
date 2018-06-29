#ifndef SPLINE_BASIS_3D_HPP
#define SPLINE_BASIS_3D_HPP

#include <polyfem/Mesh3D.hpp>
#include <polyfem/ElementBases.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/InterfaceData.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace polyfem
{
	class SplineBasis3d
	{
	public:
		static int build_bases(
			const Mesh3D &mesh,
			const int quadrature_order,
			std::vector< ElementBases > &bases,
			std::vector< LocalBoundary > &local_boundary,
			std::map<int, InterfaceData> &poly_face_to_data);

		static void fit_nodes(const Mesh3D &mesh, const int n_bases, std::vector< ElementBases > &gbases);
	};
}

#endif //SPLINE_BASIS_3D_HPP
