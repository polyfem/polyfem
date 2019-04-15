////////////////////////////////////////////////////////////////////////////////
#include <polyfem/MVPolygonalBasis2d.hpp>
#include <polyfem/PolygonQuadrature.hpp>

#include <polyfem/AssemblerUtils.hpp>

#include <memory>


namespace polyfem {
namespace {
	std::vector<int> compute_nonzero_bases_ids(const Mesh2D &mesh, const int element_index,
		const std::vector< ElementBases > &bases,
		const std::map<int, InterfaceData> &poly_edge_to_data, const Eigen::MatrixXd &poly)
	{
		const int n_edges = mesh.n_face_vertices(element_index);

		std::vector<int> local_to_global(n_edges);

		Navigation::Index index = mesh.get_index_from_face(element_index);
		for (int i = 0; i < n_edges; ++i) {
			bool found = false;

			Navigation::Index index1 = mesh.next_around_vertex(index);
			while(index1.face != index.face)
			{
				if(index1.face < 0)
					break;
				if(found)
					break;

				const ElementBases &bs=bases[index1.face];

				for (const auto &b : bs.bases) {
					for (const auto &x : b.global()) {
						const int global_node_id = x.index;
						if((x.node - poly.row(i)).norm()<1e-10)
						{
							local_to_global[i] = global_node_id;
							found = true;
							assert(b.global().size() == 1);
							break;
						}
					}
					if(found)
						break;
				}

				index1 = mesh.next_around_vertex(index1);
			}

			index1 = mesh.next_around_vertex(mesh.switch_edge(index));

			while(index1.face != index.face)
			{
				if(index1.face < 0)
					break;
				if(found)
					break;

				const ElementBases &bs=bases[index1.face];

				for (const auto &b : bs.bases) {
					for (const auto &x : b.global()) {
						const int global_node_id = x.index;
						if((x.node - poly.row(i)).norm()<1e-10)
						{
							local_to_global[i] = global_node_id;
							found = true;
							assert(b.global().size() == 1);
							break;
						}
					}
					if(found)
						break;
				}

				index1 = mesh.next_around_vertex(index1);
			}

			if(!found)
				local_to_global[i] = -1;


			index = mesh.next_around_face(index);
		}

		return local_to_global;
	}


	void meanvalue(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &b, const double tol)
	{
		const int n_boundary = polygon.rows();

		Eigen::MatrixXd segments(n_boundary, 2);
		Eigen::VectorXd radii(n_boundary);
		Eigen::VectorXd areas(n_boundary);
		Eigen::VectorXd products(n_boundary);
		Eigen::VectorXd tangents(n_boundary);
		Eigen::Matrix2d mat;

		b.resize(n_boundary, 1);
		b.setZero();


		for(int i = 0; i < n_boundary; ++i)
		{
			segments.row(i) = polygon.row(i) - point;

			radii(i) = segments.row(i).norm();

			//we are on the vertex
			if(radii(i) < tol) {
				b(i) = 1;

				return;
			}
		}


		for(int i = 0; i < n_boundary; ++i) {
			const int ip1 = (i + 1) == n_boundary ? 0 : (i+1);

			mat.row(0) = segments.row(i);
			mat.row(1) = segments.row(ip1);

			areas(i) = mat.determinant();
			products(i) = segments.row(i).dot(segments.row(ip1));

			//we are on the edge
			if(fabs(areas[i]) < tol && products(i) < 0) {
				const double denominator = 1.0/(radii(i) + radii(ip1));

				b(i) = radii(ip1) * denominator;
				b(ip1) = radii(i) * denominator;

				return;
			}
		}


		for(int i = 0; i < n_boundary; ++i) {
			const int ip1 = (i + 1) == n_boundary ? 0 : (i+1);

			tangents(i) = areas(i)/(radii(i)*radii(ip1) + products(i));
		}


		double W = 0;
		for(int i = 0; i < n_boundary; ++i) {
			const int im1 = i == 0 ? (n_boundary-1) : (i-1);

			b(i) = (tangents(im1) + tangents(i))/radii(i);
			W += b(i);
		}

		b /= W;
	}

	void meanvalue_derivative(const Eigen::MatrixXd &polygon, const Eigen::RowVector2d &point, Eigen::MatrixXd &derivatives, const double tol)
	{
		const int n_boundary = polygon.rows();


		// b.resize(n_boundary*n_points);
		// std::fill(b.begin(), b.end(), 0);

		derivatives.resize(n_boundary, 2);
		derivatives.setZero();

		Eigen::MatrixXd segments(n_boundary, 2);
		Eigen::VectorXd radii(n_boundary);
		Eigen::VectorXd areas(n_boundary);
		Eigen::VectorXd products(n_boundary);
		Eigen::VectorXd tangents(n_boundary);
		Eigen::Matrix2d mat;


		Eigen::MatrixXd areas_prime(n_boundary, 2);
		Eigen::MatrixXd products_prime(n_boundary, 2);
		Eigen::MatrixXd radii_prime(n_boundary, 2);
		Eigen::MatrixXd tangents_prime(n_boundary, 2);
		Eigen::MatrixXd w_prime(n_boundary, 2);

		// Eigen::MatrixXd b(n_boundary, 1);

		for(int i = 0; i < n_boundary; ++i)
		{
			segments.row(i) = polygon.row(i) - point;

			radii(i) = segments.row(i).norm();

			//we are on the vertex
			if(radii(i) < tol) {
				assert(false);
				// b(i) = 1;
				return;
			}
		}

		for(int i = 0; i < n_boundary; ++i) {
			const int ip1 = (i + 1) == n_boundary ? 0 : (i+1);

			mat.row(0) = segments.row(i);
			mat.row(1) = segments.row(ip1);

			areas(i) = mat.determinant();
			products(i) = segments.row(i).dot(segments.row(ip1));

			//we are on the edge
			if(fabs(areas[i]) < tol && products(i) < 0) {
				const double denominator = 1.0/(radii(i) + radii(ip1));

				// b(i) = radii(ip1) * denominator;
				// b(ip1) = radii(i) * denominator;
				assert(false);
				//TODO add derivative
				return;
			}


			const Eigen::RowVector2d vi = polygon.row(i);
			const Eigen::RowVector2d vip1 = polygon.row(ip1);

			areas_prime(i, 0) = vi(1)-vip1(1);
			areas_prime(i, 1) = vip1(0)-vi(0);

			products_prime.row(i) = 2 * point - vi - vip1;

			radii_prime.row(i)   = (point-vi)/radii(i);
		}


		for(int i = 0; i < n_boundary; ++i)
		{
			const int ip1 = (i + 1) == n_boundary ? 0 : (i+1);

			const double denominator = radii(i) * radii(ip1) + products(i);

			const Eigen::RowVector2d denominator_prime = radii_prime.row(i)*radii(ip1)+radii(i)*radii_prime.row(ip1) + products_prime.row(i);

			tangents_prime.row(i)  =(areas_prime.row(i)*denominator - areas(i)*denominator_prime)/(denominator*denominator);

			tangents(i)=areas(i)/denominator;
		}

		double W = 0;
		Eigen::RowVector2d W_prime; W_prime.setZero();


		for(int i = 0; i < n_boundary; ++i)
		{
			const int im1 = (i > 0) ? (i-1) : (n_boundary-1);

			w_prime.row(i) = ((tangents_prime.row(im1) + tangents_prime.row(i))*radii(i)-(tangents(im1)+tangents(i))*radii_prime.row(i))/(radii(i)*radii(i));;


			W_prime += w_prime.row(i);
			W += (tangents(im1)+tangents(i))/radii(i);
		}

		for(int i = 0; i < n_boundary; ++i){
			const int im1 = (i > 0) ? (i-1) : (n_boundary-1);

			const double bi=(tangents(im1)+tangents(i))/radii(i);
			derivatives.row(i)= (w_prime.row(i)*W-bi*W_prime)/(W*W);
		}
	}


} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////


int MVPolygonalBasis2d::build_bases(
			const std::string &assembler_name,
			const Mesh2D &mesh,
			const int n_bases,
			const int quadrature_order,
			std::vector< ElementBases > &bases,
			const std::vector< ElementBases > &gbases,
			const  std::map<int, InterfaceData> &poly_edge_to_data,
			std::map<int, Eigen::MatrixXd> &mapped_boundary)
{
	assert(!mesh.is_volume());
	if (poly_edge_to_data.empty()) {
		return 0;
	}

	const auto &assembler = AssemblerUtils::instance();
	const int dim = assembler.is_tensor(assembler_name) ? 2 : 1;

	Eigen::MatrixXd polygon;

	// int new_nodes = 0;

	std::map<int, int> new_nodes;

	PolygonQuadrature poly_quadr;
	for (int e = 0; e < mesh.n_elements(); ++e) {
		if (!mesh.is_polytope(e)) {
			continue;
		}

		polygon.resize(mesh.n_face_vertices(e), 2);

		for(int i = 0; i < mesh.n_face_vertices(e); ++i){
			const int gid = mesh.face_vertex(e, i);
			polygon.row(i) = mesh.point(gid);
		}

		std::vector<int> local_to_global = compute_nonzero_bases_ids(mesh, e, bases, poly_edge_to_data, polygon);

		for(int i = 0; i < local_to_global.size(); ++i){
			if(local_to_global[i] >= 0)
				continue;

			const int gid = mesh.face_vertex(e, i);
			const auto other_gid = new_nodes.find(gid);
			if(other_gid != new_nodes.end())
				local_to_global[i] = other_gid->second;
			else
			{
				const int tmp = new_nodes.size() + n_bases;
				new_nodes[gid] = tmp;
				local_to_global[i] = tmp;
			}

		}

		ElementBases &b=bases[e];
		b.has_parameterization = false;

		// Compute quadrature points for the polygon
		Quadrature tmp_quadrature;
		poly_quadr.get_quadrature(polygon, quadrature_order, tmp_quadrature);

		b.set_quadrature([tmp_quadrature](Quadrature &quad){ quad = tmp_quadrature; });

		const double tol=1e-10;
		b.set_bases_func([polygon, tol](const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val)
		{
			Eigen::MatrixXd tmp;
			val.resize(polygon.rows());
			for(size_t i = 0; i < polygon.rows(); ++i){
				val[i].val.resize(uv.rows(), 1);
			}

			for(int i = 0; i < uv.rows(); ++i){
				meanvalue(polygon, uv.row(i), tmp, tol);

				for(size_t j = 0; j < tmp.size(); ++j){
					val[j].val(i) = tmp(j);
				}
			}
		});
		b.set_grads_func([polygon, tol] (const Eigen::MatrixXd &uv, std::vector<AssemblyValues> &val)
		{
			Eigen::MatrixXd tmp;
			val.resize(polygon.rows());
			for(size_t i = 0; i < polygon.rows(); ++i){
				val[i].grad.resize(uv.rows(), 2);
			}

			for(int i = 0; i < uv.rows(); ++i){
				meanvalue_derivative(polygon, uv.row(i), tmp, tol);
				assert(tmp.rows() == polygon.rows());

				for(size_t j = 0; j < tmp.rows(); ++j){
					val[j].grad.row(i) = tmp.row(j);
				}
			}
		});



		// Set the bases which are nonzero inside the polygon
		const int n_poly_bases = int(local_to_global.size());
		b.bases.resize(n_poly_bases);
		for (int i = 0; i < n_poly_bases; ++i) {
			b.bases[i].init(-1, local_to_global[i], i, polygon.row(i));
		}

		// Polygon boundary after geometric mapping from neighboring elements
		mapped_boundary[e] = polygon;
	}

	return new_nodes.size();
}

} // namespace polyfem
