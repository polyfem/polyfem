////////////////////////////////////////////////////////////////////////////////
#include "Mesh.hpp"
////////////////////////////////////////////////////////////////////////////////

void poly_fem::Mesh::edge_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_edges(), dimension());
	for (int e = 0; e < n_edges(); ++e) {
		barycenters.row(e) = edge_barycenter(e);
	}
}

void poly_fem::Mesh::face_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_faces(), dimension());
	for (int f = 0; f < n_faces(); ++f) {
		barycenters.row(f) = face_barycenter(f);
	}
}

void poly_fem::Mesh::cell_barycenters(Eigen::MatrixXd &barycenters) const {
	barycenters.resize(n_cells(), dimension());
	for (int c = 0; c < n_cells(); ++c) {
		barycenters.row(c) = cell_barycenter(c);
	}
}

////////////////////////////////////////////////////////////////////////////////

//Queries on the tags
bool poly_fem::Mesh::is_spline_compatible(const int el_id) const
{
	if(is_volume()){
		return
		elements_tag_[el_id] == ElementType::RegularInteriorCube ||
		elements_tag_[el_id] == ElementType::RegularBoundaryCube ||
		elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube ||
		elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube;
	}
	else
	{
		return
		elements_tag_[el_id] == ElementType::RegularInteriorCube ||
		elements_tag_[el_id] == ElementType::RegularBoundaryCube ||
		elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube;
	}
}

// -----------------------------------------------------------------------------

bool poly_fem::Mesh::is_cube(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::RegularInteriorCube ||
	elements_tag_[el_id] == ElementType::RegularBoundaryCube ||

	elements_tag_[el_id] == ElementType::SimpleSingularInteriorCube ||
	elements_tag_[el_id] == ElementType::SimpleSingularBoundaryCube ||

	elements_tag_[el_id] == ElementType::MultiSingularInteriorCube ||
	elements_tag_[el_id] == ElementType::MultiSingularBoundaryCube;
}

// -----------------------------------------------------------------------------

bool poly_fem::Mesh::is_polytope(const int el_id) const
{
	return
	elements_tag_[el_id] == ElementType::InteriorPolytope ||
	elements_tag_[el_id] == ElementType::BoundaryPolytope;
}
