////////////////////////////////////////////////////////////////////////////////
#include "navigation.hpp"
////////////////////////////////////////////////////////////////////////////////

void poly_fem::Navigation::prepare_mesh(GEO::Mesh &M) {
    M.facets.connect();
    M.cells.connect();
    if(M.cells.nb() != 0 && M.facets.nb() == 0) {
        M.cells.compute_borders();
    }
}

int poly_fem::Navigation::switch_vertex(const GEO::Mesh &M, int v, int e, int f)
{
	return 0;
}

int poly_fem::Navigation::switch_edge(const GEO::Mesh &M, int v, int e, int f)
{
	return 0;
}

int poly_fem::Navigation::switch_face(const GEO::Mesh &M, int v, int e, int f)
{
	return 0;
}

