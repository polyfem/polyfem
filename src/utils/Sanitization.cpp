#include <polyfem/Sanitization.hpp>
using namespace polyfem;
using namespace Sanitization;
void input_json_sanitization()
{
	// critical
	//// from "tend", "time_steps", "dt" only two can be specified
	//// args["dhat"] <= 0
	//// poison ratio >= 0.5 in 3D or >= 1 in 2D

	// warning
	//// args["friction_coefficient"] not in [0, 1] (outside of normal range)

	// error
	//// NA
}
void input_geom_sanitization()
{
	// critical
	//// not self intersecting

	// warning
	//// manifold check
	//// elemnets are not degenrate (diahedral angel, jacobian of deformation(equilateral to input))

	// error
	//// elements not inverted
}
void input_simul_sanitization()
{
	// critical
	//// NA

	// warning
	//// tolerance (maybe 1e-5) * dhat > min_dist

	// error
	//// args["dhat"] > min_edge_length
}