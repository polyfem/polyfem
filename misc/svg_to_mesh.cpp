#ifdef POLYFEM_WITH_CLIPPER
////////////////////////////////////////////////////////////////////////////////
#include <CLI/CLI.hpp>
#include <igl/write_triangle_mesh.h>
#include <clipper/clipper.hpp>
#define NANOSVG_ALL_COLOR_KEYWORDS // Include full list of color keywords.
#define NANOSVG_IMPLEMENTATION     // Expands implementation
#include <nanosvg.h>
#include <complex>
#include <vector>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////

// Points and polygons
typedef std::complex<double> Point;
typedef std::vector<Point> Polygon;

// Scinde une courbe de bézier en multiples segments
void svgFlattenCubicBezier(ClipperLib::Path &poly,
						   double x1, double y1, double x2, double y2,
						   double x3, double y3, double x4, double y4,
						   double tol, int level)
{
	double x12, y12, x23, y23, x34, y34, x123, y123, x234, y234, x1234, y1234;

	if (level > 6)
	{
		return; // Max level, don't go further
	}

	if (std::abs(x1 + x3 - x2 - x2) + std::abs(y1 + y3 - y2 - y2) + std::abs(x2 + x4 - x3 - x3) + std::abs(y2 + y4 - y3 - y3) < tol)
	{
		poly.push_back(Point(x4, y4));
		return;
	}

	x12 = (x1 + x2) * 0.5;
	y12 = (y1 + y2) * 0.5;
	x23 = (x2 + x3) * 0.5;
	y23 = (y2 + y3) * 0.5;
	x34 = (x3 + x4) * 0.5;
	y34 = (y3 + y4) * 0.5;
	x123 = (x12 + x23) * 0.5;
	y123 = (y12 + y23) * 0.5;
	x234 = (x23 + x34) * 0.5;
	y234 = (y23 + y34) * 0.5;
	x1234 = (x123 + x234) * 0.5;
	y1234 = (y123 + y234) * 0.5;

	svgFlattenCubicBezier(poly, x1, y1, x12, y12, x123, y123, x1234, y1234, tol, level + 1);
	svgFlattenCubicBezier(poly, x1234, y1234, x234, y234, x34, y34, x4, y4, tol, level + 1);
}

// -----------------------------------------------------------------------------

// Charge une image SVG depuis un fichier
std::vector<Polygon> loadSvg(const std::string &file, float dpi = 90.0)
{
	std::vector<Polygon> contours;
	std::cerr << "[loading] " << file << " ... ";
	double tol = 1.5;
	NSVGimage *image = nsvgParseFromFile(file.c_str(), "mm", dpi);
	// For all shapes in the file
	for (NSVGshape *shape = image->shapes; shape != NULL; shape = shape->next)
	{
		// Approximate shape with a sequence of segments
		if (shape->fillRule == NSVG_FILLRULE_NONZERO)
		{
			// Winding number
		}
		else
		{
			// Even-odd
			assert(shape->fillRule == NSVG_FILLRULE_EVENODD);
		}
		Polygon poly;
		for (NSVGpath *path = shape->paths; path != NULL; path = path->next)
		{
			for (int i = 0; i < path->npts - 1; i += 3)
			{
				const float *p = &path->pts[i * 2];
				svgFlattenCubicBezier(poly, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], tol, 0);
			}
		}
		contours.push_back(poly);
	}
	std::cerr << "SVG size: " << image->width << " x " << image->height
			  << " (" << contours.size() << " polygones)" << std::endl;
	nsvgDelete(image);
	return contours;
}

////////////////////////////////////////////////////////////////////////////////

void svg_to_mesh()
{
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	struct
	{
		std::string input = "";
		std::string output = "output.obj";
	} args;

	CLI::App app{"Convert SVG image to a triangle mesh"};
	app.add_option("input,-i,--input", args.input, "Input SVG.")->required();
	app.add_option("output,-o,--output", args.output, "Output triangle mesh.");
	try
	{
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError &e)
	{
		return app.exit(e);
	}

	return EXIT_SUCCESS;
}
#else
#include <iostream>
int main(int argc, char *argv[])
{
	std::cout << "Needs clipper!!!" << std::endl;
	return EXIT_SUCCESS;
}
#endif