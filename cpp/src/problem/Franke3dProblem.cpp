#include "Franke3dProblem.hpp"

#include <iostream>

namespace poly_fem
{
	Franke3dProblem::Franke3dProblem(const std::string &name)
	: Problem(name)
	{ }

	void Franke3dProblem::rhs(const std::string &formulation, const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		assert(mesh.is_volume());
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();
		auto &z = pts.col(2).array();

		val =
		(1181472075 * x * x + 1181472075 * y * y + 1181472075 * z * z - 525098700 * x - 525098700 * y - 525098700 * z + 87516450) / 960400. *
		exp(-81./4. * x * x + 9 * x - 3 - 81./4. * y * y + 9 * y - 81./4. * z * z + 9 * z) +

		(787648050 * x * x + 3150592200 * y * y - 1225230300 * x - 2800526400 * y + 1040473350) / 960400. *
		exp(-81./4. * x * x + 63./2. * x - 83./4. - 81./2. * y * y + 36 * y) +

		(7873200 * x * x + 1749600 * x - 1117314) / 960400. *
		exp(-81./49. * x * x - 18./49. * x - 54./245. - 9./10. * y - 9./10. * z) -

		26244./ 5. * (x * x + 4 * y * y - 8./9. * x - 16./3. * y + 317./162.) *
		exp(-81 * x * x - 162 * y * y + 72 * x + 216 * y - 90);

	}

	void Franke3dProblem::bc(const std::string &formulation, const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(formulation, pts, val);
	}

	void Franke3dProblem::exact(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();
		auto &z = pts.col(2).array();

		auto cx2 = (9*x-2) * (9*x-2);
		auto cy2 = (9*y-2) * (9*y-2);
		auto cz2 = (9*z-2) * (9*z-2);

		auto cx1 = (9*x+1) * (9*x+1);
		auto cx7 = (9*x-7) * (9*x-7);

		auto cy3 = (9*y-3) * (9*y-3);
		auto cx4 = (9*x-4) * (9*x-4);
		auto cy7 = (9*y-7) * (9*y-7);

		auto cz5 = (9*y-5) * (9*y-5);

		val =
		3./4. * exp( -1./4.*cx2 - 1./4.*cy2 - 1./4.*cz2) +
		3./4. * exp(-1./49. * cx1 - 9./10.*y - 1./10. -  9./10.*z - 1./10.) +
		1./2. * exp(-1./4. * cx7 - 1./4. * cy3 - 1./4. * cz5) -
		1./5. * exp(- cx4 - cy7 - cz5);
	}

	void Franke3dProblem::exact_grad(const std::string &formulation, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();
		auto &z = pts.col(2).array();

		val.resize(pts.rows(), pts.cols());

		val.col(0) = (-59535 * x + 13230) * exp(-0.81e2 / 0.4e1 * x * x + (9 * x) - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) / 0.1960e4 + (-39690 * x + 30870) * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) / 0.1960e4 + (-4860 * x - 540) * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) / 0.1960e4 + 0.162e3 / 0.5e1 * exp(-(81 * x * x) - 0.162e3 * y * y + (72 * x) + 0.216e3 * y - 0.90e2) * (x - 0.4e1 / 0.9e1);
		val.col(1) = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * y + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z) - 0.81e2 / 0.2e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) * y + 0.18e2 * exp(-0.81e2 / 0.4e1 * x * x + 0.63e2 / 0.2e1 * x - 0.83e2 / 0.4e1 - 0.81e2 / 0.2e1 * y * y + 0.36e2 * y) + 0.324e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.162e3 * y * y + 0.72e2 * x + 0.216e3 * y - 0.90e2) * y - 0.216e3 / 0.5e1 * exp(-0.81e2 * x * x - 0.162e3 * y * y + 0.72e2 * x + 0.216e3 * y - 0.90e2);
		val.col(2) = -0.243e3 / 0.8e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) * z + 0.27e2 / 0.4e1 * exp(-0.81e2 / 0.4e1 * x * x + 0.9e1 * x - 0.3e1 - 0.81e2 / 0.4e1 * y * y + 0.9e1 * y - 0.81e2 / 0.4e1 * z * z + 0.9e1 * z) - 0.27e2 / 0.40e2 * exp(-0.81e2 / 0.49e2 * x * x - 0.18e2 / 0.49e2 * x - 0.54e2 / 0.245e3 - 0.9e1 / 0.10e2 * y - 0.9e1 / 0.10e2 * z);

	}
}
