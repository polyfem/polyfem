#include "Franke2dProblem.hpp"

#include <iostream>

namespace poly_fem
{
	Franke2dProblem::Franke2dProblem()
	{
		problem_num_ = ProblemType::Franke;
	}

	void Franke2dProblem::rhs(const Mesh &mesh, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		auto cx2 = (9*x-2) * (9*x-2);
		auto cy2 = (9*y-2) * (9*y-2);

		auto cx1 = (9*x+1) * (9*x+1);
		auto cx7 = (9*x-7) * (9*x-7);

		auto cy3 = (9*y-3) * (9*y-3);
		auto cx4 = (9*x-4) * (9*x-4);

		auto cy7 = (9*y-7) * (9*y-7);

		auto s1 = (-40.5 * x+9) * (-40.5 * x + 9);
		auto s2 = (-162./49. * x - 18./49.) * (-162./49. * x - 18./49.);
		auto s3 = (-40.5 * x + 31.5) * (-40.5 * x + 31.5);
		auto s4 = (-162. * x + 72) * (-162 * x + 72);

		auto s5 = (-40.5 * y + 9) * (-40.5 * y + 9);
		auto s6 = (-40.5 * y + 13.5) * (-40.5 * y + 13.5);
		auto s7 = (-162 * y + 126) * (-162 * y + 126);

		val = 243./4. * (-0.25 * cx2 - 0.25 * cy2).exp() -   0.75 * s1 * (-0.25 * cx2 - 0.25 *cy2).exp() +
		36693./19600. * (-1./49. * cx1 - 0.9 * y - 0.1).exp()  - 0.75 * s2 * (- 1./49 * cx1 - 0.9 * y - 0.1).exp() +
		40.5 * (-0.25 * cx7 - 0.25 * cy3).exp()   - 0.5 * s3 * (-0.25 * cx7 - 0.25 * cy3).exp() -
		324./5.  * (-cx4-cy7).exp() + 0.2 * s4 * (-cx4-cy7).exp() -
		0.75 * s5 * (-0.25 * cx2 - 0.25 *cy2).exp()  - 0.5 * s6 * (-0.25 * cx7 - 0.25 * cy3).exp() +
		0.2 * s7 * (-cx4-cy7).exp();
		val*=-1;
	}

	void Franke2dProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		exact(pts, val);
	}


	void Franke2dProblem::exact(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
	{
		auto &x = pts.col(0).array();
		auto &y = pts.col(1).array();

		auto cx2 = (9*x-2) * (9*x-2);
		auto cy2 = (9*y-2) * (9*y-2);

		auto cx1 = (9*x+1) * (9*x+1);
		auto cx7 = (9*x-7) * (9*x-7);

		auto cy3 = (9*y-3) * (9*y-3);
		auto cx4 = (9*x-4) * (9*x-4);

		auto cy7 = (9*y-7) * (9*y-7);

		// val = 0.75 * (-0.25 * cx2 - 0.25 * cy2).exp() +
		// 0.75 * (-1./49. * cx1 - 0.9 * y - 0.1).exp() +
		// 0.5 * (-0.25 * cx7 - 0.25 * cy3).exp() -
		// 0.2 * (-cx4-cy7).exp();

		val = (3./4.)*exp(-(1./4.)*cx2-(1./4.)*cy2)+(3./4.)*exp(-(1./49.)*cx1-(9./10.)*y-1./10.)+(1./2.)*exp(-(1./4.)*cx7-(1./4.)*cy3)-(1./5.)*exp(-cx4-cy7);

	}
}
