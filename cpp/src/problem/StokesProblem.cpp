#include "StokesProblem.hpp"

namespace polyfem
{
	DrivenCavity::DrivenCavity(const std::string &name)
	: Problem(name)
	{
		// boundary_ids_ = {1};
	}

	void DrivenCavity::rhs(const std::string &formulation, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	}

	void DrivenCavity::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts,const double t, Eigen::MatrixXd &val) const
	{
		val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

		for(long i = 0; i < pts.rows(); ++i)
		{
			if(mesh.get_boundary_id(global_ids(i))== 1)
				val(i, 1)=0.25;
			// else if(mesh.get_boundary_id(global_ids(i))== 3)
				// val(i, 1)=-0.25;
		}

		val *= t;
	}
}