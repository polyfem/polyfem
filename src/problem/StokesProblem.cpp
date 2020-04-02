#include "StokesProblem.hpp"

#include <polyfem/StringUtils.hpp>

namespace polyfem
{

TimeDepentendStokesProblem::TimeDepentendStokesProblem(const std::string &name)
	: Problem(name)
{
	is_time_dependent_ = false;
}

void TimeDepentendStokesProblem::set_parameters(const json &params)
{
	if (params.find("time_dependent") != params.end())
	{
		is_time_dependent_ = params["time_dependent"];
	}
}

void TimeDepentendStokesProblem::initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	// val = Eigen::MatrixXd::Ones(pts.rows(), pts.cols())*(1 - exp(-5 * 0.01));
}

ConstantVelocity::ConstantVelocity(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	// boundary_ids_ = {1};
}

void ConstantVelocity::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void ConstantVelocity::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		val(i, 1) = 1;
	}

	// val *= t;

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

DrivenCavity::DrivenCavity(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	// boundary_ids_ = {1};
}

void DrivenCavity::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void DrivenCavity::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) == 4)
			val(i, 0) = 1;
		// else if(mesh.get_boundary_id(global_ids(i))== 3)
		// val(i, 1)=-0.25;
	}

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

DrivenCavitySmooth::DrivenCavitySmooth(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	// boundary_ids_ = {1};
}

void DrivenCavitySmooth::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void DrivenCavitySmooth::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) == 4)
		{
			const double x = pts(i, 0);
			val(i, 0) = 4 * (1 - x) * x;
		}
		// else if(mesh.get_boundary_id(global_ids(i))== 3)
		// val(i, 1)=-0.25;
	}

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

Flow::Flow(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	boundary_ids_ = {1, 3, 7};
	inflow_ = 1;
	outflow_ = 3;
	flow_dir_ = 0;

	inflow_amout_ = 0.25;
	outflow_amout_ = 0.25;
}

void Flow::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void Flow::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) == inflow_)
			val(i, flow_dir_) = inflow_amout_;
		else if (mesh.get_boundary_id(global_ids(i)) == outflow_)
			val(i, flow_dir_) = outflow_amout_;
	}

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

void Flow::set_parameters(const json &params)
{
	TimeDepentendStokesProblem::set_parameters(params);

	if (params.find("inflow") != params.end())
	{
		inflow_ = params["inflow"];
	}

	if (params.find("outflow") != params.end())
	{
		outflow_ = params["outflow"];
	}

	if (params.find("inflow_amout") != params.end())
	{
		inflow_amout_ = params["inflow_amout"];
	}

	if (params.find("outflow_amout") != params.end())
	{
		outflow_amout_ = params["outflow_amout"];
	}

	if (params.find("direction") != params.end())
	{
		flow_dir_ = params["direction"];
	}

	boundary_ids_.clear();

	if (params.find("obstacle") != params.end())
	{
		const auto obstacle = params["obstacle"];
		if (obstacle.is_array())
		{
			for (size_t k = 0; k < obstacle.size(); ++k)
			{
				const auto tmp = obstacle[k];
				if (tmp.is_string())
				{
					const std::string tmps = tmp;
					const auto endings = StringUtils::split(tmps, ":");
					assert(endings.size() == 2);
					const int start = atoi(endings[0].c_str());
					const int end = atoi(endings[1].c_str());

					for (int i = start; i <= end; ++i)
						boundary_ids_.push_back(i);
				}
				else
					boundary_ids_.push_back(tmp);
			}
		}
	}

	boundary_ids_.push_back(inflow_);
	boundary_ids_.push_back(outflow_);

	std::sort(boundary_ids_.begin(), boundary_ids_.end());
	auto it = std::unique(boundary_ids_.begin(), boundary_ids_.end());
	boundary_ids_.resize(std::distance(boundary_ids_.begin(), it));

	// for(int i : boundary_ids_)
	// 	std::cout<<"i "<<i<<std::endl;
}

FlowWithObstacle::FlowWithObstacle(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	boundary_ids_ = {1, 2, 4, 7};
	U_ = 1.5;
}

void FlowWithObstacle::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void FlowWithObstacle::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) == 1)
		{
			const double y = pts(i, 1);
			val(i, 0) = U_ * 4 * y * (0.41 - y) / (0.41 * 0.41);
		}
	}

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

void FlowWithObstacle::set_parameters(const json &params)
{
	TimeDepentendStokesProblem::set_parameters(params);

	if (params.find("U") != params.end())
	{
		U_ = params["U"];
	}
}

UnitFlowWithObstacle::UnitFlowWithObstacle(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	boundary_ids_ = {1, 2, 4, 7};
	U_ = 1.5;
	inflow_ = 1;
	dir_ = 0;
}

void UnitFlowWithObstacle::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void UnitFlowWithObstacle::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) == inflow_)
		{
			if (pts.cols() == 3)
			{
				const double u = pts(i, (dir_ + 1) % 3);
				const double v = pts(i, (dir_ + 2) % 3);
				val(i, dir_) = U_ * 24 * (1 - u) * u * (1 - v) * v;
			}
			else
			{
				const double v = pts(i, (dir_ + 1) % 2);
				val(i, dir_) = U_ * 4 * (1 - v) * v;
			}
		}
	}

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

void UnitFlowWithObstacle::set_parameters(const json &params)
{
	TimeDepentendStokesProblem::set_parameters(params);

	if (params.find("U") != params.end())
	{
		U_ = params["U"];
	}

	if (params.find("inflow_id") != params.end())
	{
		inflow_ = params["inflow_id"];
	}

	if (params.find("direction") != params.end())
	{
		dir_ = params["direction"];
	}

	if (params.find("no_slip") != params.end())
	{
		boundary_ids_.clear();

		const auto no_slip = params["no_slip"];
		if (no_slip.is_array())
		{
			for (size_t k = 0; k < no_slip.size(); ++k)
			{
				const auto tmp = no_slip[k];
				if (tmp.is_string())
				{
					const std::string tmps = tmp;
					const auto endings = StringUtils::split(tmps, ":");
					assert(endings.size() == 2);
					const int start = atoi(endings[0].c_str());
					const int end = atoi(endings[1].c_str());

					for (int i = start; i <= end; ++i)
						boundary_ids_.push_back(i);
				}
				else
					boundary_ids_.push_back(tmp);
			}
		}
	}

	boundary_ids_.push_back(inflow_);

	std::sort(boundary_ids_.begin(), boundary_ids_.end());
	auto it = std::unique(boundary_ids_.begin(), boundary_ids_.end());
	boundary_ids_.resize(std::distance(boundary_ids_.begin(), it));

	// for(int i : boundary_ids_)
	// 	std::cout<<"i "<<i<<std::endl;
}

TaylorGreenVortexProblem::TaylorGreenVortexProblem(const std::string &name)
	: Problem(name), viscosity_(1)
{
}

void TaylorGreenVortexProblem::initial_solution(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	exact(pts, 0, val);
}

void TaylorGreenVortexProblem::set_parameters(const json &params)
{
	if (params.count("viscosity"))
	{
		viscosity_ = params["viscosity"];
	}
}

void TaylorGreenVortexProblem::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	for(int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		val(i, 0) =  cos(x) * sin(y) * exp(-2 * viscosity_ * t);
		val(i, 1) = -sin(x) * cos(y) * exp(-2 * viscosity_ * t);
		//val(i, 0) = 0;
		//val(i, 1) = 0.5 * (8 + 2 * M_PI * x - x * x) * cos(y / 2);
	}

}

void TaylorGreenVortexProblem::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols() * pts.cols());

	for (int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		val(i, 0) = -sin(x) * sin(y) * exp(-2 * viscosity_ * t);
		val(i, 1) =  cos(x) * cos(y) * exp(-2 * viscosity_ * t);
		val(i, 2) = -cos(x) * cos(y) * exp(-2 * viscosity_ * t);
		val(i, 3) =  sin(x) * sin(y) * exp(-2 * viscosity_ * t);
	}
}

void TaylorGreenVortexProblem::rhs(const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}
void TaylorGreenVortexProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	exact(pts, t, val);
}

} // namespace polyfem