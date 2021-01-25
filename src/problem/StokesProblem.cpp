#include "StokesProblem.hpp"

#include <polyfem/StringUtils.hpp>

namespace polyfem
{
	namespace
	{
		//https://math.stackexchange.com/questions/101480/are-there-other-kinds-of-bump-functions-than-e-frac1x2-1
		double bump(double r)
		{
			const auto f = [](double x) { return x <= 0 ? 0 : exp(-1. / x); };
			const auto g = [&f](double x) { return f(x) / (f(x) + f(1. - x)); };
			const auto h = [&g](double x) { return g(x - 1); };
			const auto k = [&h](double x) { return h(x * x); };
			return 1 - k(r);
		}
	} // namespace

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

void TimeDepentendStokesProblem::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
	// val = Eigen::MatrixXd::Ones(pts.rows(), pts.cols())*(1 - exp(-5 * 0.01));
}

ConstantVelocity::ConstantVelocity(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	boundary_ids_ = {1, 2, 3, 4, 5, 6, 7};
}

void ConstantVelocity::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void ConstantVelocity::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) != 7)
		{
			val(i, 1) = 1;
		}
	}

	// val *= t;

	// if (is_time_dependent_)
	// 	val *= (1 - exp(-5 * t));
}

TwoSpheres::TwoSpheres(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	// boundary_ids_ = {1};
}

void TwoSpheres::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void TwoSpheres::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void TwoSpheres::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (int i = 0; i < pts.rows(); ++i)
	{
		if(pts.cols() == 2)
		{
			const double r1 = sqrt((pts(i, 0) - 0.04) * (pts(i, 0) - 0.04) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2));
			const double r2 = sqrt((pts(i, 0) - 0.16) * (pts(i, 0) - 0.16) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2));

			val(i, 0) = 0.05 * bump(r1 * 70) - 0.05 * bump(r2 * 70);
		}
		else
		{
			const double r1 = sqrt((pts(i, 0) - 0.04) * (pts(i, 0) - 0.04) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2) + (pts(i, 2) - 0.2) * (pts(i, 2) - 0.2));
			const double r2 = sqrt((pts(i, 0) - 0.16) * (pts(i, 0) - 0.16) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2) + (pts(i, 2) - 0.2) * (pts(i, 2) - 0.2));

			val(i, 0) = 0.05 * bump(r1 * 70) - 0.05 * bump(r2 * 70);
		}
	}
}

void TwoSpheres::initial_density(const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), 1);
	for(int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		if(pts.cols() == 2)
		{
			const double r1 = sqrt((pts(i, 0) - 0.04) * (pts(i, 0) - 0.04) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2));
			const double r2 = sqrt((pts(i, 0) - 0.16) * (pts(i, 0) - 0.16) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2));
			
			val(i, 0) = bump(r1 * 70) + bump(r2 * 70);
		}
		else
		{
			const double r1 = sqrt((pts(i, 0) - 0.04) * (pts(i, 0) - 0.04) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2) + (pts(i, 2) - 0.2) * (pts(i, 2) - 0.2));
			const double r2 = sqrt((pts(i, 0) - 0.16) * (pts(i, 0) - 0.16) + (pts(i, 1) - 0.2) * (pts(i, 1) - 0.2) + (pts(i, 2) - 0.2) * (pts(i, 2) - 0.2));
			
			val(i, 0) = bump(r1 * 70) + bump(r2 * 70);
		}
	}
}

DrivenCavity::DrivenCavity(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	// boundary_ids_ = {1};
}

void DrivenCavity::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

void DrivenCavitySmooth::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

void Flow::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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
	boundary_ids_ = {1, 2, 4, 5, 6, 7};
	U_ = 1.5;
}

void FlowWithObstacle::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

Kovnaszy::Kovnaszy(const std::string &name)
	: Problem(name), viscosity_(1)
{
	boundary_ids_ = {1, 2, 3, 4, 5, 6, 7};
}

void Kovnaszy::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	exact(pts, 0, val);
}

void Kovnaszy::set_parameters(const json &params)
{
	if (params.count("viscosity"))
	{
		viscosity_ = params["viscosity"];
	}
}

void Kovnaszy::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	const double a = 0.5 / viscosity_ - sqrt(0.25 / viscosity_ / viscosity_ + 4 * M_PI * M_PI);
	for(int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		if(pts.cols() == 2)
		{
			val(i, 0) = 1 - exp(a*x)*cos(2*M_PI*y);
			val(i, 1) = a*exp(a*x)*sin(2*M_PI*y)/(2*M_PI);
		}
		else
		{
			val(i, 0) = 1 - exp(a*x)*cos(2*M_PI*y);
			val(i, 1) = a*exp(a*x)*sin(2*M_PI*y)/(2*M_PI);
			val(i, 2) = 0;
		}
	}
}

void Kovnaszy::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	const double time_scaling = exp(-2 * viscosity_ * t);

	val.resize(pts.rows(), pts.cols() * pts.cols());
	const double a = 0.5 / viscosity_ - sqrt(0.25 / viscosity_ / viscosity_ + 4 * M_PI * M_PI);
	for (int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		val(i, 0) = -a*exp(a*x)*cos(2*M_PI*y);
		val(i, 1) = 2*M_PI*exp(a*x)*sin(2*M_PI*y);
		val(i, 2) = a*a*exp(a*x)*sin(2*M_PI*y)/(2*M_PI);
		val(i, 3) = a*exp(a*x)*cos(2*M_PI*y);
	}
}

void Kovnaszy::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	val.setZero();
}

void Kovnaszy::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	exact(pts, t, val);
}

CornerFlow::CornerFlow(const std::string &name)
	: TimeDepentendStokesProblem(name)
{
	boundary_ids_ = {1, 2, 4, 7};
	U_ = 1.5;
}

void CornerFlow::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());
}

void CornerFlow::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val = Eigen::MatrixXd::Zero(pts.rows(), pts.cols());

	for (long i = 0; i < pts.rows(); ++i)
	{
		if (mesh.get_boundary_id(global_ids(i)) == 2)
		{
			val(i, 1) = U_;
		}
	}

	if (is_time_dependent_)
		val *= (1 - exp(-5 * t));
}

void CornerFlow::set_parameters(const json &params)
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

void UnitFlowWithObstacle::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
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

StokesLawProblem::StokesLawProblem(const std::string &name)
	: Problem(name), viscosity_(1e2), radius(0.5)
{
	boundary_ids_ = {1, 2, 3, 4, 5, 6, 7};
	is_time_dependent_ = false;
}

void StokesLawProblem::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	exact(pts, 0, val);
}

void StokesLawProblem::set_parameters(const json &params)
{
	if (params.count("viscosity"))
	{
		viscosity_ = params["viscosity"];
	}
	if (params.count("radius"))
	{
		radius = params["radius"];
	}
	if (params.find("time_dependent") != params.end())
	{
		is_time_dependent_ = params["time_dependent"];
	}
}

void StokesLawProblem::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	val.setZero();
	for(int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		if(pts.cols() == 2)
		{
			val(i, 1) = 1;
		}
		else
		{
			const double z = pts(i, 2);
			const double norm = sqrt(x*x+y*y+z*z);
			const double tmp1 = 3 * (x+y+z) / pow(norm, 5);
			const double tmp2 = (x+y+z) / pow(norm, 3);
			val(i, 0) = pow(radius, 3) / 4 * (tmp1 * x - 1 / pow(norm, 3)) + 1 - 3 * radius / 4 * (1 / norm + tmp2 * x);
			val(i, 1) = pow(radius, 3) / 4 * (tmp1 * y - 1 / pow(norm, 3)) + 1 - 3 * radius / 4 * (1 / norm + tmp2 * y);
			val(i, 2) = pow(radius, 3) / 4 * (tmp1 * z - 1 / pow(norm, 3)) + 1 - 3 * radius / 4 * (1 / norm + tmp2 * z);
		}
	}
}

void StokesLawProblem::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols() * pts.cols());
	val.setZero();
}

void StokesLawProblem::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	val.setZero();
}

void StokesLawProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	val.setZero();
	for(int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		if(pts.cols() == 2)
		{
			if (mesh.get_boundary_id(global_ids(i)) != 7)
			{
				val(i, 1) = 1;
			}
		}
		else
		{
			const double z = pts(i, 2);
			const double norm = sqrt(x*x+y*y+z*z);
			const double tmp1 = 3 * (x+y+z) / pow(norm, 5);
			const double tmp2 = (x+y+z) / pow(norm, 3);
			val(i, 0) = pow(radius, 3) / 4 * (tmp1 * x - 1 / pow(norm, 3)) + 1 - 3 * radius / 4 * (1 / norm + tmp2 * x);
			val(i, 1) = pow(radius, 3) / 4 * (tmp1 * y - 1 / pow(norm, 3)) + 1 - 3 * radius / 4 * (1 / norm + tmp2 * y);
			val(i, 2) = pow(radius, 3) / 4 * (tmp1 * z - 1 / pow(norm, 3)) + 1 - 3 * radius / 4 * (1 / norm + tmp2 * z);
		}
	}
}

TaylorGreenVortexProblem::TaylorGreenVortexProblem(const std::string &name)
	: Problem(name), viscosity_(1)
{
	boundary_ids_ = {1, 2, 3, 4, 5, 6, 7};
}

void TaylorGreenVortexProblem::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
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
	const double T = 1;
	for(int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		if(pts.cols() == 2)
		{
			const double time_scaling = exp(-2 * viscosity_ * T * T * t);
			val(i, 0) =  cos(T * x) * sin(T * y) * time_scaling;
			val(i, 1) = -sin(T * x) * cos(T * y) * time_scaling;
		}
		else
		{
			const double z = pts(i, 2);
			const double a = 1.;
			const double time_scaling = -a * exp(-viscosity_ * t);
			val(i, 0) = (exp(a * x)*sin(a * y+z)+exp(a * z)*cos(a * x+y))*time_scaling;
			val(i, 1) = (exp(a * y)*sin(a * z+x)+exp(a * x)*cos(a * y+z))*time_scaling;
			val(i, 2) = (exp(a * z)*sin(a * x+y)+exp(a * y)*sin(a * z+x))*time_scaling;
		}
	}
}

void TaylorGreenVortexProblem::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	const double time_scaling = exp(-2 * viscosity_ * t);

	val.resize(pts.rows(), pts.cols() * pts.cols());

	for (int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		val(i, 0) = -sin(x) * sin(y) * time_scaling;
		val(i, 1) =  cos(x) * cos(y) * time_scaling;
		val(i, 2) = -cos(x) * cos(y) * time_scaling;
		val(i, 3) =  sin(x) * sin(y) * time_scaling;
	}
}

void TaylorGreenVortexProblem::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	val.setZero();
}

void TaylorGreenVortexProblem::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	exact(pts, t, val);
}

template <typename T>
Eigen::Matrix<T, 2, 1> simple_function_const(T x, T y, const double t)
{
	Eigen::Matrix<T, 2, 1> res;

	res(0) = 1;
	res(1) = 1;

	return res;
}

template <typename T>
Eigen::Matrix<T, 3, 1> simple_function_const(T x, T y, T z, const double t)
{
	Eigen::Matrix<T, 3, 1> res;

	res(0) = 1;
	res(1) = 1;
	res(2) = 0;

	return res;
}

template <typename T>
Eigen::Matrix<T, 2, 1> simple_function_lin(T x, T y, const double t)
{
	Eigen::Matrix<T, 2, 1> res;

	res(0) = x / 2. + y;
	res(1) = -x - y / 2.;

	return res;
}

template <typename T>
Eigen::Matrix<T, 3, 1> simple_function_lin(T x, T y, T z, const double t)
{
	Eigen::Matrix<T, 3, 1> res;

	res(0) = x / 2. + y;
	res(1) = -x - y / 2.;
	res(2) = 0;

	return res;
}

template <typename T>
Eigen::Matrix<T, 2, 1> simple_function_cub(T x, T y, const double t)
{
	Eigen::Matrix<T, 2, 1> res;

	res(0) = x * x * x / 3. + x * y * y;
	res(1) = -x * x * y - y * y * y / 3.;

	return res;
}

template <typename T>
Eigen::Matrix<T, 3, 1> simple_function_cub(T x, T y, T z, const double t)
{
	Eigen::Matrix<T, 3, 1> res;

	res(0) = x * x * x / 3. + x * y * y;
	res(1) = -x * x * y - y * y * y / 3.;
	res(2) = 0;

	return res;
}

template <typename T>
Eigen::Matrix<T, 2, 1> simple_function_quad(T x, T y, const double t)
{
	Eigen::Matrix<T, 2, 1> res;

	res(0) = x * x / 2. + x * y;
	res(1) = -x * y - y * y / 2.;

	return res;
}

template <typename T>
Eigen::Matrix<T, 3, 1> simple_function_quad(T x, T y, T z, const double t)
{
	Eigen::Matrix<T, 3, 1> res;

	res(0) = x * x / 2. + x * y;
	res(1) = -x * y - y * y / 2.;
	res(2) = 0;

	return res;
}

template <typename T>
Eigen::Matrix<T, 2, 1> sine_function(T x, T y, const double t)
{
	Eigen::Matrix<T, 2, 1> res;

	res(0) = cos(x) * sin(y);
	res(1) = -sin(x) * cos(y);

	return res;
}

template <typename T>
Eigen::Matrix<T, 3, 1> sine_function(T x, T y, T z, const double t)
{
	Eigen::Matrix<T, 3, 1> res;

	res(0) = cos(x) * sin(y);
	res(1) = -sin(x) * cos(y);
	res(2) = 0;

	return res;
}

SimpleStokeProblemExact::SimpleStokeProblemExact(const std::string &name)
	: ProblemWithSolution(name)
{
	func_ = 0;
}

void SimpleStokeProblemExact::set_parameters(const json &params)
{
	if (params.find("func") != params.end())
	{
		func_ = params["func"];
	}
}

VectorNd SimpleStokeProblemExact::eval_fun(const VectorNd &pt, const double t) const
{
	if (pt.size() == 2){
		switch (func_)
		{
		case 0:
			return simple_function_quad(pt(0), pt(1), t);
		case 1:
			return simple_function_cub(pt(0), pt(1), t);
		case 2:
			return simple_function_lin(pt(0), pt(1), t);
		case 3:
			return simple_function_const(pt(0), pt(1), t);
		}
	}
	else if (pt.size() == 3){
		switch (func_)
		{
		case 0:
			return simple_function_quad(pt(0), pt(1), pt(2), t);
		case 1:
			return simple_function_cub(pt(0), pt(1), pt(2), t);
		case 2:
			return simple_function_lin(pt(0), pt(1), pt(2), t);
		case 3:
			return simple_function_const(pt(0), pt(1), pt(2), t);
		}
	}

	assert(false);
	return VectorNd(pt.size());
}

AutodiffGradPt SimpleStokeProblemExact::eval_fun(const AutodiffGradPt &pt, const double t) const
{
	if (pt.size() == 2)
	{
		switch (func_)
		{
		case 0:
			return simple_function_quad(pt(0), pt(1), t);
		case 1:
			return simple_function_cub(pt(0), pt(1), t);
		case 2:
			return simple_function_lin(pt(0), pt(1), t);
		case 3:
			return simple_function_const(pt(0), pt(1), t);
		}
	}
	else if (pt.size() == 3)
	{
		switch (func_)
		{
		case 0:
			return simple_function_quad(pt(0), pt(1), pt(2), t);
		case 1:
			return simple_function_cub(pt(0), pt(1), pt(2), t);
		case 2:
			return simple_function_lin(pt(0), pt(1), pt(2), t);
		case 3:
			return simple_function_const(pt(0), pt(1), pt(2), t);
		}
	}

	assert(false);
	return AutodiffGradPt(pt.size());
}

AutodiffHessianPt SimpleStokeProblemExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
{
	if (pt.size() == 2)
	{
		switch (func_)
		{
		case 0:
			return simple_function_quad(pt(0), pt(1), t);
		case 1:
			return simple_function_cub(pt(0), pt(1), t);
		case 2:
			return simple_function_lin(pt(0), pt(1), t);
		case 3:
			return simple_function_const(pt(0), pt(1), t);
		}
	}
	else if (pt.size() == 3)
	{
		switch (func_)
		{
		case 0:
			return simple_function_quad(pt(0), pt(1), pt(2), t);
		case 1:
			return simple_function_cub(pt(0), pt(1), pt(2), t);
		case 2:
			return simple_function_lin(pt(0), pt(1), pt(2), t);
		case 3:
			return simple_function_const(pt(0), pt(1), pt(2), t);
		}
	}

	assert(false);
	return AutodiffHessianPt(pt.size());
}


SineStokeProblemExact::SineStokeProblemExact(const std::string &name)
	: ProblemWithSolution(name)
{
}

VectorNd SineStokeProblemExact::eval_fun(const VectorNd &pt, const double t) const
{
	if (pt.size() == 2)
		return sine_function(pt(0), pt(1), t);
	else if (pt.size() == 3)
		return sine_function(pt(0), pt(1), pt(2), t);

	assert(false);
	return VectorNd(pt.size());
}

AutodiffGradPt SineStokeProblemExact::eval_fun(const AutodiffGradPt &pt, const double t) const
{
	if (pt.size() == 2)
		return sine_function(pt(0), pt(1), t);
	else if (pt.size() == 3)
		return sine_function(pt(0), pt(1), pt(2), t);

	assert(false);
	return AutodiffGradPt(pt.size());
}

AutodiffHessianPt SineStokeProblemExact::eval_fun(const AutodiffHessianPt &pt, const double t) const
{
	if (pt.size() == 2)
		return sine_function(pt(0), pt(1), t);
	else if (pt.size() == 3)
		return sine_function(pt(0), pt(1), pt(2), t);

	assert(false);
	return AutodiffHessianPt(pt.size());
}

TransientStokeProblemExact::TransientStokeProblemExact(const std::string &name)
	: Problem(name), func_(0), viscosity_(1)
{
}

void TransientStokeProblemExact::initial_solution(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &pts, Eigen::MatrixXd &val) const
{
	exact(pts, 0, val);
}

void TransientStokeProblemExact::set_parameters(const json &params)
{
	if (params.count("viscosity"))
	{
		viscosity_ = params["viscosity"];
	}

	if (params.find("func") != params.end())
	{
		func_ = params["func"];
	}
}

void TransientStokeProblemExact::exact(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());
	for (int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);

		if(pts.cols() == 2)
		{
			val(i, 0) = -t + x*x / 2 + x* y;
			val(i, 1) = t - x * y - y *y / 2;
		}
		else
		{
			const double z = pts(i, 2);
			val(i, 0) = x*x;
			val(i, 1) = y*y;
			val(i, 2) = -2*z*(x+y);
		}

		// const double w = 1.;
		// const double k = sqrt(0.5 * w / viscosity_);
		// const double Q = 1./(pow(cosh(k), 2) - pow(cos(k), 2));

		// val(i, 0) = Q * ( sinh(k * y) * cos(k * y) * sinh(k) * cos(k) + cosh(k * y) * sin(k * y) * cosh(k) * sin(k) ) * cos(w * t) + Q * ( sinh(k * y) * cos(k * y) * sin(k) * cosh(k) - cosh(k * y) * sin(k * y) * cos(k) * sinh(k) ) * sin(w * t);
		// val(i, 1) = 0;
	}
}

void TransientStokeProblemExact::exact_grad(const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols() * pts.cols());

	// for (int i = 0; i < pts.rows(); ++i)
	// {
	// 	const double x = pts(i, 0);
	// 	const double y = pts(i, 1);

	// 	val(i, 0) = -sin(x) * sin(y) * time_scaling;
	// 	val(i, 1) = cos(x) * cos(y) * time_scaling;
	// 	val(i, 2) = -cos(x) * cos(y) * time_scaling;
	// 	val(i, 3) = sin(x) * sin(y) * time_scaling;
	// }
}

void TransientStokeProblemExact::rhs(const AssemblerUtils &assembler, const std::string &formulation, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	val.resize(pts.rows(), pts.cols());

	for (int i = 0; i < pts.rows(); ++i)
	{
		const double x = pts(i, 0);
		const double y = pts(i, 1);
		if(pts.cols() == 2)
		{
			val(i, 0) = -viscosity_ - t * y + 1. / 2. * x * (x * x + x * y + y * y);
			val(i, 1) =  viscosity_ - t * x + 1. / 2. * y * (x * x + x * y + y * y) + 2;
		}
		else
		{
			const double z = pts(i, 2);
			val(i, 0) = 2*(x*x*x-1);
			val(i, 1) = 2*(y*y*y-1);
			val(i, 2) = 2*z*(x*x+4*x*y+y*y);
		}
	}
	val*=-1;
}

void TransientStokeProblemExact::bc(const Mesh &mesh, const Eigen::MatrixXi &global_ids, const Eigen::MatrixXd &uv, const Eigen::MatrixXd &pts, const double t, Eigen::MatrixXd &val) const
{
	exact(pts, t, val);
}

} // namespace polyfem