#pragma once

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <fstream>
#include <string>

namespace polyfem::solver
{
	class SolveData;
}

namespace polyfem::io
{
	class EnergyCSVWriter
	{
	public:
		EnergyCSVWriter(const std::string &path, const solver::SolveData &solve_data);
		~EnergyCSVWriter();

		void write(const int i, const Eigen::MatrixXd &sol);

	private:
		std::ofstream file;
		const solver::SolveData &solve_data;
	};

	class RuntimeStatsCSVWriter
	{
	public:
		RuntimeStatsCSVWriter(const std::string &path, const int n_bases, const int n_elements, const double t0, const double dt);
		~RuntimeStatsCSVWriter();

		void write(const int t, const double forward, const double remeshing, const double global_relaxation);

	private:
		std::ofstream file;
		const int n_bases;
		const int n_elements;
		const double t0;
		const double dt;
	};
} // namespace polyfem::io
