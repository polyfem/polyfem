#include "SolverCSVWriter.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/SolveData.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/getRSS.h>

#include <spdlog/fmt/fmt.h>

namespace polyfem::io
{
	EnergyCSVWriter::EnergyCSVWriter(const std::string &path, const solver::SolveData &solve_data)
		: file(path), solve_data(solve_data)
	{
		file << "i,";
		for (const auto &[name, _] : solve_data.named_forms())
			file << name << ",";
		file << "total_energy" << std::endl;
	}

	EnergyCSVWriter::~EnergyCSVWriter()
	{
		file.close();
	}

	void EnergyCSVWriter::write(const int i, const Eigen::MatrixXd &sol)
	{
		const double s = solve_data.time_integrator
							 ? solve_data.time_integrator->acceleration_scaling()
							 : 1;
		file << i << ",";
		for (const auto &[_, form] : solve_data.named_forms())
		{
			// Divide by acceleration scaling to get the energy (units of J).
			file << ((form && form->enabled()) ? form->value(sol) : 0) / s << ",";
		}
		file << solve_data.nl_problem->value(sol) / s << "\n";
		file.flush();
	}

	RuntimeStatsCSVWriter::RuntimeStatsCSVWriter(
		const std::string &path,
		const int n_bases,
		const int n_elements,
		const double t0,
		const double dt)
		: file(path), n_bases(n_bases), n_elements(n_elements), t0(t0), dt(dt)
	{
		file << "step,time,forward,remeshing,global_relaxation,peak_mem,#V,#T" << std::endl;
	}

	RuntimeStatsCSVWriter::~RuntimeStatsCSVWriter()
	{
		file.close();
	}

	void RuntimeStatsCSVWriter::write(
		const int t,
		const double forward,
		const double remeshing,
		const double global_relaxation)
	{
		const double peak_mem = getPeakRSS() / double(1 << 30);
		file << fmt::format(
			"{},{},{},{},{},{},{},{}\n",
			t, t0 + dt * t, forward, remeshing, global_relaxation, peak_mem,
			n_bases, n_elements);
		file.flush();
	}
} // namespace polyfem::io
