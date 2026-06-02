#include <polyfem/State.hpp>

#include <polyfem/io/VarFormOutputWriter.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include <filesystem>
#include <stdexcept>

namespace polyfem
{
	namespace
	{
		const varform::VarForm &require_varform(const State &state)
		{
			if (!state.variational_formulation)
				throw std::runtime_error("polyfem::State is varform-only; use polyfem::legacy::State for legacy formulations.");
			return *state.variational_formulation;
		}
	}

	void State::compute_errors(const Eigen::MatrixXd &sol)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return;

		const io::OutputState output = require_varform(*this).output_state();

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		stats = output.stats;
		stats.compute_errors(output.n_bases, output.bases, output.geom_bases(), *output.mesh, *output.problem, tend, sol);
	}

	std::string State::root_path() const
	{
		if (utils::is_param_valid(args, "root_path"))
			return args["root_path"].get<std::string>();
		return "";
	}

	std::string State::resolve_input_path(const std::string &path, const bool only_if_exists) const
	{
		return utils::resolve_path(path, root_path(), only_if_exists);
	}

	std::string State::resolve_output_path(const std::string &path) const
	{
		if (output_dir.empty() || path.empty() || std::filesystem::path(path).is_absolute())
			return path;

		return std::filesystem::weakly_canonical(std::filesystem::path(output_dir) / path).string();
	}

	void State::save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &)
	{
		io::VarFormOutputWriter(require_varform(*this)).save_timestep(time, t, t0, dt, sol);
	}

	void State::save_json(const Eigen::MatrixXd &sol)
	{
		io::VarFormOutputWriter(require_varform(*this)).save_json(sol);
	}

	void State::save_json(const Eigen::MatrixXd &sol, std::ostream &out)
	{
		io::VarFormOutputWriter(require_varform(*this)).save_json(sol, out);
	}

	void State::save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &)
	{
		io::VarFormOutputWriter(require_varform(*this)).save_subsolve(i, t, sol);
	}

	void State::export_data(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &)
	{
		io::VarFormOutputWriter(require_varform(*this)).export_data(sol);
	}

	void State::save_restart_json(const double t0, const double dt, const int t) const
	{
		io::VarFormOutputWriter(require_varform(*this)).save_restart_json(t0, dt, t);
	}
} // namespace polyfem
