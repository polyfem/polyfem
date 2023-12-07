#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem
{
	class State;

	namespace solver
	{
		class VariableToSimulation;
		class AdjointNLProblem;
	} // namespace solver

	/// main class that contains the polyfem adjoint solver and all its state
	class OptState
	{
	public:
		//---------------------------------------------------
		//-----------------initialization--------------------
		//---------------------------------------------------

		~OptState() = default;
		/// Constructor
		OptState();

		/// initialize the polyfem solver with a json settings
		/// @param[in] args input arguments
		/// @param[in] strict_validation strict validation of input
		void init(const json &args, const bool strict_validation);

		/// main input arguments containing all defaults
		json args;

		/// initializing the logger
		/// @param[in] log_file is to write it to a file (use log_file="") to output to stdout
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		/// @param[in] file_log_level 0 all message, 6 no message. 2 is info, 1 is debug
		/// @param[in] is_quit quiets the log
		void init_logger(
			const std::string &log_file,
			const spdlog::level::level_enum log_level,
			const spdlog::level::level_enum file_log_level,
			const bool is_quiet);

		/// initializing the logger writes to an output stream
		/// @param[in] os output stream
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void init_logger(std::ostream &os, const spdlog::level::level_enum log_level);

		/// change log level
		/// @param[in] log_level 0 all message, 6 no message. 2 is info, 1 is debug
		void set_log_level(const spdlog::level::level_enum log_level);

		/// @brief create the opt states
		void create_states(const spdlog::level::level_enum &log_level = spdlog::level::level_enum::trace, const int max_threads = -1);

		/// init variables
		void init_variables();

		void crate_problem();

		void initial_guess(Eigen::VectorXd &x); // shoud be const

		double eval(Eigen::VectorXd &x) const;

		void solve(Eigen::VectorXd &x);

	private:
		inline std::string root_path() const
		{
			if (utils::is_param_valid(args, "root_path"))
				return args["root_path"].get<std::string>();
			return "";
		}

		/// initializing the logger meant for internal usage
		void init_logger(const std::vector<spdlog::sink_ptr> &sinks, const spdlog::level::level_enum log_level);

		/// logger sink to stdout
		spdlog::sink_ptr console_sink_ = nullptr;
		spdlog::sink_ptr file_sink_ = nullptr;

		//---------------------------------------------------
		//-----------------state--------------------
		//---------------------------------------------------

		/// State used in the opt
		std::vector<std::shared_ptr<State>> states;

		/// @brief variables
		std::vector<int> variable_sizes;
		int ndof;

		std::vector<std::shared_ptr<solver::VariableToSimulation>> variable_to_simulations;

		std::shared_ptr<solver::AdjointNLProblem> nl_problem;

	public:
		/// Directory for output files
		std::string output_dir;
	};
} // namespace polyfem