#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

#include <spdlog/spdlog.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace GEO
{
	class Mesh;
}

namespace polyfem::varform
{
	class VarForm;
}

namespace polyfem
{
	/// VarForm-only simulation state.
	class State
	{
	public:
		State();
		~State() = default;

		/// initialize the polyfem solver with a json settings
		/// @param[in] args input arguments
		/// @param[in] strict_validation strict validation of input
		void init(const json &args, const bool strict_validation);

		/// @param[in] max_threads max number of threads
		void set_max_threads(const int max_threads = std::numeric_limits<int>::max());

		/// main input arguments containing all defaults
		json args;

		/// active variational formulation
		std::shared_ptr<varform::VarForm> variational_formulation;

		/// Optional UI progress callback.
		std::function<void(int, int, double, double)> time_callback = nullptr;

		//---------------------------------------------------
		//-----------------logger----------------------------
		//---------------------------------------------------

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

	private:
		/// initializing the logger meant for internal usage
		void init_logger(const std::vector<spdlog::sink_ptr> &sinks, const spdlog::level::level_enum log_level);

		/// logger sink to stdout
		spdlog::sink_ptr console_sink_ = nullptr;
		spdlog::sink_ptr file_sink_ = nullptr;

	public:
		/// solves the problem, call other methods
		void solve(Eigen::MatrixXd &sol);

		//---------------------------------------------------
		//-----------------Geometry--------------------------
		//---------------------------------------------------

		/// loads the mesh from the json arguments
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		/// @param[in] names keys in the hdf5
		/// @param[in] cells list of cells from hdf5
		/// @param[in] vertices list of vertices from hdf5
		void load_mesh(bool non_conforming = false,
					   const std::vector<std::string> &names = std::vector<std::string>(),
					   const std::vector<Eigen::MatrixXi> &cells = std::vector<Eigen::MatrixXi>(),
					   const std::vector<Eigen::MatrixXd> &vertices = std::vector<Eigen::MatrixXd>());

		/// loads the mesh from a geogram mesh
		/// @param[in] meshin geo mesh
		/// @param[in] boundary_marker the input of the lambda is the face barycenter, the output is the sideset id
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		/// @param[in] skip_boundary_sideset skip_boundary_sideset = false it uses the lambda boundary_marker to assign the sideset
		void load_mesh(GEO::Mesh &meshin, const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &boundary_marker, bool non_conforming = false, bool skip_boundary_sideset = false);

		/// loads the mesh from V and F,
		/// @param[in] V is #vertices x dim
		/// @param[in] F is #elements x size (size = 3 for triangle mesh, size=4 for a quad mesh if dim is 2)
		/// @param[in] non_conforming creates a conforming/non conforming mesh
		void load_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, bool non_conforming = false);
	};

} // namespace polyfem
