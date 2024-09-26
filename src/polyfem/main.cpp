#include <filesystem>
#include <chrono>

#include <CLI/CLI.hpp>

#include <h5pp/h5pp.h>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/quadrature/TriQuadrature.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/quadrature/TetQuadrature.hpp>
#include <polyfem/quadrature/HexQuadrature.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/Rational.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>

using namespace polyfem;
using namespace basis;
using namespace utils;
using namespace quadrature;
using namespace assembler;

enum class CellType
{
	Tri, Quad, Tet, Hex
};

class Timer {
	private:
	using myClock = std::chrono::steady_clock;
	myClock::time_point startTime;
	myClock::duration duration = myClock::duration::zero();
	bool running = false;

	public:
	void start() {
		if (!running) {
			startTime = myClock::now();
			running = true;
		}
	}
	void stop() {
		if (running) {
			duration = myClock::now() - startTime;
			running = false;
		}
	}
	void reset() { duration = myClock::duration::zero(); running = false; }


	template <typename U>
	typename U::rep read() const {
		return std::chrono::duration_cast<U>(duration).count();
	}
};

int main(int argc, char **argv)
{
	using namespace polyfem;

	CLI::App command_line{"polyfem"};

	command_line.ignore_case();
	command_line.ignore_underscore();

	// Eigen::setNbThreads(1);
	unsigned max_threads = std::numeric_limits<unsigned>::max();
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	int discr_order = 0;
	command_line.add_option("--order", discr_order, "Polynomial order of FE basis");

	const std::vector<std::pair<std::string, CellType>>
		CELL_TYPES_TO_LEVELS = {
			{"tri", CellType::Tri},
			{"tet", CellType::Tet},
			{"hex", CellType::Hex},
			{"quad", CellType::Quad}};
	
	CellType cell_type = CellType::Tri;
	command_line.add_option("--cell", cell_type, "Cell type: tri, tet, hex, quad")
		->transform(CLI::CheckedTransformer(CELL_TYPES_TO_LEVELS, CLI::ignore_case));

	auto input = command_line.add_option_group("input");

	std::string hdf5_file = "";
	input->add_option("--hdf5", hdf5_file, "Simulation HDF5 file")->check(CLI::ExistingFile);

	input->require_option(1);

	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	spdlog::level::level_enum log_level = spdlog::level::debug;
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

	std::vector<spdlog::sink_ptr> sinks = {std::make_shared<spdlog::sinks::stdout_color_sink_mt>()};
	set_logger(std::make_shared<spdlog::logger>("polyfem", sinks.begin(), sinks.end()));
	logger().set_level(log_level);

	NThread::get().set_num_threads(max_threads);

	if (!hdf5_file.empty())
	{
		h5pp::File file(hdf5_file, h5pp::FileAccess::READONLY);

		// auto dsetInfo = file.getDatasetInfo("NumberOfSimplices");
		// logger().info("hdf5 info: {}", dsetInfo.string());

		const int n_elem = file.readDataset<Eigen::Vector<long int, -1>>("NumberOfSimplices")(0);
		const int dim = file.readDataset<Eigen::Vector<long int, -1>>("Dimension")(0);
		const int n_loc_nodes = file.readDataset<Eigen::Vector<long int, -1>>("NumberOfHighOrderNodes")(0);

		logger().info("Load rational nodes ...");
		const std::vector<std::string> nodes_vec = file.readDataset<std::vector<std::string>>("Nodes");
		if (nodes_vec.size() != 4 * dim * n_loc_nodes * n_elem)
			log_and_throw_error("Invalid node array size! Expect {}, Actual {}", 4 * dim * n_loc_nodes * n_elem, nodes_vec.size());
		
		logger().info("Convert rational nodes to floating point ...");
		std::vector<Eigen::MatrixXd> nodesA, nodesB;
		nodesA.assign(n_elem, Eigen::MatrixXd::Zero(n_loc_nodes, dim));
		nodesB.assign(n_elem, Eigen::MatrixXd::Zero(n_loc_nodes, dim));
		for (int e = 0, id = 0; e < n_elem; e++)
		{
			for (int l = 0; l < n_loc_nodes; l++)
			{
				for (int d = 0; d < dim; d++)
				{
					Rational rat;
					nodesA[e](l, d) = rat.get_double(nodes_vec[id + 0], nodes_vec[id + 1]);
					nodesB[e](l, d) = rat.get_double(nodes_vec[id + 2], nodes_vec[id + 3]);
					id += 4;
				}
			}
		}

		logger().info("Perform Jacobian check on quadrature points ...");
		const std::vector<int> triangle_elem_nodes = {3, 6, 10, 15, 21, 28};
		const std::vector<int> tet_elem_nodes = {4, 10, 20, 35, 56};
		const std::vector<int> quad_elem_nodes = {4, 9, 16, 25};
		const std::vector<int> hex_elem_nodes = {8, 27, 64};

		switch (cell_type)
		{
		case CellType::Tri:
			if (n_loc_nodes != triangle_elem_nodes[discr_order])
				log_and_throw_error("Invalid number of local nodes {}! Expect {}", n_loc_nodes, triangle_elem_nodes[discr_order]);
		break;
		case CellType::Tet:
			if (n_loc_nodes != tet_elem_nodes[discr_order])
				log_and_throw_error("Invalid number of local nodes {}! Expect {}", n_loc_nodes, tet_elem_nodes[discr_order]);
		break;
		case CellType::Quad:
			if (n_loc_nodes != quad_elem_nodes[discr_order])
				log_and_throw_error("Invalid number of local nodes {}! Expect {}", n_loc_nodes, quad_elem_nodes[discr_order]);
		break;
		case CellType::Hex:
			if (n_loc_nodes != hex_elem_nodes[discr_order])
				log_and_throw_error("Invalid number of local nodes {}! Expect {}", n_loc_nodes, hex_elem_nodes[discr_order]);
		break;
		}

		std::vector<basis::ElementBases> bases(nodesB.size());
		int real_order = 1;
		const std::string assembler = "NeoHookean";
		switch (cell_type)
		{
		case CellType::Tri:
			real_order = AssemblerUtils::quadrature_order(assembler, discr_order, AssemblerUtils::BasisType::SIMPLEX_LAGRANGE, dim);

			for (int e = 0; e < nodesB.size(); e++)
			{
				ElementBases &b = bases[e];
				b.bases.resize(n_loc_nodes);
				b.set_quadrature([real_order](Quadrature &quad) {
					TriQuadrature tri_quadrature;
					tri_quadrature.get_quadrature(real_order, quad);
				});

				for (int j = 0; j < n_loc_nodes; ++j)
				{
					b.bases[j].init(discr_order, j, j, nodesB[e].row(j));

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::p_basis_value_2d(discr_order, j, uv, val); });
					b.bases[j].set_grad([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::p_grad_basis_value_2d(discr_order, j, uv, val); });
				}
			}
		break;
		case CellType::Tet:
			real_order = AssemblerUtils::quadrature_order(assembler, discr_order, AssemblerUtils::BasisType::SIMPLEX_LAGRANGE, dim);

			for (int e = 0; e < nodesB.size(); e++)
			{
				ElementBases &b = bases[e];
				b.bases.resize(n_loc_nodes);
				b.set_quadrature([real_order](Quadrature &quad) {
					TetQuadrature tet_quadrature;
					tet_quadrature.get_quadrature(real_order, quad);
				});

				for (int j = 0; j < n_loc_nodes; ++j)
				{
					b.bases[j].init(discr_order, j, j, nodesB[e].row(j));

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::p_basis_value_3d(discr_order, j, uv, val); });
					b.bases[j].set_grad([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::p_grad_basis_value_3d(discr_order, j, uv, val); });
				}
			}
		break;
		case CellType::Quad:
			real_order = AssemblerUtils::quadrature_order(assembler, discr_order, AssemblerUtils::BasisType::CUBE_LAGRANGE, dim);

			for (int e = 0; e < nodesB.size(); e++)
			{
				ElementBases &b = bases[e];
				b.bases.resize(n_loc_nodes);
				b.set_quadrature([real_order](Quadrature &quad) {
					QuadQuadrature quad_quadrature;
					quad_quadrature.get_quadrature(real_order, quad);
				});

				for (int j = 0; j < n_loc_nodes; ++j)
				{
					b.bases[j].init(discr_order, j, j, nodesB[e].row(j));

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_2d(discr_order, j, uv, val); });
					b.bases[j].set_grad([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_2d(discr_order, j, uv, val); });
				}
			}
		break;
		case CellType::Hex:
			real_order = AssemblerUtils::quadrature_order(assembler, discr_order, AssemblerUtils::BasisType::CUBE_LAGRANGE, dim);

			for (int e = 0; e < nodesB.size(); e++)
			{
				ElementBases &b = bases[e];
				b.bases.resize(n_loc_nodes);
				b.set_quadrature([real_order](Quadrature &quad) {
					HexQuadrature hex_quadrature;
					hex_quadrature.get_quadrature(real_order, quad);
				});

				for (int j = 0; j < n_loc_nodes; ++j)
				{
					b.bases[j].init(discr_order, j, j, nodesB[e].row(j));

					b.bases[j].set_basis([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_basis_value_3d(discr_order, j, uv, val); });
					b.bases[j].set_grad([discr_order, j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { autogen::q_grad_basis_value_3d(discr_order, j, uv, val); });
				}
			}
		break;
		}

		Timer timer;
		timer.start();

		std::vector<bool> result(bases.size());
		for (int e = 0; e < bases.size(); e++)
		{
			assembler::ElementAssemblyValues vals;
			result[e] = vals.is_geom_mapping_positive(dim == 3, bases[e]);
		}

		timer.stop();
		const double microseconds = static_cast<double>(timer.read<std::chrono::nanoseconds>()) / 1000;

		for (bool i : result)
			std::cout << i << ' ';

		std::cout << std::endl;
		logger().info("Checked {} elements in {} ms", n_elem, microseconds);
	}
	else
		log_and_throw_error("No HDF5 input specified!");

	return EXIT_SUCCESS;
}
