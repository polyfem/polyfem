////////////////////////////////////////////////////////////////////////////////
#include "LinearSolver.hpp"
#include "CLI11.hpp"
#include <igl/Timer.h>
#include <unsupported/Eigen/SparseExtra>
#include <vector>
#include <algorithm>
////////////////////////////////////////////////////////////////////////////////

void read_triplets(const std::string &filename, Eigen::SparseMatrix<double> &A) {
	std::ifstream in(filename);
	std::vector<Eigen::Triplet<double>> T;

	// Count lines
	int nl = std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
	in.seekg(in.beg);
	T.reserve(nl);

	// Read triplets
	int i, j;
	double k;
	int n = 0, m = 0;
	while (in >> i >> j >> k) {
		T.emplace_back(i, j, k);
		n = std::max(i+1, n);
		m = std::max(j+1, m);
	}

	std::cout << "-- Building sparse matrix" << std::endl;
	A.resize(n, m);
	A.setFromTriplets(T.begin(), T.end());
}

void read_vector(const std::string &filename, Eigen::VectorXd &X) {
	std::ifstream in(filename);
	double v;
	std::vector<double> T;
	while (in >> v) {
		T.push_back(v);
	}
	X.resize(T.size());
	for (size_t i = 0; i < T.size(); ++i) {
		X[i] = T[i];
	}
}

void write_vector(const std::string &filename, Eigen::VectorXd X) {
	std::ofstream out(filename);
	out << (X.array()).format(Eigen::IOFormat(
		Eigen::FullPrecision, Eigen::DontAlignCols," ","\n","","","","\n"));
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]) {
	std::ios_base::sync_with_stdio(false);

	// Default arguments
	struct {
		std::string matrix = "stiffness.txt";
		std::string rhs = "rhs.txt";
		std::string output = "";
		std::string json = "";
		std::string solver = "Eigen::SimplicialLDLT";
		std::string precond = "";
		bool list_solvers = false;
		bool list_precond = false;
		bool check_spd = false;
	} args;

	// Parse arguments
	CLI::App app{"solver"};
	app.add_option("matrix,-m,--matrix", args.matrix, "Matrix of the linear system.")->required();
	app.add_option("rhs,-r,--rhs", args.rhs, "Right-hand side vector.")->required();
	app.add_option("-o,--output", args.output, "Output solution.");
	app.add_option("-j,--json", args.json, "Write json file with statistics.");
	app.add_option("-s,--solver", args.solver, "Solver to use.");
	app.add_option("-p,--precond", args.precond, "Preconditioner (for iterative solvers).");
	app.add_option("-S,--list_solvers", args.list_solvers);
	app.add_option("-P,--list_precond", args.list_precond);
	app.add_option("--check_spd", args.check_spd);
	try {
		app.parse(argc, argv);
	} catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	// List available options and quit
	if (args.list_solvers) {
		for (std::string s : poly_fem::LinearSolver::availableSolvers()) {
			std::cout << s << std::endl;
		}
		return 0;
	}

	if (args.list_precond) {
		for (std::string s : poly_fem::LinearSolver::availablePrecond()) {
			std::cout << s << std::endl;
		}
		return 0;
	}

	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd x, b;
	json params;
	params["solver"] = args.solver;
	params["precond"] = args.precond;

	// Read input
	std::cout << "-- Reading triplets" << std::endl;
	//read_triplets(args.matrix, A);
	//Eigen::saveMarket(A, "A.mat");
	Eigen::loadMarket(A, args.matrix);
	std::cout << "-- Reading rhs" << std::endl;
	// read_vector(args.rhs, b);
	// Eigen::saveMarketVector(b, "b.mat");
	Eigen::loadMarketVector(b, args.rhs);
	x.resizeLike(b);
	x.setZero();

	params["size"] = x.size();
	params["non_zeros"] = A.nonZeros();
	std::cout << "-- Sparsity: " << A.nonZeros() << " / " << A.size() << std::endl;

	// Check whether the system is SPD
	if (args.check_spd) {
		Eigen::SparseMatrix<double> AT = A.transpose();
		// std::cout << std::endl << std::endl;
		std::cout << A << std::endl;
		// std::cout << std::endl << std::endl;
		// std::cout << AT << std::endl;
		// std::cout << std::endl << std::endl;
		if (A.isApprox(AT)) {
			std::cout << "-- Matrix is symmetric" << std::endl;
		} else {
			std::cout << "-- Matrix is not symmetric" << std::endl;
		}
		// Compute the Cholesky decomposition of A
		Eigen::LLT<Eigen::MatrixXd> lltOfA(A);
		if (lltOfA.info() == Eigen::NumericalIssue) {
			std::cout << "-- Possibly indefinite matrix!" << std::endl;
		} else {
			//std::cout << "-- Matrix is positive or negative semidefinite." << std::endl;
			std::cout << "-- Matrix is positive definite." << std::endl;
		}
		Eigen::LDLT<Eigen::MatrixXd> ldltOfA(A);
		if (ldltOfA.info() == Eigen::NumericalIssue) {
			std::cout << "-- Possibly indefinite matrix!" << std::endl;
		} else {
			std::cout << "-- Matrix is positive or negative semidefinite." << std::endl;
		}
	}

	// Solve
	igl::Timer timer;
	auto solver = poly_fem::LinearSolver::create(args.solver, args.precond);
	std::cout << "-- Using solver: " << solver->name() << std::endl;

	std::cout << "-- Analyzing sparsity pattern..." << std::endl;
	timer.start();
	solver->analyzePattern(A);
	timer.stop();
	params["time_analyze"] = timer.getElapsedTime();

	std::cout << "-- Factorizing matrix..." << std::endl;
	timer.start();
	solver->factorize(A);
	timer.stop();
	params["time_factorize"] = timer.getElapsedTime();

	std::cout << "-- Solving linear system..." << std::endl;
	timer.start();
	solver->solve(b, x);
	timer.stop();
	params["time_solve"] = timer.getElapsedTime();
	solver->getInfo(params);
	params["error"] = (A*x-b).norm();

	// Write output
	if (!args.output.empty()) {
		write_vector(args.output, x);
	}
	if (args.json.empty()) {
		std::cout << params.dump(4) << std::endl;
	} else {
		std::ofstream out(args.json);
		out << params.dump(4) << std::endl;
	}

	return 0;
}
