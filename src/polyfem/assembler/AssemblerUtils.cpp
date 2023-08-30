
#include "AssemblerUtils.hpp"

#include <polyfem/assembler/Bilaplacian.hpp>
#include <polyfem/assembler/Helmholtz.hpp>
#include <polyfem/assembler/HookeLinearElasticity.hpp>
#include <polyfem/assembler/IncompressibleLinElast.hpp>
#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/LinearElasticity.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/MooneyRivlinElasticity.hpp>
#include <polyfem/assembler/AMIPSEnergy.hpp>
#include <polyfem/assembler/MultiModel.hpp>
#include <polyfem/assembler/NavierStokes.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/assembler/OgdenElasticity.hpp>
#include <polyfem/assembler/SaintVenantElasticity.hpp>
#include <polyfem/assembler/Stokes.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

// #include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	using namespace basis;
	using namespace utils;

	namespace assembler
	{

		std::string AssemblerUtils::other_assembler_name(const std::string &formulation)
		{
			if (formulation == "Bilaplacian")
				return "BilaplacianAux";
			else if (formulation == "Stokes" || formulation == "NavierStokes" || formulation == "OperatorSplitting")
				return "StokesPressure";
			else if (formulation == "IncompressibleLinearElasticity")
				return "IncompressibleLinearElasticityPressure";

			return "";
		}

		std::shared_ptr<Assembler> AssemblerUtils::make_assembler(const std::string &formulation)
		{
			if (formulation == "Helmholtz")
				return std::make_shared<Helmholtz>();
			else if (formulation == "Laplacian")
				return std::make_shared<Laplacian>();
			else if (formulation == "Bilaplacian")
				return std::make_shared<BilaplacianMain>();
			else if (formulation == "BilaplacianAux")
				return std::make_shared<BilaplacianAux>();

			else if (formulation == "LinearElasticity")
				return std::make_shared<LinearElasticity>();
			else if (formulation == "HookeLinearElasticity")
				return std::make_shared<HookeLinearElasticity>();
			else if (formulation == "IncompressibleLinearElasticity")
				return std::make_shared<IncompressibleLinearElasticityDispacement>();
			else if (formulation == "IncompressibleLinearElasticityPressure")
				return std::make_shared<IncompressibleLinearElasticityPressure>();

			else if (formulation == "SaintVenant")
				return std::make_shared<SaintVenantElasticity>();
			else if (formulation == "NeoHookean")
				return std::make_shared<NeoHookeanElasticity>();
			else if (formulation == "MooneyRivlin")
				return std::make_shared<MooneyRivlinElasticity>();
			else if (formulation == "MultiModels")
				return std::make_shared<MultiModel>();
			else if (formulation == "UnconstrainedOgden")
				return std::make_shared<UnconstrainedOgdenElasticity>();
			else if (formulation == "IncompressibleOgden")
				return std::make_shared<IncompressibleOgdenElasticity>();

			else if (formulation == "Stokes")
				return std::make_shared<StokesVelocity>();
			else if (formulation == "StokesPressure")
				return std::make_shared<StokesPressure>();
			else if (formulation == "NavierStokes")
				return std::make_shared<NavierStokesVelocity>();
			else if (formulation == "OperatorSplitting")
				return std::make_shared<OperatorSplitting>();

			else if (formulation == "AMIPS")
				return std::make_shared<AMIPSEnergy>();

			log_and_throw_error("Inavalid assembler name {}", formulation);
		}

		std::shared_ptr<MixedAssembler> AssemblerUtils::make_mixed_assembler(const std::string &formulation)
		{
			if (formulation == "Bilaplacian")
				return std::make_shared<BilaplacianMixed>();
			else if (formulation == "IncompressibleLinearElasticity")
				return std::make_shared<IncompressibleLinearElasticityMixed>();
			else if (formulation == "Stokes" || formulation == "NavierStokes" || formulation == "OperatorSplitting")
				return std::make_shared<StokesMixed>();

			log_and_throw_error("Inavalid mixed assembler name {}", formulation);
		}

		void AssemblerUtils::merge_mixed_matrices(
			const int n_bases, const int n_pressure_bases, const int problem_dim, const bool add_average,
			const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
			StiffnessMatrix &stiffness)
		{
			assert(velocity_stiffness.rows() == velocity_stiffness.cols());
			assert(velocity_stiffness.rows() == n_bases * problem_dim);

			assert(mixed_stiffness.size() == 0 || mixed_stiffness.rows() == n_bases * problem_dim);
			assert(mixed_stiffness.size() == 0 || mixed_stiffness.cols() == n_pressure_bases);

			assert(pressure_stiffness.size() == 0 || pressure_stiffness.rows() == n_pressure_bases);
			assert(pressure_stiffness.size() == 0 || pressure_stiffness.cols() == n_pressure_bases);

			const int avg_offset = add_average ? 1 : 0;

			std::vector<Eigen::Triplet<double>> blocks;
			blocks.reserve(velocity_stiffness.nonZeros() + 2 * mixed_stiffness.nonZeros() + pressure_stiffness.nonZeros() + 2 * avg_offset * velocity_stiffness.rows());

			for (int k = 0; k < velocity_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(velocity_stiffness, k); it; ++it)
				{
					blocks.emplace_back(it.row(), it.col(), it.value());
				}
			}

			for (int k = 0; k < mixed_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(mixed_stiffness, k); it; ++it)
				{
					blocks.emplace_back(it.row(), n_bases * problem_dim + it.col(), it.value());
					blocks.emplace_back(it.col() + n_bases * problem_dim, it.row(), it.value());
				}
			}

			for (int k = 0; k < pressure_stiffness.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(pressure_stiffness, k); it; ++it)
				{
					blocks.emplace_back(n_bases * problem_dim + it.row(), n_bases * problem_dim + it.col(), it.value());
				}
			}

			if (add_average)
			{
				const double val = 1.0 / n_pressure_bases;
				for (int i = 0; i < n_pressure_bases; ++i)
				{
					blocks.emplace_back(n_bases * problem_dim + i, n_bases * problem_dim + n_pressure_bases, val);
					blocks.emplace_back(n_bases * problem_dim + n_pressure_bases, n_bases * problem_dim + i, val);
				}
			}

			stiffness.resize(n_bases * problem_dim + n_pressure_bases + avg_offset, n_bases * problem_dim + n_pressure_bases + avg_offset);
			stiffness.setFromTriplets(blocks.begin(), blocks.end());
			stiffness.makeCompressed();

			// static int c = 0;
			// Eigen::saveMarket(stiffness, "stiffness.txt");
			// Eigen::saveMarket(velocity_stiffness, "velocity_stiffness.txt");
			// Eigen::saveMarket(mixed_stiffness, "mixed_stiffness.txt");
			// Eigen::saveMarket(pressure_stiffness, "pressure_stiffness.txt");
		}

		int AssemblerUtils::quadrature_order(const std::string &assembler, const int basis_degree, const BasisType &b_type, const int dim)
		{
			if (assembler == "Mass")
			{
				if (b_type == BasisType::SIMPLEX_LAGRANGE || b_type == BasisType::CUBE_LAGRANGE)
					return std::max(basis_degree * 2, 1);
				else
					return basis_degree * 2 + 1;
			}
			else if (assembler == "NavierStokes")
			{
				if (b_type == BasisType::SIMPLEX_LAGRANGE)
					return std::max((basis_degree - 1) + basis_degree, 1);
				else if (b_type == BasisType::CUBE_LAGRANGE)
					return std::max(basis_degree * 2, 1);
				else
					return basis_degree * 2 + 1;
			}
			else
			{
				if (b_type == BasisType::SIMPLEX_LAGRANGE)
					return std::max((basis_degree - 1) * 2, 1);
				else if (b_type == BasisType::CUBE_LAGRANGE)
					return std::max(basis_degree * 2, 1);
				else
					return (basis_degree - 1) * 2 + 1;
			}
		}

	} // namespace assembler
} // namespace polyfem
