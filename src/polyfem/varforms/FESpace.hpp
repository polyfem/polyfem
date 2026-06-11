#pragma once

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace polyfem::varform
{
	struct GeometryMapping
	{
		std::shared_ptr<std::vector<basis::ElementBases>> bases;
		int n_bases = 0;

		Eigen::VectorXi disc_orders;

		std::map<int, Eigen::MatrixXd> polys;
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		std::shared_ptr<mesh::MeshNodes> mesh_nodes;
	};

	struct FESpace
	{
		int value_dim = 1;
		int n_bases = 0;

		Eigen::VectorXi disc_orders;
		Eigen::VectorXi disc_ordersq;

		std::shared_ptr<std::vector<basis::ElementBases>> bases;
		std::shared_ptr<GeometryMapping> geometry;
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;

		int ndof() const
		{
			return n_bases * value_dim;
		}
	};

	struct AssemblyCaches
	{
		assembler::AssemblyValsCache values;
		assembler::AssemblyValsCache mass;
		assembler::AssemblyValsCache pure_mass;
	};

	struct VarFormBoundaryState
	{
		std::vector<int> boundary_nodes;
		std::vector<mesh::LocalBoundary> total_local_boundary;
		std::vector<mesh::LocalBoundary> local_boundary;
		std::vector<mesh::LocalBoundary> local_neumann_boundary;

		std::vector<mesh::LocalBoundary> local_pressure_boundary;
		std::unordered_map<int, std::vector<mesh::LocalBoundary>> local_pressure_cavity;
		std::vector<int> pressure_boundary_nodes;

		std::vector<int> dirichlet_nodes;
		std::vector<RowVectorNd> dirichlet_nodes_position;
		std::vector<int> neumann_nodes;
		std::vector<RowVectorNd> neumann_nodes_position;
	};

	struct SolutionBlock
	{
		int offset = 0;
		int dof = 0;
		bool is_time_integrated = false;
		bool is_algebraic = false;
	};

	class SolutionLayout
	{
	public:
		int add_block(
			const int dof,
			const bool is_time_integrated,
			const bool is_algebraic = false)
		{
			SolutionBlock block;
			block.offset = total_dof();
			block.dof = dof;
			block.is_time_integrated = is_time_integrated;
			block.is_algebraic = is_algebraic;
			blocks_.push_back(block);
			return int(blocks_.size()) - 1;
		}

		SolutionBlock get_block(const int id) const
		{
			return blocks_.at(id);
		}

		int total_dof() const
		{
			if (blocks_.empty())
				return 0;

			const SolutionBlock &last = blocks_.back();
			return last.offset + last.dof;
		}

	private:
		std::vector<SolutionBlock> blocks_;
	};

	struct ScalarSpaces
	{
		FESpace value;
		SolutionLayout layout;
		int value_block = -1;
	};

	struct ElasticSpaces
	{
		FESpace displacement;
		SolutionLayout layout;
		int displacement_block = -1;
	};

	struct FluidSpaces
	{
		FESpace pressure;

		SolutionLayout layout;
		int velocity_block = -1;
		int pressure_block = -1;
		int pressure_mean_constraint_block = -1;
	};

	struct IncompressibleElasticSpaces
	{
		FESpace pressure;

		SolutionLayout layout;
		int displacement_block = -1;
		int pressure_block = -1;
	};

	struct BilaplacianSpaces
	{
		FESpace helper;

		SolutionLayout layout;
		int value_block = -1;
		int helper_block = -1;
	};
} // namespace polyfem::varform
