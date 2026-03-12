#pragma once

#include <vector>
#include <Eigen/Core>

namespace polyfem
{
	class State;
}

namespace polyfem::solver
{
	class VariableToNodes
	{
	public:
		VariableToNodes(const State &state, const std::vector<int> &active_dimensions);
		virtual ~VariableToNodes() {}
		virtual void set_output_indexing(const std::vector<int> &node_ids) final;
		const Eigen::VectorXi &get_output_indexing() const { return output_indexing_; }

	protected:
		std::vector<int> active_dimensions_;
		int dim;

		Eigen::VectorXi output_indexing_;
	};

	class VariableToInteriorNodes : public VariableToNodes
	{
	public:
		VariableToInteriorNodes(const State &state, const std::vector<int> &active_dimensions, const std::vector<int> &volume_selection);
	};

	class VariableToBoundaryNodes : public VariableToNodes
	{
	public:
		VariableToBoundaryNodes(const State &state, const std::vector<int> &active_dimensions, const std::vector<int> &surface_selection);
	};

	class VariableToBoundaryNodesExclusive : public VariableToNodes
	{
	public:
		VariableToBoundaryNodesExclusive(const State &state, const std::vector<int> &active_dimensions, const std::vector<int> &exclude_surface_selections);
	};
} // namespace polyfem::solver