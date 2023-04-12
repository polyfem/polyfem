#pragma once

#include "Parametrization.hpp"

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/State.hpp>

#include <Eigen/Core>
#include <map>

namespace polyfem::solver
{
	class VariableToNodes : public CompositeParametrization
	{
	public:
		using CompositeParametrization::CompositeParametrization;

		virtual void set_output_indexing(const std::vector<int> node_ids) final;

	protected:
		int dim;
	};

	class VariableToInteriorNodes : public VariableToNodes
	{
	public:
		VariableToInteriorNodes(const std::vector<std::shared_ptr<Parametrization>> &parametrizations, const State &state, const int volume_selection);
	};

	class VariableToBoundaryNodes : public VariableToNodes
	{
	public:
		VariableToBoundaryNodes(const std::vector<std::shared_ptr<Parametrization>> &parametrizations, const State &state, const int surface_selection);
	};

	class VariableToBoundaryNodesExclusive : public VariableToNodes
	{
	public:
		VariableToBoundaryNodesExclusive(const std::vector<std::shared_ptr<Parametrization>> &parametrizations, const State &state, const std::vector<int> &exclude_surface_selections);
	};
} // namespace polyfem::solver