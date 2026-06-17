#pragma once

#include <polyfem/legacy/State.hpp>

#include <Eigen/Core>

#include <vector>
#include <string>
#include <memory>

namespace polyfem::solver
{

	/// @brief Validate active geometry nodes selection given states.
	/// @param[in] active_geom_nodes Vector of active geometry node indexes. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	bool is_active_geom_nodes_valid(const Eigen::VectorXi &active_geom_nodes,
									const std::vector<std::shared_ptr<legacy::State>> &states,
									std::string &reason);

	/// @brief Validate active dimensions selection given states.
	/// @param[in] active_dimensions Vector of active dimensions. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	bool is_active_dims_valid(const Eigen::VectorXi &active_dimensions,
							  const std::vector<std::shared_ptr<legacy::State>> &states,
							  std::string &reason);

	/// @brief Validate active solution space dofs selection given states.
	/// @param[in] active_dofs Vector of active state dof indexes. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	bool is_active_dofs_valid(const Eigen::VectorXi &active_dofs,
							  const std::vector<std::shared_ptr<legacy::State>> &states,
							  std::string &reason);

	/// @brief Validate active time slices selection given states.
	/// @param[in] active_time_slices Vector of active time slice indexes. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	bool is_active_time_slices_valid(const Eigen::VectorXi &active_time_slices,
									 const std::vector<std::shared_ptr<legacy::State>> &states,
									 std::string &reason);

	/// @brief Validate active Dirichlet boundary ids selection given states.
	/// @param[in] active_boundary_ids Vector of active Dirichlet boundary ids. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	/// @note We do not support boundary with inactive dimension for now.
	bool is_active_dirichlet_boundary_ids_valid(const Eigen::VectorXi &active_boundary_ids,
												const std::vector<std::shared_ptr<legacy::State>> &states,
												std::string &reason);

	/// @brief Validate active Dirichlet node ids selection given states.
	/// @param[in] active_node_ids Vector of active geometric node ids. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	bool is_active_dirichlet_node_valid(const Eigen::VectorXi &active_dirichlet_nodes,
										const std::vector<std::shared_ptr<legacy::State>> &states,
										std::string &reason);

	/// @brief Validate active pressure boundary ids selection given states.
	/// @param[in] active_boundary_ids Vector of active pressure boundary ids. Empty implies all active.
	/// @param[in] states Shared ptr to all states.
	/// @param[out] reason Reason why validation failed. Set only when invalid.
	bool is_active_pressure_boundary_ids_valid(const Eigen::VectorXi &active_boundary_ids,
											   const std::vector<std::shared_ptr<legacy::State>> &states,
											   std::string &reason);

} // namespace polyfem::solver
