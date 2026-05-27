#pragma once

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/solver/SolveData.hpp>

#include <ipc/collision_mesh.hpp>

#include <map>

namespace polyfem::varform
{
	class ElasticVarForm : public VarForm
	{
	public:
		io::OutputState output_state() const override;
		std::vector<io::OutputField> output_fields(
			const io::OutputSample &sample,
			const Eigen::MatrixXd &solution,
			const io::OutputFieldOptions &options) const override;

	protected:
		QuadratureOrders n_boundary_samples() const;

		/// @brief Get a constant reference to the geometry mapping bases.
		/// @return A constant reference to the geometry mapping bases.
		const std::vector<basis::ElementBases> &geom_bases() const
		{
			return iso_parametric ? bases : geom_bases_;
		}

		/// assembler corresponding to governing physical equations
		std::shared_ptr<assembler::Assembler> assembler = nullptr;
		std::shared_ptr<assembler::Mass> mass_matrix_assembler = nullptr;

		/// FE bases, the size is #elements
		std::vector<basis::ElementBases> bases;

		/// number of bases
		int n_bases = 0;

		/// vector of discretization orders, used when not all elements have the same degree, one per element
		Eigen::VectorXi disc_orders, disc_ordersq;

		/// nodes on the boundary of polygonal elements, used for harmonic bases
		std::map<int, basis::InterfaceData> poly_edge_to_data;

		/// Mapping from input nodes to FE nodes
		std::shared_ptr<polyfem::mesh::MeshNodes> mesh_nodes;

		/// used to store assembly values for small problems
		assembler::AssemblyValsCache ass_vals_cache;
		assembler::AssemblyValsCache mass_ass_vals_cache;

		/// Mass matrix, it is computed only for time dependent problems
		StiffnessMatrix mass;
		/// average system mass, used for contact with IPC
		double avg_mass = 0;
		Eigen::MatrixXd rhs;

		solver::SolveData solve_data;

		mesh::Obstacle obstacle;
		/// @brief IPC collision mesh
		ipc::CollisionMesh collision_mesh;

		std::shared_ptr<assembler::PressureAssembler> elasticity_pressure_assembler = nullptr;

		std::shared_ptr<assembler::ViscousDamping> damping_assembler = nullptr;
		std::shared_ptr<assembler::ViscousDampingPrev> damping_prev_assembler = nullptr;
	};
} // namespace polyfem::varform
