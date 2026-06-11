#include "polyfem/varforms/FESpace.hpp"

#include <Eigen/Dense>

namespace polyfem::varform
{
	void GeometryMapping::mirror_fe_space(const FESpace &space)
	{
		n_bases = space.n_bases;
		bases = space.bases;
		disc_orders = space.disc_orders;
		polys = space.polys;
		polys_3d = space.polys_3d;
		mesh_nodes = space.mesh_nodes;
	}

	int SolutionLayout::add_block(
		const int dof,
		const bool is_time_integrated,
		const bool is_algebraic)
	{
		SolutionBlock block;
		block.offset = total_dof();
		block.dof = dof;
		block.is_time_integrated = is_time_integrated;
		block.is_algebraic = is_algebraic;
		blocks_.push_back(block);
		return int(blocks_.size()) - 1;
	}

	SolutionBlock SolutionLayout::get_block(const int id) const
	{
		return blocks_.at(id);
	}

	int SolutionLayout::total_dof() const
	{
		if (blocks_.empty())
			return 0;

		const SolutionBlock &last = blocks_.back();
		return last.offset + last.dof;
	}

} // namespace polyfem::varform
