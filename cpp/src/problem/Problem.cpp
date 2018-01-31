#include "Problem.hpp"

#include "LinearProblem.hpp"
#include "QuadraticProblem.hpp"
#include "Franke2dProblem.hpp"
#include "Franke3dProblem.hpp"
#include "ZeroBCProblem.hpp"
#include "ElasticProblem.hpp"

#include <iostream>

namespace poly_fem
{
	std::shared_ptr<Problem> Problem::get_problem(const ProblemType type)
	{
		switch(type)
		{
			case ProblemType::Linear: return std::make_shared<LinearProblem>();
			case ProblemType::Quadratic: return std::make_shared<QuadraticProblem>();
			case ProblemType::Franke: return std::make_shared<Franke2dProblem>();
			case ProblemType::Franke3d: return std::make_shared<Franke3dProblem>();
			case ProblemType::Zero_BC: return std::make_shared<ZeroBCProblem>();
			case ProblemType::Elastic: return std::make_shared<ElasticProblem>();

			default:
			assert(false);
			return std::make_shared<LinearProblem>();
		}
	}

	void Problem::remove_neumann_nodes(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes)
	{
		std::vector< LocalBoundary > new_local_boundary;
		for(auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			LocalBoundary new_lb(lb.element_id(), lb.type());
			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_g_id = lb.global_primitive_id(i);
				const int tag = mesh.get_boundary_id(primitive_g_id);

				if(boundary_ids_.empty() || std::find(boundary_ids_.begin(), boundary_ids_.end(), tag) != boundary_ids_.end())
					new_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
			}

			if(!new_lb.empty())
				new_local_boundary.emplace_back(new_lb);
		}
		std::swap(local_boundary, new_local_boundary);

		boundary_nodes.clear();

		for(auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = bases[lb.element_id()];
			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = b.local_nodes_for_primitive(primitive_global_id, mesh);

				for(long n = 0; n < nodes.size(); ++n){
					auto &bs = b.bases[nodes(n)];
					for(size_t g = 0; g < bs.global().size(); ++g)
						boundary_nodes.push_back(bs.global()[g].index);
				}
			}
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.resize(std::distance(boundary_nodes.begin(), it));
	}
}
