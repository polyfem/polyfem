#include "Problem.hpp"

#include "MiscProblem.hpp"
#include "FrankeProblem.hpp"
#include "ElasticProblem.hpp"
#include "CustomProblem.hpp"
#include "KernelProblem.hpp"
#include "TestProblem.hpp"

#include <memory>
#include <iostream>

namespace poly_fem
{
	Problem::Problem(const std::string &name)
	: name_(name)
	{ }

	void Problem::setup_bc(const Mesh &mesh, const std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &boundary_nodes, std::vector< LocalBoundary > &local_neumann_boundary)
	{
		std::vector< LocalBoundary > new_local_boundary;
		local_neumann_boundary.clear();
		for(auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			LocalBoundary new_lb(lb.element_id(), lb.type());
			LocalBoundary new_neumann_lb(lb.element_id(), lb.type());
			for(int i = 0; i < lb.size(); ++i)
			{
				const int primitive_g_id = lb.global_primitive_id(i);
				const int tag = mesh.get_boundary_id(primitive_g_id);

				if(boundary_ids_.empty() || std::find(boundary_ids_.begin(), boundary_ids_.end(), tag) != boundary_ids_.end())
					new_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);

				if(std::find(neumann_boundary_ids_.begin(), neumann_boundary_ids_.end(), tag) != neumann_boundary_ids_.end())
					new_neumann_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
			}

			if(!new_lb.empty())
				new_local_boundary.emplace_back(new_lb);

			if(!new_neumann_lb.empty())
				local_neumann_boundary.emplace_back(new_neumann_lb);
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


	const ProblemFactory &ProblemFactory::factory()
	{
		static ProblemFactory instance;

		return instance;
	}

	ProblemFactory::ProblemFactory()
	{
		problems_.emplace("Linear", std::make_shared<LinearProblem>("Linear"));
		problems_.emplace("Quadratic", std::make_shared<QuadraticProblem>("Quadratic"));
		problems_.emplace("Cubic", std::make_shared<CubicProblem>("Cubic"));
		problems_.emplace("Franke", std::make_shared<FrankeProblem>("Franke"));
		problems_.emplace("Zero_BC", std::make_shared<ZeroBCProblem>("Zero_BC"));
		problems_.emplace("Elastic", std::make_shared<ElasticProblem>("Elastic"));
		problems_.emplace("ElasticForce", std::make_shared<ElasticForceProblem>("ElasticForce"));
		problems_.emplace("ElasticZeroBC", std::make_shared<ElasticProblemZeroBC>("ElasticZeroBC"));
		problems_.emplace("ElasticExact", std::make_shared<ElasticProblemExact>("ElasticExact"));
		problems_.emplace("CompressionElasticExact", std::make_shared<CompressionElasticProblemExact>("CompressionElasticExact"));
		problems_.emplace("QuadraticElasticExact", std::make_shared<QuadraticElasticProblemExact>("QuadraticElasticExact"));
		problems_.emplace("LinearElasticExact", std::make_shared<LinearElasticProblemExact>("LinearElasticExact"));
		problems_.emplace("Custom", std::make_shared<CustomProblem>("Custom"));
		problems_.emplace("Kernel", std::make_shared<KernelProblem>("Kernel"));

		problems_.emplace("TestProblem", std::make_shared<TestProblem>("TestProblem"));

		for(auto it = problems_.begin(); it != problems_.end(); ++it)
			problem_names_.push_back(it->first);
	}

	std::shared_ptr<Problem> ProblemFactory::get_problem(const std::string &problem) const
	{
		auto it = problems_.find(problem);

		if(it == problems_.end())
			return problems_.at("Linear");

		return it->second;
	}


}
