#include <polyfem/Problem.hpp>

#include <polyfem/ProblemWithSolution.hpp>
#include <polyfem/MiscProblem.hpp>
#include <polyfem/FrankeProblem.hpp>
#include <polyfem/ElasticProblem.hpp>
#include <polyfem/PointBasedProblem.hpp>
#include <polyfem/GenericProblem.hpp>
#include <polyfem/KernelProblem.hpp>
#include <polyfem/StokesProblem.hpp>
#include <polyfem/TestProblem.hpp>
#include <polyfem/NodeProblem.hpp>

#include <memory>
#include <iostream>

namespace polyfem
{
	Problem::Problem(const std::string &name)
		: name_(name)
	{
	}

	void Problem::setup_bc(const Mesh &mesh, const std::vector<ElementBases> &bases, const std::vector<ElementBases> &pressure_bases, std::vector<LocalBoundary> &local_boundary, std::vector<int> &boundary_nodes, std::vector<LocalBoundary> &local_neumann_boundary, std::vector<int> &pressure_boundary_nodes)
	{
		std::vector<LocalBoundary> new_local_boundary;
		std::vector<LocalBoundary> new_local_pressure_dirichlet_boundary;
		local_neumann_boundary.clear();
		for (auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			LocalBoundary new_lb(lb.element_id(), lb.type());
			LocalBoundary new_neumann_lb(lb.element_id(), lb.type());
			LocalBoundary new_pressure_dirichlet_lb(lb.element_id(), lb.type());
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_g_id = lb.global_primitive_id(i);
				const int tag = mesh.get_boundary_id(primitive_g_id);

				if (tag <= 0)
					continue;

				if (boundary_ids_.empty() || std::find(boundary_ids_.begin(), boundary_ids_.end(), tag) != boundary_ids_.end())
					new_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
				if (std::find(neumann_boundary_ids_.begin(), neumann_boundary_ids_.end(), tag) != neumann_boundary_ids_.end())
					new_neumann_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
				if (std::find(pressure_boundary_ids_.begin(), pressure_boundary_ids_.end(), tag) != pressure_boundary_ids_.end())
					new_neumann_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
				if (std::find(splitting_pressure_boundary_ids_.begin(), splitting_pressure_boundary_ids_.end(), tag) != splitting_pressure_boundary_ids_.end())
					new_pressure_dirichlet_lb.add_boundary_primitive(lb.global_primitive_id(i), lb[i]);
			}

			if (!new_lb.empty())
				new_local_boundary.emplace_back(new_lb);
			if (!new_neumann_lb.empty())
				local_neumann_boundary.emplace_back(new_neumann_lb);
			if (!new_pressure_dirichlet_lb.empty())
				new_local_pressure_dirichlet_boundary.emplace_back(new_pressure_dirichlet_lb);
		}
		local_boundary.clear();
		std::swap(local_boundary, new_local_boundary);

		boundary_nodes.clear();
		pressure_boundary_nodes.clear();

		const int dim = is_scalar() ? 1 : mesh.dimension();

		for (auto it = local_boundary.begin(); it != local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = bases[lb.element_id()];
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = b.local_nodes_for_primitive(primitive_global_id, mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					auto &bs = b.bases[nodes(n)];
					for (size_t g = 0; g < bs.global().size(); ++g)
					{
						const int base_index = bs.global()[g].index * dim;
						for (int d = 0; d < dim; ++d)
						{
							if (is_dimension_dirichet(mesh.get_boundary_id(primitive_global_id), d))
								boundary_nodes.push_back(base_index + d);
						}
					}
				}
			}
		}

		for (auto it = new_local_pressure_dirichlet_boundary.begin(); it != new_local_pressure_dirichlet_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &b = pressure_bases[lb.element_id()];
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = b.local_nodes_for_primitive(primitive_global_id, mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					auto &bs = b.bases[nodes(n)];
					for (size_t g = 0; g < bs.global().size(); ++g)
					{
						const int base_index = bs.global()[g].index;
						pressure_boundary_nodes.push_back(base_index);
					}
				}
			}
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.resize(std::distance(boundary_nodes.begin(), it));

		std::sort(pressure_boundary_nodes.begin(), pressure_boundary_nodes.end());
		auto it_ = std::unique(pressure_boundary_nodes.begin(), pressure_boundary_nodes.end());
		pressure_boundary_nodes.resize(std::distance(pressure_boundary_nodes.begin(), it_));
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
		problems_.emplace("Sine", std::make_shared<SineProblem>("Sine"));
		problems_.emplace("Franke", std::make_shared<FrankeProblem>("Franke"));
		problems_.emplace("FrankeOld", std::make_shared<FrankeProblemOld>("FrankeOld"));
		problems_.emplace("GenericScalarExact", std::make_shared<GenericScalarProblemExact>("GenericScalarExact"));
		problems_.emplace("Zero_BC", std::make_shared<ZeroBCProblem>("Zero_BC"));

		problems_.emplace("Elastic", std::make_shared<ElasticProblem>("Elastic"));
		problems_.emplace("Walk", std::make_shared<WalkProblem>("Walk"));
		problems_.emplace("TorsionElastic", std::make_shared<TorsionElasticProblem>("TorsionElastic"));
		problems_.emplace("DoubleTorsionElastic", std::make_shared<DoubleTorsionElasticProblem>("DoubleTorsionElastic"));
		problems_.emplace("GenericScalar", std::make_shared<GenericScalarProblem>("GenericScalar"));
		problems_.emplace("GenericTensor", std::make_shared<GenericTensorProblem>("GenericTensor"));
		problems_.emplace("ElasticZeroBC", std::make_shared<ElasticProblemZeroBC>("ElasticZeroBC"));
		problems_.emplace("ElasticExact", std::make_shared<ElasticProblemExact>("ElasticExact"));
		problems_.emplace("CompressionElasticExact", std::make_shared<CompressionElasticProblemExact>("CompressionElasticExact"));
		problems_.emplace("QuadraticElasticExact", std::make_shared<QuadraticElasticProblemExact>("QuadraticElasticExact"));
		problems_.emplace("LinearElasticExact", std::make_shared<LinearElasticProblemExact>("LinearElasticExact"));
		problems_.emplace("PointBasedTensor", std::make_shared<PointBasedTensorProblem>("PointBasedTensor"));
		problems_.emplace("Kernel", std::make_shared<KernelProblem>("Kernel"));
		problems_.emplace("Node", std::make_shared<NodeProblem>("Node"));

		problems_.emplace("TimeDependentScalar", std::make_shared<TimeDependentProblem>("TimeDependentScalar"));
		problems_.emplace("MinSurf", std::make_shared<MinSurfProblem>("MinSurf"));
		problems_.emplace("Gravity", std::make_shared<GravityProblem>("Gravity"));

		problems_.emplace("ConstantVelocity", std::make_shared<ConstantVelocity>("ConstantVelocity"));
		problems_.emplace("TwoSpheres", std::make_shared<TwoSpheres>("TwoSpheres"));
		problems_.emplace("DrivenCavity", std::make_shared<DrivenCavity>("DrivenCavity"));
		problems_.emplace("DrivenCavityC0", std::make_shared<DrivenCavityC0>("DrivenCavityC0"));
		problems_.emplace("DrivenCavitySmooth", std::make_shared<DrivenCavitySmooth>("DrivenCavitySmooth"));
		problems_.emplace("Flow", std::make_shared<Flow>("Flow"));
		problems_.emplace("FlowWithObstacle", std::make_shared<FlowWithObstacle>("FlowWithObstacle"));
		problems_.emplace("CornerFlow", std::make_shared<CornerFlow>("CornerFlow"));
		problems_.emplace("UnitFlowWithObstacle", std::make_shared<UnitFlowWithObstacle>("UnitFlowWithObstacle"));
		problems_.emplace("StokesLaw", std::make_shared<StokesLawProblem>("StokesLaw"));
		problems_.emplace("TaylorGreenVortex", std::make_shared<TaylorGreenVortexProblem>("TaylorGreenVortex"));
		problems_.emplace("SimpleStokeProblemExact", std::make_shared<SimpleStokeProblemExact>("SimpleStokeProblemExact"));
		problems_.emplace("SineStokeProblemExact", std::make_shared<SineStokeProblemExact>("SineStokeProblemExact"));
		problems_.emplace("TransientStokeProblemExact", std::make_shared<TransientStokeProblemExact>("TransientStokeProblemExact"));
		problems_.emplace("Kovnaszy", std::make_shared<Kovnaszy>("Kovnaszy"));
		problems_.emplace("Airfoil", std::make_shared<Airfoil>("Airfoil"));
		problems_.emplace("Lshape", std::make_shared<Lshape>("Lshape"));

		problems_.emplace("TestProblem", std::make_shared<TestProblem>("TestProblem"));

		problems_.emplace("BilaplacianProblemWithSolution", std::make_shared<BilaplacianProblemWithSolution>("BilaplacianProblemWithSolution"));

		for (auto it = problems_.begin(); it != problems_.end(); ++it)
			problem_names_.push_back(it->first);
	}

	std::shared_ptr<Problem> ProblemFactory::get_problem(const std::string &problem) const
	{
		auto it = problems_.find(problem);

		if (it == problems_.end())
			return problems_.at("Linear");

		return it->second;
	}

} // namespace polyfem
