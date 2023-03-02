#include "ProblemFactory.hpp"

#include "ProblemWithSolution.hpp"
#include "MiscProblem.hpp"
#include "FrankeProblem.hpp"
#include "ElasticProblem.hpp"
#include "PointBasedProblem.hpp"
#include "KernelProblem.hpp"
#include "StokesProblem.hpp"
#include "TestProblem.hpp"
#include "NodeProblem.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem
{
	using namespace assembler;

	namespace problem
	{

		const ProblemFactory &ProblemFactory::factory()
		{
			static ProblemFactory instance;

			return instance;
		}

		ProblemFactory::ProblemFactory()
		{
			problems_.emplace("Linear", []() { return std::make_shared<LinearProblem>("Linear"); });
			problems_.emplace("Quadratic", []() { return std::make_shared<QuadraticProblem>("Quadratic"); });
			problems_.emplace("Cubic", []() { return std::make_shared<CubicProblem>("Cubic"); });
			problems_.emplace("Sine", []() { return std::make_shared<SineProblem>("Sine"); });
			problems_.emplace("Franke", []() { return std::make_shared<FrankeProblem>("Franke"); });
			problems_.emplace("FrankeOld", []() { return std::make_shared<FrankeProblemOld>("FrankeOld"); });
			problems_.emplace("GenericScalarExact", []() { return std::make_shared<GenericScalarProblemExact>("GenericScalarExact"); });
			problems_.emplace("Zero_BC", []() { return std::make_shared<ZeroBCProblem>("Zero_BC"); });

			problems_.emplace("Elastic", []() { return std::make_shared<ElasticProblem>("Elastic"); });
			problems_.emplace("Walk", []() { return std::make_shared<WalkProblem>("Walk"); });
			problems_.emplace("TorsionElastic", []() { return std::make_shared<TorsionElasticProblem>("TorsionElastic"); });
			problems_.emplace("DoubleTorsionElastic", []() { return std::make_shared<DoubleTorsionElasticProblem>("DoubleTorsionElastic"); });
			problems_.emplace("ElasticZeroBC", []() { return std::make_shared<ElasticProblemZeroBC>("ElasticZeroBC"); });
			problems_.emplace("ElasticExact", []() { return std::make_shared<ElasticProblemExact>("ElasticExact"); });
			problems_.emplace("ElasticCantileverExact", []() { return std::make_shared<ElasticCantileverExact>("ElasticCantileverExact"); });
			problems_.emplace("CompressionElasticExact", []() { return std::make_shared<CompressionElasticProblemExact>("CompressionElasticExact"); });
			problems_.emplace("QuadraticElasticExact", []() { return std::make_shared<QuadraticElasticProblemExact>("QuadraticElasticExact"); });
			problems_.emplace("LinearElasticExact", []() { return std::make_shared<LinearElasticProblemExact>("LinearElasticExact"); });
			problems_.emplace("PointBasedTensor", []() { return std::make_shared<PointBasedTensorProblem>("PointBasedTensor"); });
			// problems_.emplace("Kernel", []() { return std::make_shared<KernelProblem>("Kernel"); });
			problems_.emplace("Node", []() { return std::make_shared<NodeProblem>("Node"); });

			problems_.emplace("TimeDependentScalar", []() { return std::make_shared<TimeDependentProblem>("TimeDependentScalar"); });
			problems_.emplace("MinSurf", []() { return std::make_shared<MinSurfProblem>("MinSurf"); });
			problems_.emplace("Gravity", []() { return std::make_shared<GravityProblem>("Gravity"); });

			problems_.emplace("ConstantVelocity", []() { return std::make_shared<ConstantVelocity>("ConstantVelocity"); });
			problems_.emplace("TwoSpheres", []() { return std::make_shared<TwoSpheres>("TwoSpheres"); });
			problems_.emplace("DrivenCavity", []() { return std::make_shared<DrivenCavity>("DrivenCavity"); });
			problems_.emplace("DrivenCavityC0", []() { return std::make_shared<DrivenCavityC0>("DrivenCavityC0"); });
			problems_.emplace("DrivenCavitySmooth", []() { return std::make_shared<DrivenCavitySmooth>("DrivenCavitySmooth"); });
			problems_.emplace("Flow", []() { return std::make_shared<Flow>("Flow"); });
			problems_.emplace("FlowWithObstacle", []() { return std::make_shared<FlowWithObstacle>("FlowWithObstacle"); });
			problems_.emplace("CornerFlow", []() { return std::make_shared<CornerFlow>("CornerFlow"); });
			problems_.emplace("UnitFlowWithObstacle", []() { return std::make_shared<UnitFlowWithObstacle>("UnitFlowWithObstacle"); });
			problems_.emplace("StokesLaw", []() { return std::make_shared<StokesLawProblem>("StokesLaw"); });
			problems_.emplace("TaylorGreenVortex", []() { return std::make_shared<TaylorGreenVortexProblem>("TaylorGreenVortex"); });
			problems_.emplace("SimpleStokeProblemExact", []() { return std::make_shared<SimpleStokeProblemExact>("SimpleStokeProblemExact"); });
			problems_.emplace("SineStokeProblemExact", []() { return std::make_shared<SineStokeProblemExact>("SineStokeProblemExact"); });
			problems_.emplace("TransientStokeProblemExact", []() { return std::make_shared<TransientStokeProblemExact>("TransientStokeProblemExact"); });
			problems_.emplace("Kovnaszy", []() { return std::make_shared<Kovnaszy>("Kovnaszy"); });
			problems_.emplace("Airfoil", []() { return std::make_shared<Airfoil>("Airfoil"); });
			problems_.emplace("Lshape", []() { return std::make_shared<Lshape>("Lshape"); });

			problems_.emplace("TestProblem", []() { return std::make_shared<TestProblem>("TestProblem"); });

			problems_.emplace("BilaplacianProblemWithSolution", []() { return std::make_shared<BilaplacianProblemWithSolution>("BilaplacianProblemWithSolution"); });

			for (auto it = problems_.begin(); it != problems_.end(); ++it)
				problem_names_.push_back(it->first);
		}

		std::shared_ptr<Problem> ProblemFactory::get_problem(const std::string &problem) const
		{
			auto it = problems_.find(problem);

			if (it == problems_.end())
			{
				logger().error("Problem {} does not exist", problem);
				return problems_.at("Linear")();
			}

			return it->second();
		}
	} // namespace problem
} // namespace polyfem