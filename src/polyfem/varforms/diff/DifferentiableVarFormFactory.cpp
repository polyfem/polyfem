#include <polyfem/varforms/diff/DifferentiableVarFormFactory.hpp>

#include <polyfem/varforms/VarFormFactory.hpp>
#include <polyfem/varforms/diff/DifferentiableLinearElasticVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableNonlinearElasticVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableScalarVarForm.hpp>

namespace polyfem::varform
{
	bool DifferentiableVarFormFactory::supports(const std::string &formulation, const json &args)
	{
		const auto regular = VarFormFactory::create(formulation, args);
		return dynamic_cast<const ScalarVarForm *>(regular.get()) != nullptr
			   || dynamic_cast<const LinearElasticVarForm *>(regular.get()) != nullptr
			   || dynamic_cast<const NonlinearElasticStaticVarForm *>(regular.get()) != nullptr
			   || dynamic_cast<const NonlinearElasticTransientVarForm *>(regular.get()) != nullptr;
	}

	std::shared_ptr<VarForm> DifferentiableVarFormFactory::create(const std::string &formulation, const json &args)
	{
		const auto regular = VarFormFactory::create(formulation, args);
		if (dynamic_cast<const ScalarVarForm *>(regular.get()))
			return std::make_shared<DifferentiableScalarVarForm>();
		if (dynamic_cast<const LinearElasticVarForm *>(regular.get()))
			return std::make_shared<DifferentiableLinearElasticVarForm>();
		if (dynamic_cast<const NonlinearElasticStaticVarForm *>(regular.get()))
			return std::make_shared<DifferentiableNonlinearElasticStaticVarForm>();
		if (dynamic_cast<const NonlinearElasticTransientVarForm *>(regular.get()))
			return std::make_shared<DifferentiableNonlinearElasticTransientVarForm>();
		return nullptr;
	}
} // namespace polyfem::varform
