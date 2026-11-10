#pragma once

namespace polyfem::solver
{
	enum class ParameterType
	{
		Shape,
		LameParameter,
		FrictionCoefficient,
		DampingCoefficient,
		InitialCondition,
		DirichletBC,
		PressureBC,
		MacroStrain,
		PeriodicShape
	};
}
