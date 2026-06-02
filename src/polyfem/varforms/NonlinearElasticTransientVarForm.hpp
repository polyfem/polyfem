#pragma once

#include <polyfem/varforms/ElasticVarForm.hpp>

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/assembler/PressureAssembler.hpp>

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>

namespace polyfem::varform
{
	class NonlinearElasticVarForm : public ElasticVarForm
	{
	protected:
		void init_solve(Eigen::MatrixXd &sol, const double t);
		void init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t);
		void solve_tensor_nonlinear(int step, Eigen::MatrixXd &sol, const bool init_lagging = true);

		std::shared_ptr<assembler::PressureAssembler> build_pressure_assembler() const;
		std::vector<int> primitive_to_node() const;
		std::vector<int> node_to_primitive() const;
	};

	class NonlinearElasticTransientVarForm : public NonlinearElasticVarForm
	{
	public:
		void solve(Eigen::MatrixXd &sol) override;

		std::string name() const override { return "NonlinearElasticTransient"; }
	};

	class NonlinearElasticStaticVarForm : public NonlinearElasticVarForm
	{
	public:
		void solve(Eigen::MatrixXd &sol) override;

		std::string name() const override { return "NonlinearElasticStatic"; }
	};
} // namespace polyfem::varform
