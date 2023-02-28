#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include "LinearElasticity.hpp"
// #include "HookeLinearElasticity.hpp"
#include "SaintVenantElasticity.hpp"
#include "NeoHookeanElasticity.hpp"
#include "GenericElastic.hpp"
#include "MooneyRivlinElasticity.hpp"
#include "OgdenElasticity.hpp"

#include <Eigen/Dense>
#include <array>

namespace polyfem::assembler
{
	class MultiModel
	{
	public:
		// neccessary for mixing linear model with non-linear collision response
		Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const;
		// compute gradient of elastic energy, as assembler
		Eigen::VectorXd assemble_grad(const NonLinearAssemblerData &data) const;
		// compute elastic energy
		double compute_energy(const NonLinearAssemblerData &data) const;

		// uses autodiff to compute the rhs for a fabbricated solution
		// uses autogenerated code to compute div(sigma)
		// pt is the evaluation of the solution at a point
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		// compute von mises stress for an element at the local points
		void compute_von_mises_stresses(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		// compute stress tensor for an element at the local points
		void compute_stress_tensor(const int el_id, const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const ElasticityTensorType &type, Eigen::MatrixXd &tensor) const;

		// size of the problem, this is a tensor problem so the size is the size of the mesh
		inline int size() const { return size_; }
		void set_size(const int size);

		// inialize material parameter
		void add_multimaterial(const int index, const json &params);
		// initialized multi models
		inline void init_multimodels(const std::vector<std::string> &mats) { multi_material_models_ = mats; }

		// class that stores and compute lame parameters per point
		const LameParameters &lame_params() const { return linear_elasticity_.lame_params(); }
		void set_params(const LameParameters &params)
		{
			neo_hookean_.set_params(params);
			linear_elasticity_.set_params(params);
			// TODO set params
		}

	private:
		int size_ = -1;
		std::vector<std::string> multi_material_models_;

		SaintVenantElasticity saint_venant_;
		NeoHookeanElasticity neo_hookean_;
		LinearElasticity linear_elasticity_;
		// HookeLinearElasticity hooke_;
		MooneyRivlinElasticity mooney_rivlin_elasticity_;
		UnconstrainedOgdenElasticity ogden_elasticity_;
	};
} // namespace polyfem::assembler
