#pragma once

#include <MeshFEMSparse/SystemAssembler.hh>

#include <stdexcept>
#include <utility>
#include <variant>

namespace polyfem::assembler
{
	class FastSystemAssembler
	{
	public:
		FastSystemAssembler(const int dim, const int n_basis)
			: assembler_(make_assembler(dim, n_basis)),
			  dim_(dim)
		{
		}

		template <class StencilCallable>
		std::unique_ptr<MeshFEM::BlockCSCHessianBase> sparsityPattern(const size_t num_stencils, StencilCallable &&stencil) const
		{
			return std::visit(
				[&](const auto &assembler) -> std::unique_ptr<MeshFEM::BlockCSCHessianBase> {
					return assembler.blockSparsityPattern(num_stencils, std::forward<StencilCallable>(stencil));
				},
				assembler_);
		}

		template <class EvalCallable, class StencilCallable>
		void assembleHessian(
			MeshFEM::BlockCSCHessianBase &H,
			const size_t num_stencils,
			EvalCallable &&eval,
			StencilCallable &&stencil) const
		{
			return std::visit(
				[&](const auto &assembler) {
					assembler.assembleHessianDynamicPEH(
						H, num_stencils,
						std::forward<EvalCallable>(eval),
						std::forward<StencilCallable>(stencil));
				},
				assembler_);
		}

		int getDim() const { return dim_; }

	private:
		using AssemblerVariant = std::variant<MeshFEM::SystemAssembler<1>, MeshFEM::SystemAssembler<2>, MeshFEM::SystemAssembler<3>>;

		static AssemblerVariant make_assembler(const int dim, const int n_basis)
		{
			if (dim == 1)
				return AssemblerVariant(std::in_place_type<MeshFEM::SystemAssembler<1>>, n_basis);
			if (dim == 2)
				return AssemblerVariant(std::in_place_type<MeshFEM::SystemAssembler<2>>, n_basis);
			if (dim == 3)
				return AssemblerVariant(std::in_place_type<MeshFEM::SystemAssembler<3>>, n_basis);
			throw std::runtime_error("Unsupported dimension for SystemAssembler: " + std::to_string(dim));
		}

		AssemblerVariant assembler_;
		int dim_;
	};
} // namespace polyfem::assembler
