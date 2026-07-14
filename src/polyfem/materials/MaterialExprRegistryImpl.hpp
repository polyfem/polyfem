#pragma once

#include <polyfem/utils/Span.hpp>

#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#include <cuda/buffer>
#include <cuda/algorithm>
#include <cuda/std/type_traits>
#endif

namespace polyfem::material
{

	template <typename T>
	struct MaterialStoreView
	{
		/// Array of Material expressions or expression view if on device. Might be empty.
		Span<const T> expr;
		/// Per element material expression id. Empty if expr is empty.
		/// id == -1 is a special value indicating no material expr.
		Span<const int> expr_ids;
	};

	template <typename M>
	class MaterialStore
	{
	private:
		/// Array of Material expressions. Might be empty.
		std::vector<M> expr_;
		/// Per element material expression id. Empty if expr is empty.
		/// id == -1 is a special value indicating no material expr.
		std::vector<int> expr_ids_;

#ifdef POLYFEM_WITH_CUDA
		mutable bool need_host_device_sync_ = true;
		mutable DeviceBuf<typename M::ExprViewType> d_expr_;
		mutable DeviceBuf<int> d_expr_ids_;
#endif

	public:
		/// @brief Set/Replace material expr T of element.
		/// @note Invalidates device material storage, which is expensive.
		void set(int element_num, Span<const int> target_elements, M material)
		{
			assert(element_num >= 0);

#ifdef POLYFEM_WITH_CUDA
			need_host_device_sync_ = true;
#endif

			expr_.push_back(std::move(material));
			int material_id = expr_.size() - 1;

			if (expr_ids_.empty())
			{
				expr_ids_.resize(element_num, -1);
			}

			for (int t : target_elements)
			{
				assert(t >= 0 && t < element_num);
				expr_ids_[t] = material_id;
			}
		};

		MaterialStoreView<M> view() const
		{
			return MaterialStoreView<M>{expr_, expr_ids_};
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		/// @note Invalidates device material storage, which is expensive.
		M *get_mutable(int element)
		{
			if (expr_ids_.empty() || expr_ids_[element] == -1)
				return nullptr;

#ifdef POLYFEM_WITH_CUDA
			need_host_device_sync_ = true;
#endif

			return expr_.data() + expr_ids_[element];
		}

		const M *get(int element) const
		{
			if (expr_ids_.empty() || expr_ids_[element] == -1)
				return nullptr;
			return expr_.data() + expr_ids_[element];
		}

#ifdef POLYFEM_WITH_CUDA

		bool is_device_compatible() const
		{
			for (const auto &expr : expr_)
			{
				if (!polyfem::material::is_device_compatible(expr))
					return false;
			}
			return true;
		}

		/// Return view on device memory. Lazily sync data.
		///
		/// If expression is not device compatible, return view to empty expression which evals to Nan.
		MaterialStoreView<typename M::ExprViewType> device_view(ExecutionPolicy policy) const
		{
			auto &p = policy;

			if (need_host_device_sync_)
			{
				std::vector<typename M::ExprViewType> tmp(expr_.size());
				for (int i = 0; i < expr_.size(); ++i)
				{
					tmp[i] = polyfem::material::make_device_expr(expr_[i], policy);
				}

				d_expr_ = copy_to_device_async<typename M::ExprViewType>(tmp, policy);
				d_expr_ids_ = copy_to_device_async<int>(expr_ids_, policy);
				p.stream->sync();
				need_host_device_sync_ = false;
			}

			return MaterialStoreView<typename M::ExprViewType>{*d_expr_, *d_expr_ids_};
		}

		/// Release device storage.
		void clear_device_storage()
		{
			need_host_device_sync_ = true;
			d_expr_ = {};
			d_expr_ids_ = {};
		}
#endif
	};

	template <typename... M>
	class MaterialExprRegistryImpl
	{
	private:
		int element_num_;
		std::tuple<MaterialStore<M>...> materials_;

	public:
		MaterialExprRegistryImpl(int element_num) : element_num_(element_num) {};

		// -----------------------------------------------------
		// element APIs
		// -----------------------------------------------------

		int element_num() const { return element_num_; };

		// -----------------------------------------------------
		// material APIs
		// -----------------------------------------------------

		/// @brief Return true if element has material expr T.
		template <typename T>
		bool has_material(int element) const
		{
			assert(element >= 0 && element < element_num_);
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			auto v = s.view();
			return !(v.expr_ids.empty() || v.expr_ids[element] == -1);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		/// @note Invalidates device material storage, which is expensive.
		template <typename T>
		T *get_mutable(int element)
		{
			assert(element >= 0 && element < element_num_);
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			return s.get_mutable(element);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		const T *get(int element) const
		{
			assert(element >= 0 && element < element_num_);
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			return s.get(element);
		}

#ifdef POLYFEM_WITH_CUDA
		template <typename T>
		bool is_device_compatible() const
		{
			static_assert((std::is_same_v<T, M> || ...), "T is not a material expression type");

			auto &s = std::get<MaterialStore<T>>(materials_);
			return s.is_device_compatible();
		}

		/// @brief Return all materials expr view of type T.
		template <typename T>
		MaterialStoreView<typename T::ExprViewType> get_all_device_expr_views(ExecutionPolicy p) const
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			return s.device_view(p);
		}
#endif

		/// @brief Set/Replace material expr T of element.
		/// @note Invalidates device material storage, which is expensive.
		template <typename T>
		void set(Span<const int> elements, T material)
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			s.set(element_num_, elements, std::move(material));
		}
	};

} // namespace polyfem::material
