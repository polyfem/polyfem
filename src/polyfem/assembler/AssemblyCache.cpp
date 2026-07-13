#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/utils/Range.hpp>

#include <cassert>
#include <Eigen/Core>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CudaUtils.hpp>
#endif

namespace polyfem::assembler
{

	void AssemblyTempStorage::resize(int dim, int basis_num, int geom_basis_num, int quad_num)
	{

		int n = basis_num * quad_num;
		int gn = geom_basis_num * quad_num;

		basis_values.resize(n);
		basis_grad_x.resize(n);
		basis_grad_phy_x.resize(n);
		gbasis_values.resize(gn);
		gbasis_grad_x.resize(gn);
		physical_x.resize(quad_num);
		det_J.resize(quad_num);
		J_inverse_transpose.resize(quad_num * dim * dim);
		weighted_measure.resize(quad_num);

		if (dim > 1)
		{
			basis_grad_y.resize(n);
			basis_grad_phy_y.resize(n);
			gbasis_grad_y.resize(gn);
			physical_y.resize(quad_num);
		}

		if (dim > 2)
		{
			basis_grad_z.resize(n);
			basis_grad_phy_z.resize(n);
			gbasis_grad_z.resize(gn);
			physical_z.resize(quad_num);
		}
	}

	AssemblyCacheDesc AssemblyCache::copy_from_temp(bool is_mass, const AssemblyTempStorage &temp)
	{
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		auto append_and_set_range = [](Span<const double> src, std::vector<double> &dst, Range &range) {
			range.offset = dst.size();
			range.num = src.size();
			dst.insert(dst.end(), src.begin(), src.end());
		};

		AssemblyCacheDesc new_desc;
		new_desc.is_empty = false;
		new_desc.is_mass = is_mass;

		append_and_set_range(temp.basis_values, basis_values_, new_desc.basis_val_range);
		append_and_set_range(temp.basis_grad_x, basis_grad_x_, new_desc.basis_grad_x_range);
		append_and_set_range(temp.basis_grad_y, basis_grad_y_, new_desc.basis_grad_y_range);
		append_and_set_range(temp.basis_grad_z, basis_grad_z_, new_desc.basis_grad_z_range);
		append_and_set_range(temp.basis_grad_phy_x, basis_grad_phy_x_, new_desc.basis_grad_phy_x_range);
		append_and_set_range(temp.basis_grad_phy_y, basis_grad_phy_y_, new_desc.basis_grad_phy_y_range);
		append_and_set_range(temp.basis_grad_phy_z, basis_grad_phy_z_, new_desc.basis_grad_phy_z_range);
		append_and_set_range(temp.physical_x, physical_x_, new_desc.physical_x_range);
		append_and_set_range(temp.physical_y, physical_y_, new_desc.physical_y_range);
		append_and_set_range(temp.physical_z, physical_z_, new_desc.physical_z_range);
		append_and_set_range(temp.det_J, det_J_, new_desc.det_J_range);
		append_and_set_range(temp.J_inverse_transpose, J_inverse_transpose_, new_desc.J_inverse_transpose_range);
		append_and_set_range(temp.weighted_measure, weighted_measure_, new_desc.weighted_measure_range);

		return new_desc;
	}

	void AssemblyCache::clear()
	{
		desc_.clear();
		basis_values_.clear();
		basis_grad_x_.clear();
		basis_grad_y_.clear();
		basis_grad_z_.clear();
		basis_grad_phy_x_.clear();
		basis_grad_phy_y_.clear();
		basis_grad_phy_z_.clear();
		physical_x_.clear();
		physical_y_.clear();
		physical_z_.clear();
		det_J_.clear();
		J_inverse_transpose_.clear();
		weighted_measure_.clear();

#ifdef POLYFEM_WITH_CUDA
		clear_device_storage();
#endif
	}

	int AssemblyCache::append(bool is_mass, const AssemblyTempStorage &temp)
	{
		AssemblyCacheDesc new_desc = copy_from_temp(is_mass, temp);
		desc_.push_back(new_desc);
		return desc_.size() - 1;
	}

	void AssemblyCache::update(int cache_id, bool is_mass, const AssemblyTempStorage &temp)
	{
		assert(cache_id >= 0 && cache_id < desc_.size());

		AssemblyCacheDesc new_desc = copy_from_temp(is_mass, temp);
		desc_[cache_id] = new_desc;
	}

	AssemblyCacheView AssemblyCache::view() const
	{
		return AssemblyCacheView{
			desc_,
			basis_values_,
			basis_grad_x_,
			basis_grad_y_,
			basis_grad_z_,
			basis_grad_phy_x_,
			basis_grad_phy_y_,
			basis_grad_phy_z_,
			physical_x_,
			physical_y_,
			physical_z_,
			det_J_,
			J_inverse_transpose_,
			weighted_measure_};
	}

#ifdef POLYFEM_WITH_CUDA
	AssemblyCacheView AssemblyCache::device_view(ExecutionPolicy policy) const
	{
		if (need_host_device_sync_)
		{
			d_desc_ = copy_to_device_async<AssemblyCacheDesc>(desc_, policy);
			d_basis_values_ = copy_to_device_async<double>(basis_values_, policy);
			d_basis_grad_x_ = copy_to_device_async<double>(basis_grad_x_, policy);
			d_basis_grad_y_ = copy_to_device_async<double>(basis_grad_y_, policy);
			d_basis_grad_z_ = copy_to_device_async<double>(basis_grad_z_, policy);
			d_basis_grad_phy_x_ = copy_to_device_async<double>(basis_grad_phy_x_, policy);
			d_basis_grad_phy_y_ = copy_to_device_async<double>(basis_grad_phy_y_, policy);
			d_basis_grad_phy_z_ = copy_to_device_async<double>(basis_grad_phy_z_, policy);
			d_physical_x_ = copy_to_device_async<double>(physical_x_, policy);
			d_physical_y_ = copy_to_device_async<double>(physical_y_, policy);
			d_physical_z_ = copy_to_device_async<double>(physical_z_, policy);
			d_det_J_ = copy_to_device_async<double>(det_J_, policy);
			d_J_inverse_transpose_ = copy_to_device_async<double>(J_inverse_transpose_, policy);
			d_weighted_measure_ = copy_to_device_async<double>(weighted_measure_, policy);

			policy.stream->sync();
			need_host_device_sync_ = false;
		}
		return AssemblyCacheView{
			*d_desc_,
			*d_basis_values_,
			*d_basis_grad_x_,
			*d_basis_grad_y_,
			*d_basis_grad_z_,
			*d_basis_grad_phy_x_,
			*d_basis_grad_phy_y_,
			*d_basis_grad_phy_z_,
			*d_physical_x_,
			*d_physical_y_,
			*d_physical_z_,
			*d_det_J_,
			*d_J_inverse_transpose_,
			*d_weighted_measure_};
	}

	void AssemblyCache::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_desc_ = {};
		d_basis_values_ = {};
		d_basis_grad_x_ = {};
		d_basis_grad_y_ = {};
		d_basis_grad_z_ = {};
		d_basis_grad_phy_x_ = {};
		d_basis_grad_phy_y_ = {};
		d_basis_grad_phy_z_ = {};
		d_physical_x_ = {};
		d_physical_y_ = {};
		d_physical_z_ = {};
		d_det_J_ = {};
		d_J_inverse_transpose_ = {};
		d_weighted_measure_ = {};
	}
#endif

	int ElementAssemblyCacheView::quad_num() const
	{
		return static_cast<int>(weighted_measure.size());
	}

	double ElementAssemblyCacheView::get_basis_value(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_values[idx];
	}

	double ElementAssemblyCacheView::get_basis_grad_x(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_grad_x[idx];
	}

	double ElementAssemblyCacheView::get_basis_grad_y(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_grad_y[idx];
	}

	double ElementAssemblyCacheView::get_basis_grad_z(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_grad_z[idx];
	}

	template <int dim>
	Eigen::Vector<double, dim> ElementAssemblyCacheView::get_basis_grad(int local_basis_id, int quad_id) const
	{
		Eigen::Vector<double, dim> ret;
		ret(0) = get_basis_grad_x(local_basis_id, quad_id);
		if constexpr (dim > 1)
			ret(1) = get_basis_grad_y(local_basis_id, quad_id);
		if constexpr (dim > 2)
			ret(2) = get_basis_grad_z(local_basis_id, quad_id);
		return ret;
	}

	template Eigen::Vector<double, 1> ElementAssemblyCacheView::get_basis_grad<1>(int, int) const;
	template Eigen::Vector<double, 2> ElementAssemblyCacheView::get_basis_grad<2>(int, int) const;
	template Eigen::Vector<double, 3> ElementAssemblyCacheView::get_basis_grad<3>(int, int) const;

	double ElementAssemblyCacheView::get_basis_grad_phy_x(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_grad_phy_x[idx];
	}

	double ElementAssemblyCacheView::get_basis_grad_phy_y(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_grad_phy_y[idx];
	}

	double ElementAssemblyCacheView::get_basis_grad_phy_z(int local_basis_id, int quad_id) const
	{
		int idx = local_basis_id * quad_num() + quad_id;
		return basis_grad_phy_z[idx];
	}

	template <int dim>
	Eigen::Vector<double, dim> ElementAssemblyCacheView::get_basis_grad_phy(int local_basis_id, int quad_id) const
	{
		Eigen::Vector<double, dim> ret;
		ret(0) = get_basis_grad_phy_x(local_basis_id, quad_id);
		if constexpr (dim > 1)
			ret(1) = get_basis_grad_phy_y(local_basis_id, quad_id);
		if constexpr (dim > 2)
			ret(2) = get_basis_grad_phy_z(local_basis_id, quad_id);
		return ret;
	}

	template Eigen::Vector<double, 1> ElementAssemblyCacheView::get_basis_grad_phy<1>(int, int) const;
	template Eigen::Vector<double, 2> ElementAssemblyCacheView::get_basis_grad_phy<2>(int, int) const;
	template Eigen::Vector<double, 3> ElementAssemblyCacheView::get_basis_grad_phy<3>(int, int) const;

	double ElementAssemblyCacheView::get_physical_x(int quad_id) const
	{
		return physical_x[quad_id];
	}

	double ElementAssemblyCacheView::get_physical_y(int quad_id) const
	{
		return physical_y[quad_id];
	}

	double ElementAssemblyCacheView::get_physical_z(int quad_id) const
	{
		return physical_z[quad_id];
	}

	template <int dim>
	Eigen::Vector<double, dim> ElementAssemblyCacheView::get_physical(int quad_id) const
	{
		Eigen::Vector<double, dim> ret;
		ret(0) = physical_x[quad_id];
		if constexpr (dim > 1)
			ret(1) = physical_y[quad_id];
		if constexpr (dim > 2)
			ret(2) = physical_z[quad_id];
		return ret;
	}

	template Eigen::Vector<double, 1> ElementAssemblyCacheView::get_physical<1>(int) const;
	template Eigen::Vector<double, 2> ElementAssemblyCacheView::get_physical<2>(int) const;
	template Eigen::Vector<double, 3> ElementAssemblyCacheView::get_physical<3>(int) const;

	double ElementAssemblyCacheView::get_det_J(int quad_id) const
	{
		return det_J[quad_id];
	}

	double ElementAssemblyCacheView::get_weighted_measure(int quad_id) const
	{
		return weighted_measure[quad_id];
	}

	Span<const double> ElementAssemblyCacheView::get_J_inverse_transpose(int quad_id, int dim) const
	{
		int idx = quad_id * dim * dim;
		return Span<const double>(J_inverse_transpose.data() + idx, dim * dim);
	}

	ElementAssemblyCacheView AssemblyCacheView::slice(int cache_id) const
	{
		const auto &d = desc[cache_id];
		return ElementAssemblyCacheView{
			d.is_empty,
			d.is_mass,
			slice_by_range(basis_values, d.basis_val_range),
			slice_by_range(basis_grad_x, d.basis_grad_x_range),
			slice_by_range(basis_grad_y, d.basis_grad_y_range),
			slice_by_range(basis_grad_z, d.basis_grad_z_range),
			slice_by_range(basis_grad_phy_x, d.basis_grad_phy_x_range),
			slice_by_range(basis_grad_phy_y, d.basis_grad_phy_y_range),
			slice_by_range(basis_grad_phy_z, d.basis_grad_phy_z_range),
			slice_by_range(physical_x, d.physical_x_range),
			slice_by_range(physical_y, d.physical_y_range),
			slice_by_range(physical_z, d.physical_z_range),
			slice_by_range(det_J, d.det_J_range),
			slice_by_range(J_inverse_transpose, d.J_inverse_transpose_range),
			slice_by_range(weighted_measure, d.weighted_measure_range)};
	}

} // namespace polyfem::assembler
