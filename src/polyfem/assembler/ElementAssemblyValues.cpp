#include "ElementAssemblyValues.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	using namespace polyfem::basis;
	using namespace polyfem::quadrature;

	// --- Public methods -----------------------------------------------------

	void ElementAssemblyValues::compute(const int el_index, const bool is_volume, const ElementBases &basis, const ElementBases &gbasis)
	{
		basis.compute_quadrature(quadrature);
		compute(el_index, is_volume, quadrature.points, basis, gbasis);
	}

	void ElementAssemblyValues::compute(
		const int el_index,
		const bool is_volume,
		const Eigen::MatrixXd &pts,
		const ElementBases &basis,
		const ElementBases &gbasis)
	{
		element_id = el_index;
		basis_values.resize(basis.bases.size());

		// check if the geometric basis is the same as the basis
		const bool is_basis_gbasis = (&basis == &gbasis);

		if (is_basis_gbasis)
			g_basis_values_cache_.resize(gbasis.bases.size());

		const int n_local_bases = int(basis.bases.size());
		const int n_local_g_bases = int(gbasis.bases.size());

		// evaluate on reference element
		basis.evaluate_bases(pts, basis_values);
		basis.evaluate_grads(pts, basis_values);

		if (is_basis_gbasis)
		{
			gbasis.evaluate_bases(pts, g_basis_values_cache_);
			gbasis.evaluate_grads(pts, g_basis_values_cache_);
		}

		for (int i = 0; i < n_local_bases; ++i)
		{
			assert(basis_values[i].val.cols() == 1);
			assert(basis_values[i].grad.cols() == pts.cols());
			basis_values[i].global = basis.bases[i].global();
		}

		if (!gbasis.has_parameterization)
		{
			return finalize_global_element(pts); // v = G(pts)
		}

		// compute geometric mapping as linear combination of geometric basis functions
		const auto &gbasis_values = is_basis_gbasis ? basis_values : g_basis_values_cache_;
		assert(gbasis_values.size() == n_local_g_bases);
		val.setZero(pts.rows(), gbasis.bases[0].global()[0].node.size());

		// loop over geometric basis functions
		for (int i = 0; i < n_local_g_bases; ++i)
		{
			const Basis &b = gbasis.bases[i];
			const auto &val_i = gbasis_values[i].val;

			assert(gbasis.has_parameterization);
			assert(val_i.size() == val.rows());

			// loop over relevant global nodes
			for (size_t j = 0; j < b.global().size(); ++j)
			{
				for (long k = 0; k < val.rows(); ++k)
				{
					// add contribution from geometric basis function + node
					val.row(k) += val_i(k) * b.global()[j].node * b.global()[j].val;
				}
			}
		}

		// compute Jacobian
		finalize(gbasis, gbasis_values, is_volume);
	}

	bool ElementAssemblyValues::is_geom_mapping_positive(
		const bool is_volume, const ElementBases &gbasis) const
	{
		// The geometry mapping is assumed positive if the element has no parameterization
		if (!gbasis.has_parameterization)
			return true;

		const int dim = gbasis.bases[0].global()[0].node.size();

		const int n_local_bases = int(gbasis.bases.size());
		if (n_local_bases <= 0)
			return true;

		Quadrature quad;
		gbasis.compute_quadrature(quad);

		std::vector<AssemblyValues> vals;
		gbasis.evaluate_grads(quad.points, vals);

		Eigen::MatrixXd dxmv = Eigen::MatrixXd::Zero(quad.points.rows(), dim);
		Eigen::MatrixXd dymv = Eigen::MatrixXd::Zero(quad.points.rows(), dim);
		Eigen::MatrixXd dzmv;
		if (is_volume)
			dzmv = Eigen::MatrixXd::Zero(quad.points.rows(), dim);

		for (int i = 0; i < n_local_bases; ++i)
		{
			const Basis &b = gbasis.bases[i];
			for (size_t j = 0; j < b.global().size(); ++j)
			{
				for (long k = 0; k < quad.points.rows(); ++k)
				{
					// ∇ₖϕᵢ Nⱼ wⱼ
					dxmv.row(k) += vals[i].grad(k, 0) * b.global()[j].node * b.global()[j].val;
					dymv.row(k) += vals[i].grad(k, 1) * b.global()[j].node * b.global()[j].val;
					if (is_volume)
						dzmv.row(k) += vals[i].grad(k, 2) * b.global()[j].node * b.global()[j].val;
				}
			}
		}

		return is_geom_mapping_positive(dxmv, dymv, dzmv);
	}

	// --- Private methods ----------------------------------------------------

	void ElementAssemblyValues::finalize(
		const ElementBases &gbasis,
		const std::vector<AssemblyValues> &gbasis_values,
		const bool is_volume)
	{
		assert(gbasis.has_parameterization);

		const int uv_dim = is_volume ? 3 : 2;
		const int dim = gbasis.bases[0].global()[0].node.size();

		// --- resize ----------------------------------------------------------

		det.resize(val.rows());    // determinant of Jacobian per quadrature point
		jac_it.resize(val.rows()); // Jacobian's inverse transpose per quadrature point

		// Gradient of the basis pre-multiplied by the inverse transpose
		// of the Jacobian of the geometric mapping of the element.
		for (auto &basis_value : basis_values)
		{
			basis_value.grad_t_m.resize(basis_value.grad.rows(), dim);
		}

		// --- resize ----------------------------------------------------------

		// Loop over quadrature points:
		for (long i = 0; i < val.rows(); ++i)
		{
			MatrixNd F = MatrixNd::Zero(uv_dim, dim);

			for (int j = 0; j < gbasis_values.size(); ++j)
			{
				const Basis &b = gbasis.bases[j];
				assert(gbasis_values[j].grad.rows() == val.rows());
				assert(gbasis_values[j].grad.cols() == uv_dim);

				for (size_t k = 0; k < b.global().size(); ++k)
				{
					// add given geometric basis function + node's contribution to the Jacobian
					for (int d = 0; d < uv_dim; ++d)
					{
						F.row(d) += gbasis_values[j].grad(i, d) * b.global()[k].node * b.global()[k].val;
					}
				}
			}

			// Save Jacobian's determinant and inverse transpose:
			if (F.rows() == F.cols())
			{
				det(i) = F.determinant();
				jac_it[i] = F.inverse().transpose();
				// std::stringstream ss;
				// ss << "F:\n"
				//    << F << "\n"
				//    << "gt_det: " << det(i) << "\n"
				//    << "gt_jac_it:\n"
				//    << jac_it[i];
				// logger().critical(ss.str());
			}
			else
			{
				// TODO: product of singular values?
				det(i) = sqrt((F * F.transpose()).determinant());
				jac_it[i] = ((F * F.transpose()).inverse()) * F;

				// logger().critical("Non-square Jacobian matrix");
				// std::stringstream ss;
				// ss << "F:\n"
				//    << F << "\n"
				//    << "det: " << det(i) << "\n"
				//    << "jac_it:\n"
				//    << jac_it[i];
				// logger().critical(ss.str());
			}

			// Save pre-multiplied gradients:
			for (size_t j = 0; j < basis_values.size(); ++j)
			{
				basis_values[j].grad_t_m.row(i) = basis_values[j].grad.row(i) * jac_it[i];
			}
		}
	}

	void ElementAssemblyValues::finalize_global_element(const Eigen::MatrixXd &v)
	{
		val = v;

		has_parameterization = false;
		det.resize(v.rows(), 1);

		jac_it.resize(v.rows());
		for (long i = 0; i < v.rows(); ++i)
			jac_it[i] = Eigen::MatrixXd::Identity(v.cols(), v.cols());

		det.setConstant(1); // volume (det of the geometric mapping)
		for (size_t j = 0; j < basis_values.size(); ++j)
			basis_values[j].grad_t_m = basis_values[j].grad; // / scaling
	}

	bool ElementAssemblyValues::is_geom_mapping_positive(
		const Eigen::MatrixXd &dx, const Eigen::MatrixXd &dy, const Eigen::MatrixXd &dz) const
	{
		assert(dx.rows() == dy.rows() && dx.cols() == dy.cols());
		assert(dz.size() == 0 || (dx.rows() == dz.rows() && dx.cols() == dz.cols()));
		assert(dx.cols() == 2 || dx.cols() == 3);
		const int rows = dz.size() ? 3 : 2;
		const int cols = dx.cols();

		for (long i = 0; i < dx.rows(); ++i)
		{
			MatrixNd F(rows, cols);
			F.row(0) = dx.row(i);
			F.row(1) = dy.row(i);
			if (rows == 3)
				F.row(2) = dz.row(i);

			double det;
			if (rows == cols)
			{
				det = F.determinant();
			}
			else
			{
				// TODO: product of singular values?
				det = (F * F.transpose()).determinant();
			}

			if (det <= 0)
			{
				return false;
			}
		}

		return true;
	}
} // namespace polyfem::assembler
