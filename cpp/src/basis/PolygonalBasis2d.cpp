#include "PolygonalBasis2d.hpp"

#include "QuadBoundarySampler.hpp"
#include "PolygonQuadrature.hpp"

#include "Harmonic.hpp"
#include "Biharmonic.hpp"

namespace poly_fem
{
	namespace
	{
		Eigen::MatrixXd compute_first_prev(const BoundaryData &bdata0, const BoundaryData &bdata1, const std::vector< ElementBases > &gbases)
		{
			Eigen::MatrixXd samples0, samples1;
			Eigen::MatrixXd mapped0, mapped1;

			bool has_samples = QuadBoundarySampler::sample(bdata0.flag == BoundaryData::RIGHT_FLAG, bdata0.flag == BoundaryData::BOTTOM_FLAG, bdata0.flag == BoundaryData::LEFT_FLAG, bdata0.flag == BoundaryData::TOP_FLAG, 2, false, samples0);
			assert(has_samples);

			has_samples = QuadBoundarySampler::sample(bdata1.flag == BoundaryData::RIGHT_FLAG, bdata1.flag == BoundaryData::BOTTOM_FLAG, bdata1.flag == BoundaryData::LEFT_FLAG, bdata1.flag == BoundaryData::TOP_FLAG, 2, false, samples1);
			assert(has_samples);

			const ElementBases &gb0=gbases[bdata0.face_id];
			const ElementBases &gb1=gbases[bdata1.face_id];

			gb0.eval_geom_mapping(samples0, mapped0);
			gb1.eval_geom_mapping(samples1, mapped1);

			if((mapped0.row(1)-mapped1.row(0)).squaredNorm() < 1e-8)
				return mapped0.row(0);

			if((mapped0.row(1)-mapped1.row(1)).squaredNorm() < 1e-8)
				return mapped0.row(0);

			if((mapped0.row(0)-mapped1.row(0)).squaredNorm() < 1e-8)
				return mapped0.row(1);

			if((mapped0.row(0)-mapped1.row(1)).squaredNorm() < 1e-8)
				return mapped0.row(1);

			assert(false);
			return mapped0.row(1);
		}

		void sample_polygon(const int element_index, const int samples_res, const Mesh2D &mesh, std::map<int, BoundaryData> &poly_edge_to_data, const std::vector< ElementBases > &bases, const std::vector< ElementBases > &gbases, std::vector<int> &local_to_global, const double eps, const bool c1_continuous, Eigen::MatrixXd &boundary_samples, Eigen::MatrixXd &poly_samples, const Eigen::MatrixXd &basis_integrals, Eigen::MatrixXd &rhs)
		{
			const int n_edges = mesh.n_element_vertices(element_index);

			const int poly_local_n = (samples_res - 1)/3;
			const int n_samples      = (samples_res - 1) * n_edges;
			const int n_poly_samples = poly_local_n * n_edges;

			boundary_samples.resize(n_samples, 2);
			boundary_samples.setConstant(0);
			poly_samples.resize(n_poly_samples, 2);

			Eigen::MatrixXd samples, mapped, basis_val, grad_basis_val;
			std::vector<Eigen::MatrixXd> grads;

			Navigation::Index index = mesh.get_index_from_face(element_index);
			for(int i = 0; i < n_edges; ++i)
			{
				const BoundaryData &bdata = poly_edge_to_data[index.edge];
				local_to_global.insert(local_to_global.end(), bdata.node_id.begin(), bdata.node_id.end());

				index = mesh.next_around_face(index);
			}

			std::sort( local_to_global.begin(), local_to_global.end() );
			local_to_global.erase( std::unique( local_to_global.begin(), local_to_global.end() ), local_to_global.end() );
            // assert(int(local_to_global.size()) <= n_edges);

			rhs = Eigen::MatrixXd::Zero(n_samples + (c1_continuous? (2*n_samples): 0), local_to_global.size());
            // rhs.resize(n_samples + (c1_continuous? (2*n_samples): 0), local_to_global.size());


			index = mesh.get_index_from_face(element_index);

			Eigen::MatrixXd prev = compute_first_prev(poly_edge_to_data[index.edge], poly_edge_to_data[mesh.next_around_face(index).edge], gbases);

			for(int i = 0; i < n_edges; ++i)
			{
                //no boundary polygons
				assert(mesh.switch_face(index).face >= 0);

				const BoundaryData &bdata = poly_edge_to_data[index.edge];

				const ElementBases &b=bases[bdata.face_id];
				const ElementBases &gb=gbases[bdata.face_id];

				assert(bdata.face_id == mesh.switch_face(index).face);

				const bool has_samples = QuadBoundarySampler::sample(bdata.flag == BoundaryData::RIGHT_FLAG, bdata.flag == BoundaryData::BOTTOM_FLAG, bdata.flag == BoundaryData::LEFT_FLAG, bdata.flag == BoundaryData::TOP_FLAG, samples_res, false, samples);
				assert(has_samples);

				gb.eval_geom_mapping(samples, mapped);

				if(c1_continuous)
				{
					gb.eval_geom_mapping_grads(samples, grads);
				}

				bool must_reverse = true;
				{
					const double dist_first = (mapped.row(0)-prev).norm();

					if(dist_first < 1e-8)
					{
						samples = samples.block(1, 0, samples.rows()-1, samples.cols()).eval();
						mapped = mapped.block(1, 0, mapped.rows()-1, mapped.cols()).eval();

						must_reverse = false;
					}
					else
					{
                        assert((mapped.row(mapped.rows()-1) - prev).norm() < 1e-8);

						samples = samples.block(0, 0, samples.rows()-1, samples.cols()).eval();
						mapped = mapped.block(0, 0, mapped.rows()-1, mapped.cols()).eval();

						mapped = mapped.colwise().reverse().eval();

						must_reverse = true;
					}
				}

                // assert(bdata.node_id.size() == 3);
				for(std::size_t bi = 0; bi < bdata.node_id.size(); ++bi)
				{
					const int local_index = bdata.local_indices[bi];
                    // assert(b.bases[local_index].global_index() == bdata.node_id[bi]);
					const long basis_index = std::distance(local_to_global.begin(), std::find(local_to_global.begin(), local_to_global.end(), bdata.node_id[bi]));

					b.bases[local_index].basis(samples, basis_val);

					if(must_reverse)
						basis_val = basis_val.reverse().eval();
					rhs.block(i*(samples_res-1), basis_index, basis_val.rows(), 1) += basis_val * bdata.vals[bi];

					if(c1_continuous)
					{
						b.bases[local_index].grad(samples, grad_basis_val);

						if(must_reverse)
							grad_basis_val = grad_basis_val.colwise().reverse().eval();

						for(long k = 0; k < grad_basis_val.rows(); ++k)
						{
							const Eigen::MatrixXd trans_grad = grad_basis_val.row(k) * grads[k];

							rhs(n_samples + 2*i*(samples_res-1) + 2*k,     basis_index) = trans_grad(0);
							rhs(n_samples + 2*i*(samples_res-1) + 2*k + 1, basis_index) = trans_grad(1);
						}
					}

				}


				prev = mapped.row(mapped.rows()-1);
				assert(mapped.rows() == (samples_res-1));
				boundary_samples.block(i*(samples_res-1), 0, mapped.rows(), mapped.cols()) = mapped;
				const double offset = double(samples_res-1)/(poly_local_n+1);

				for(int j = 0; j < poly_local_n; ++j)
				{
					const int poly_index = (j+1)*offset;

					if(eps > 0)
					{
						const int im = poly_index - 1;
						const int ip = poly_index + 1;

						const Eigen::MatrixXd e0 = (mapped.row(poly_index) - mapped.row(im)).normalized();
						const Eigen::MatrixXd e1 = (mapped.row(ip) - mapped.row(poly_index)).normalized();

						const Eigen::Vector2d n0(e0(1), -e0(0));
						const Eigen::Vector2d n1(e1(1), -e1(0));
                        //TODO discad point if inside
						const Eigen::Vector2d n = (n0+n1).normalized();

						poly_samples.row(i*poly_local_n+j) = n.transpose()*eps + mapped.row(poly_index);
					}
					else
						poly_samples.row(i*poly_local_n+j) = mapped.row(poly_index);
				}

				index = mesh.next_around_face(index);
			}
		}
	}

	void PolygonalBasis2d::build_bases(
		const int samples_res,
		const Mesh2D &mesh,
		const int n_bases,
		const std::vector<ElementType> &els_tag,
		const int quadrature_order,
		const std::vector< ElementAssemblyValues > &values,
		const std::vector< ElementAssemblyValues > &gvalues,
		std::vector< ElementBases > &bases,
		const std::vector< ElementBases > &gbases,
		std::map<int, BoundaryData> &poly_edge_to_data,
		std::map<int, Eigen::MatrixXd> &polys)
	{
		using std::max;
		assert(!mesh.is_volume());

		const int n_els = mesh.n_elements();

		if(poly_edge_to_data.empty())
			return;

		Eigen::MatrixXd basis_integrals(n_bases, 2);
		basis_integrals.setZero();

		for(int e = 0; e < n_els; ++e)
		{
			const ElementAssemblyValues &vals = values[e];
			const ElementAssemblyValues &gvals = gvalues[e];

			const int n_local_bases = int(vals.basis_values.size());
			for(int j = 0; j < n_local_bases; ++j)
			{
				const AssemblyValues &v=vals.basis_values[j];
				const double integralx = (v.grad_t_m.col(0).array() * gvals.det.array() * vals.quadrature.weights.array()).sum();
				const double integraly = (v.grad_t_m.col(1).array() * gvals.det.array() * vals.quadrature.weights.array()).sum();

				for(std::size_t ii = 0; ii < v.global.size(); ++ii)
				{
					basis_integrals(v.global[ii].index, 0) += integralx * v.global[ii].val;
					basis_integrals(v.global[ii].index, 1) += integraly * v.global[ii].val;
				}
			}
		}

		const bool use_harmonic = true;
		const bool c1_continuous = !use_harmonic && true;


		PolygonQuadrature poly_quad;
		Eigen::Matrix2d det_mat;
		Eigen::MatrixXd p0, p1;

		for(int e = 0; e < n_els; ++e)
		{
			if(els_tag[e] != ElementType::InteriorPolytope && els_tag[e] != ElementType::BoundaryPolytope)
				continue;

			const int n_edges = mesh.n_element_vertices(e);
			double area = 0;
			for(int i = 0; i < n_edges; ++i)
			{
				const int ip = (i + 1) % n_edges;

				mesh.point(mesh.vertex_global_index(e, i), p0);
				mesh.point(mesh.vertex_global_index(e, ip), p1);
				det_mat.row(0) = p0;
				det_mat.row(1) = p1;

				area += det_mat.determinant();
			}
			area = fabs(area);
            // const double eps = use_harmonic ? (0.08*area) : 0;
			const double eps = 0.08*area;

			std::vector<int> local_to_global;
			Eigen::MatrixXd boundary_samples, poly_samples;
			Eigen::MatrixXd rhs;

			sample_polygon(e, samples_res, mesh, poly_edge_to_data, bases, gbases, local_to_global, eps, c1_continuous, boundary_samples, poly_samples, basis_integrals, rhs);

			ElementBases &b=bases[e];
			b.has_parameterization = false;
			poly_quad.get_quadrature(boundary_samples, quadrature_order, b.quadrature);

			polys[e] = boundary_samples;

			const int n_poly_bases = int(local_to_global.size());
			b.bases.resize(n_poly_bases);

			if(use_harmonic)
			{
                // igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
                // viewer.data.add_points(poly_samples, Eigen::Vector3d(0,1,1).transpose());

                // Eigen::MatrixXd asd(boundary_samples.rows(), 3);
                // asd.col(0)=boundary_samples.col(0);
                // asd.col(1)=boundary_samples.col(1);
                // asd.col(2)=rhs.col(0);
                // viewer.data.add_points(asd, Eigen::Vector3d(1,0,1).transpose());

                // for(int asd = 0; asd < boundary_samples.rows(); ++asd)
                    // viewer.data.add_label(boundary_samples.row(asd), std::to_string(asd));

				Eigen::MatrixXd local_basis_integral(rhs.cols(), 2);
				for(long k = 0; k < rhs.cols(); ++k)
				{
					local_basis_integral(k, 0) = -basis_integrals(local_to_global[k],0);
					local_basis_integral(k, 1) = -basis_integrals(local_to_global[k],1);
				}

				Harmonic harmonic(poly_samples, boundary_samples, local_basis_integral, rhs);

				for(int i = 0; i < n_poly_bases; ++i)
				{
					b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd::Zero(1,2));
					b.bases[i].set_basis([harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { harmonic.basis(i, uv, val); });
					b.bases[i].set_grad( [harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { harmonic.grad(i, uv, val); });
				}
			}
			else
			{
				Biharmonic biharmonic(poly_samples, boundary_samples, rhs);

				for(int i = 0; i < n_poly_bases; ++i)
				{
					b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd::Zero(1,2));
					b.bases[i].set_basis([biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.basis(i, uv, val); });
					b.bases[i].set_grad( [biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.grad(i, uv, val); });
				}
			}
		}
	}
}