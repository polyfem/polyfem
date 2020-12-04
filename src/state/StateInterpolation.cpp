#include <polyfem/State.hpp>

#include <polyfem/BoundarySampler.hpp>
#include <polyfem/RefElementSampler.hpp>

#include <polyfem/HexQuadrature.hpp>
#include <polyfem/QuadQuadrature.hpp>
#include <polyfem/TetQuadrature.hpp>
#include <polyfem/TriQuadrature.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <polyfem/Logger.hpp>

#include <igl/AABB.h>
#include <igl/per_face_normals.h>

namespace polyfem
{

    void State::interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        assert(mesh->is_volume());

        const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

        Eigen::MatrixXd points, uv;
        Eigen::VectorXd weights;

        int actual_dim = 1;
        if (!problem->is_scalar())
            actual_dim = 3;

        igl::AABB<Eigen::MatrixXd, 3> tree;
        tree.init(pts, faces);

        const auto &gbases = iso_parametric() ? bases : geom_bases;
        result.resize(faces.rows(), actual_dim);
        result.setConstant(std::numeric_limits<double>::quiet_NaN());

        int counter = 0;

        for (int e = 0; e < mesh3d.n_elements(); ++e)
        {
            const ElementBases &gbs = gbases[e];
            const ElementBases &bs = bases[e];

            for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
            {
                const int face_id = mesh3d.cell_face(e, lf);
                if (!mesh3d.is_boundary_face(face_id))
                    continue;

                if (mesh3d.is_simplex(e))
                    BoundarySampler::quadrature_for_tri_face(lf, 4, face_id, mesh3d, uv, points, weights);
                else if (mesh3d.is_cube(e))
                    BoundarySampler::quadrature_for_quad_face(lf, 4, face_id, mesh3d, uv, points, weights);
                else
                    assert(false);

                ElementAssemblyValues vals;
                vals.compute(e, true, points, bs, gbs);
                RowVectorNd loc_val(actual_dim);
                loc_val.setZero();

                // UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

                // const auto nodes = bs.local_nodes_for_primitive(face_id, mesh3d);

                // for(long n = 0; n < nodes.size(); ++n)
                for (size_t j = 0; j < bs.bases.size(); ++j)
                {
                    // const auto &b = bs.bases[nodes(n)];
                    // const AssemblyValues &v = vals.basis_values[nodes(n)];
                    const AssemblyValues &v = vals.basis_values[j];
                    for (int d = 0; d < actual_dim; ++d)
                    {
                        for (size_t g = 0; g < v.global.size(); ++g)
                        {
                            loc_val(d) += (v.global[g].val * v.val.array() * fun(v.global[g].index * actual_dim + d) * weights.array()).sum();
                        }
                    }
                }

                int I;
                Eigen::RowVector3d C;
                const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

                const double dist = tree.squared_distance(pts, faces, bary, I, C);
                assert(dist < 1e-16);

                assert(std::isnan(result(I, 0)));
                if (compute_avg)
                    result.row(I) = loc_val / weights.sum();
                else
                    result.row(I) = loc_val;
                ++counter;
            }
        }

        assert(counter == result.rows());
    }

    void State::interpolate_boundary_function_at_vertices(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        if (!mesh->is_volume())
        {
            logger().error("This function works only on volumetric meshes!");
            return;
        }

        assert(mesh->is_volume());

        const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

        Eigen::MatrixXd points;

        int actual_dim = 1;
        if (!problem->is_scalar())
            actual_dim = 3;

        igl::AABB<Eigen::MatrixXd, 3> tree;
        tree.init(pts, faces);

        const auto &gbases = iso_parametric() ? bases : geom_bases;
        result.resize(pts.rows(), actual_dim);
        result.setZero();

        for (int e = 0; e < mesh3d.n_elements(); ++e)
        {
            const ElementBases &gbs = gbases[e];
            const ElementBases &bs = bases[e];

            for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
            {
                const int face_id = mesh3d.cell_face(e, lf);
                if (!mesh3d.is_boundary_face(face_id))
                    continue;

                if (mesh3d.is_simplex(e))
                    autogen::p_nodes_3d(1, points);
                else if (mesh3d.is_cube(e))
                    autogen::q_nodes_3d(1, points);
                else
                    assert(false);

                ElementAssemblyValues vals;
                vals.compute(e, true, points, bs, gbs);
                MatrixXd loc_val(points.rows(), actual_dim);
                loc_val.setZero();

                // UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

                for (size_t j = 0; j < bs.bases.size(); ++j)
                {
                    const Basis &b = bs.bases[j];
                    const AssemblyValues &v = vals.basis_values[j];

                    for (int d = 0; d < actual_dim; ++d)
                    {
                        for (size_t ii = 0; ii < b.global().size(); ++ii)
                            loc_val.col(d) += b.global()[ii].val * v.val * fun(b.global()[ii].index * actual_dim + d);
                    }
                }

                int I;
                Eigen::RowVector3d C;
                const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

                const double dist = tree.squared_distance(pts, faces, bary, I, C);
                assert(dist < 1e-16);

                for (int lv_id = 0; lv_id < faces.cols(); ++lv_id)
                {
                    const int v_id = faces(I, lv_id);
                    const auto p = pts.row(v_id);
                    const auto &mapped = vals.val;

                    bool found = false;

                    for (int n = 0; n < mapped.rows(); ++n)
                    {
                        if ((p - mapped.row(n)).norm() < 1e-10)
                        {
                            result.row(v_id) = loc_val.row(n);
                            found = true;
                            break;
                        }
                    }

                    assert(found);
                }
            }
        }
    }

    void State::interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result)
    {
        interpolate_boundary_tensor_function(pts, faces, fun, Eigen::MatrixXd::Zero(pts.rows(), pts.cols()), compute_avg, result);
    }

    void State::interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const MatrixXd &disp, const bool compute_avg, MatrixXd &result)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        if (disp.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        if (!mesh->is_volume())
        {
            logger().error("This function works only on volumetric meshes!");
            return;
        }
        if (problem->is_scalar())
        {
            logger().error("Define a tensor problem!");
            return;
        }

        assert(mesh->is_volume());
        assert(!problem->is_scalar());

        const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

        MatrixXd normals;
        igl::per_face_normals((pts + disp).eval(), faces, normals);
        // std::cout<<normals<<std::endl;

        Eigen::MatrixXd points, uv;
        Eigen::VectorXd weights;

        const int actual_dim = 3;

        igl::AABB<Eigen::MatrixXd, 3> tree;
        tree.init(pts, faces);

        const auto &gbases = iso_parametric() ? bases : geom_bases;
        result.resize(faces.rows(), actual_dim);
        result.setConstant(std::numeric_limits<double>::quiet_NaN());

        int counter = 0;

        for (int e = 0; e < mesh3d.n_elements(); ++e)
        {
            const ElementBases &gbs = gbases[e];
            const ElementBases &bs = bases[e];

            for (int lf = 0; lf < mesh3d.n_cell_faces(e); ++lf)
            {
                const int face_id = mesh3d.cell_face(e, lf);
                if (!mesh3d.is_boundary_face(face_id))
                    continue;

                if (mesh->is_simplex(e))
                    BoundarySampler::quadrature_for_tri_face(lf, 4, face_id, mesh3d, uv, points, weights);
                else if (mesh->is_cube(e))
                    BoundarySampler::quadrature_for_quad_face(lf, 4, face_id, mesh3d, uv, points, weights);
                else
                    assert(false);

                // ElementAssemblyValues vals;
                // vals.compute(e, true, points, bs, gbs);
                Eigen::MatrixXd loc_val;

                // UIState::ui_state().debug_data().add_points(vals.val, Eigen::RowVector3d(1,0,0));

                // const auto nodes = bs.local_nodes_for_primitive(face_id, mesh3d);
                assembler.compute_tensor_value(formulation(), e, bs, gbs, points, fun, loc_val);
                Eigen::VectorXd tmp(loc_val.cols());
                for (int d = 0; d < loc_val.cols(); ++d)
                    tmp(d) = (loc_val.col(d).array() * weights.array()).sum();
                const Eigen::MatrixXd tensor = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 3, 3);

                int I;
                Eigen::RowVector3d C;
                const Eigen::RowVector3d bary = mesh3d.face_barycenter(face_id);

                const double dist = tree.squared_distance(pts, faces, bary, I, C);
                assert(dist < 1e-16);

                assert(std::isnan(result(I, 0)));
                result.row(I) = normals.row(I) * tensor;
                if (compute_avg)
                    result.row(I) /= weights.sum();
                ++counter;
            }
        }

        assert(counter == result.rows());
    }

    void State::average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool boundary_only)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        if (problem->is_scalar())
        {
            logger().error("Define a tensor problem!");
            return;
        }

        assert(!problem->is_scalar());
        const int actual_dim = mesh->dimension();

        MatrixXd avg_scalar(n_bases, 1);
        // MatrixXd avg_tensor(n_points * actual_dim*actual_dim, 1);
        MatrixXd areas(n_bases, 1);
        avg_scalar.setZero();
        // avg_tensor.setZero();
        areas.setZero();

        Eigen::MatrixXd local_val;
        const auto &gbases = iso_parametric() ? bases : geom_bases;

        ElementAssemblyValues vals;
        for (int i = 0; i < int(bases.size()); ++i)
        {
            const ElementBases &bs = bases[i];
            const ElementBases &gbs = gbases[i];
            Eigen::MatrixXd local_pts;

            if (mesh->is_simplex(i))
            {
                if (mesh->dimension() == 3)
                    autogen::p_nodes_3d(bs.bases.front().order(), local_pts);
                else
                    autogen::p_nodes_2d(bs.bases.front().order(), local_pts);
            }
            else
            {
                if (mesh->dimension() == 3)
                    autogen::q_nodes_3d(bs.bases.front().order(), local_pts);
                else
                    autogen::q_nodes_2d(bs.bases.front().order(), local_pts);
            }
            // else if(mesh->is_cube(i))
            // 	local_pts = sampler.cube_points();
            // // else
            // 	// local_pts = vis_pts_poly[i];

            vals.compute(i, actual_dim == 3, bases[i], gbases[i]);
            const Quadrature &quadrature = vals.quadrature;
            const double area = (vals.det.array() * quadrature.weights.array()).sum();

            assembler.compute_scalar_value(formulation(), i, bs, gbs, local_pts, fun, local_val);
            // assembler.compute_tensor_value(formulation(), i, bs, gbs, local_pts, fun, local_val);

            for (size_t j = 0; j < bs.bases.size(); ++j)
            {
                const Basis &b = bs.bases[j];
                if (b.global().size() > 1)
                    continue;

                auto &global = b.global().front();
                areas(global.index) += area;
                avg_scalar(global.index) += local_val(j) * area;
            }
        }

        avg_scalar.array() /= areas.array();

        interpolate_function(n_points, 1, bases, avg_scalar, result_scalar, boundary_only);
        // interpolate_function(n_points, actual_dim*actual_dim, bases, avg_tensor, result_tensor, boundary_only);
    }

    void State::compute_vertex_values(int actual_dim,
                                      const std::vector<ElementBases> &basis,
                                      const MatrixXd &fun,
                                      Eigen::MatrixXd &result)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        if (!mesh->is_volume())
        {
            logger().error("This function works only on volumetric meshes!");
            return;
        }

        if (!mesh)
        {
            return;
        }
        if (!mesh->is_volume())
        {
            return;
        }
        const Mesh3D &mesh3d = *dynamic_cast<const Mesh3D *>(mesh.get());

        result.resize(mesh3d.n_vertices(), actual_dim);
        result.setZero();

        // std::array<int, 8> get_ordered_vertices_from_hex(const int element_index) const;
        // std::array<int, 4> get_ordered_vertices_from_tet(const int element_index) const;

        const auto &sampler = RefElementSampler::sampler();
        std::vector<AssemblyValues> tmp;
        std::vector<bool> marked(mesh3d.n_vertices(), false);
        for (int i = 0; i < int(basis.size()); ++i)
        {
            const ElementBases &bs = basis[i];
            MatrixXd local_pts;
            std::vector<int> vertices;

            if (mesh->is_simplex(i))
            {
                local_pts = sampler.simplex_corners();
                auto vtx = mesh3d.get_ordered_vertices_from_tet(i);
                vertices.assign(vtx.begin(), vtx.end());
            }
            else if (mesh->is_cube(i))
            {
                local_pts = sampler.cube_corners();
                auto vtx = mesh3d.get_ordered_vertices_from_hex(i);
                vertices.assign(vtx.begin(), vtx.end());
            }
            //TODO poly?
            assert((int)vertices.size() == (int)local_pts.rows());

            MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);
            bs.evaluate_bases(local_pts, tmp);
            for (size_t j = 0; j < bs.bases.size(); ++j)
            {
                const Basis &b = bs.bases[j];

                for (int d = 0; d < actual_dim; ++d)
                {
                    for (size_t ii = 0; ii < b.global().size(); ++ii)
                        local_res.col(d) += b.global()[ii].val * tmp[j].val * fun(b.global()[ii].index * actual_dim + d);
                }
            }

            for (size_t lv = 0; lv < vertices.size(); ++lv)
            {
                int v = vertices[lv];
                if (marked[v])
                {
                    assert((result.row(v) - local_res.row(lv)).norm() < 1e-6);
                }
                else
                {
                    result.row(v) = local_res.row(lv);
                    marked[v] = true;
                }
            }
        }
    }

    void flattened_tensor_coeffs(const MatrixXd &S, MatrixXd &X)
    {
        if (S.cols() == 4)
        {
            X.resize(S.rows(), 3);
            X.col(0) = S.col(0);
            X.col(1) = S.col(3);
            X.col(2) = S.col(1);
        }
        else if (S.cols() == 9)
        {
            // [S11, S22, S33, S12, S13, S23]
            X.resize(S.rows(), 6);
            X.col(0) = S.col(0);
            X.col(1) = S.col(4);
            X.col(2) = S.col(8);
            X.col(3) = S.col(1);
            X.col(4) = S.col(2);
            X.col(5) = S.col(5);
        }
        else
        {
            logger().error("Invalid tensor dimensions.");
        }
    }

    void State::compute_stress_at_quadrature_points(const MatrixXd &fun, Eigen::MatrixXd &result, Eigen::VectorXd &von_mises)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }
        if (problem->is_scalar())
        {
            logger().error("Define a tensor problem!");
            return;
        }

        const int actual_dim = mesh->dimension();
        assert(!problem->is_scalar());

        Eigen::MatrixXd local_val, local_stress, local_mises;
        const auto &gbases = iso_parametric() ? bases : geom_bases;

        int num_quadr_pts = 0;
        result.resize(disc_orders.sum(), actual_dim == 2 ? 3 : 6);
        result.setZero();
        von_mises.resize(disc_orders.sum(), 1);
        von_mises.setZero();
        for (int e = 0; e < mesh->n_elements(); ++e)
        {
            // Compute quadrature points for element
            Quadrature quadr;
            if (mesh->is_simplex(e))
            {
                if (mesh->is_volume())
                {
                    TetQuadrature f;
                    f.get_quadrature(disc_orders(e), quadr);
                }
                else
                {
                    TriQuadrature f;
                    f.get_quadrature(disc_orders(e), quadr);
                }
            }
            else if (mesh->is_cube(e))
            {
                if (mesh->is_volume())
                {
                    HexQuadrature f;
                    f.get_quadrature(disc_orders(e), quadr);
                }
                else
                {
                    QuadQuadrature f;
                    f.get_quadrature(disc_orders(e), quadr);
                }
            }
            else
            {
                continue;
            }

            assembler.compute_scalar_value(formulation(), e, bases[e], gbases[e],
                                           quadr.points, fun, local_mises);
            assembler.compute_tensor_value(formulation(), e, bases[e], gbases[e],
                                           quadr.points, fun, local_val);

            if (num_quadr_pts + local_val.rows() >= result.rows())
            {
                result.conservativeResize(
                    std::max(num_quadr_pts + local_val.rows() + 1, 2 * result.rows()),
                    result.cols());
                von_mises.conservativeResize(result.rows(), von_mises.cols());
            }
            flattened_tensor_coeffs(local_val, local_stress);
            result.block(num_quadr_pts, 0, local_stress.rows(), local_stress.cols()) = local_stress;
            von_mises.block(num_quadr_pts, 0, local_mises.rows(), local_mises.cols()) = local_mises;
            num_quadr_pts += local_val.rows();
        }
        result.conservativeResize(num_quadr_pts, result.cols());
        von_mises.conservativeResize(num_quadr_pts, von_mises.cols());
    }

    void State::interpolate_function(const int n_points, const MatrixXd &fun, MatrixXd &result, const bool boundary_only)
    {
        int actual_dim = 1;
        if (!problem->is_scalar())
            actual_dim = mesh->dimension();
        interpolate_function(n_points, actual_dim, bases, fun, result, boundary_only);
    }

    void State::interpolate_function(const int n_points, const int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, MatrixXd &result, const bool boundary_only)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }

        std::vector<AssemblyValues> tmp;

        result.resize(n_points, actual_dim);

        int index = 0;
        const auto &sampler = RefElementSampler::sampler();

        Eigen::MatrixXi vis_faces_poly;

        for (int i = 0; i < int(basis.size()); ++i)
        {
            const ElementBases &bs = basis[i];
            MatrixXd local_pts;

            if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
                continue;

            if (mesh->is_simplex(i))
                local_pts = sampler.simplex_points();
            else if (mesh->is_cube(i))
                local_pts = sampler.cube_points();
            else
            {
                if (mesh->is_volume())
                    sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, local_pts, vis_faces_poly);
                else
                    sampler.sample_polygon(polys[i], local_pts, vis_faces_poly);
            }

            MatrixXd local_res = MatrixXd::Zero(local_pts.rows(), actual_dim);
            bs.evaluate_bases(local_pts, tmp);
            for (size_t j = 0; j < bs.bases.size(); ++j)
            {
                const Basis &b = bs.bases[j];

                for (int d = 0; d < actual_dim; ++d)
                {
                    for (size_t ii = 0; ii < b.global().size(); ++ii)
                        local_res.col(d) += b.global()[ii].val * tmp[j].val * fun(b.global()[ii].index * actual_dim + d);
                }
            }

            result.block(index, 0, local_res.rows(), actual_dim) = local_res;
            index += local_res.rows();
        }
    }

    void State::interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, MatrixXd &result, MatrixXd &result_grad)
    {
        interpolate_at_local_vals(el_index, local_pts, sol, result, result_grad);
    }

    void State::interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }

        int actual_dim = 1;
        if (!problem->is_scalar())
            actual_dim = mesh->dimension();

        assert(local_pts.cols() == mesh->dimension());
        assert(fun.size() == sol.size());
        assert(fun.rows() == sol.rows());

        const auto &gbases = iso_parametric() ? bases : geom_bases;
        const ElementBases &gbs = gbases[el_index];
        const ElementBases &bs = bases[el_index];

        ElementAssemblyValues vals;
        vals.compute(el_index, mesh->is_volume(), local_pts, bs, gbs);

        result.resize(vals.val.rows(), actual_dim);
        result.setZero();

        result_grad.resize(vals.val.rows(), mesh->dimension() * actual_dim);
        result_grad.setZero();

        const int n_loc_bases = int(vals.basis_values.size());

        for (int i = 0; i < n_loc_bases; ++i)
        {
            const auto &val = vals.basis_values[i];

            for (size_t ii = 0; ii < val.global.size(); ++ii)
            {
                for (int d = 0; d < actual_dim; ++d)
                {
                    result.col(d) += val.global[ii].val * fun(val.global[ii].index * actual_dim + d) * val.val;
                    result_grad.block(0, d * val.grad_t_m.cols(), result_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * fun(val.global[ii].index * actual_dim + d) * val.grad_t_m;
                }
            }
        }
    }

    void State::compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }

        result.resize(n_points, 1);
        assert(!problem->is_scalar());

        int index = 0;
        const auto &sampler = RefElementSampler::sampler();

        Eigen::MatrixXi vis_faces_poly;
        Eigen::MatrixXd local_val;
        const auto &gbases = iso_parametric() ? bases : geom_bases;

        for (int i = 0; i < int(bases.size()); ++i)
        {
            if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
                continue;

            const ElementBases &bs = bases[i];
            const ElementBases &gbs = gbases[i];
            Eigen::MatrixXd local_pts;

            if (mesh->is_simplex(i))
                local_pts = sampler.simplex_points();
            else if (mesh->is_cube(i))
                local_pts = sampler.cube_points();
            else
            {
                if (mesh->is_volume())
                    sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, local_pts, vis_faces_poly);
                else
                    sampler.sample_polygon(polys[i], local_pts, vis_faces_poly);
            }

            assembler.compute_scalar_value(formulation(), i, bs, gbs, local_pts, fun, local_val);

            result.block(index, 0, local_val.rows(), 1) = local_val;
            index += local_val.rows();
        }
    }

    void State::compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool boundary_only)
    {
        if (!mesh)
        {
            logger().error("Load the mesh first!");
            return;
        }
        if (fun.size() <= 0)
        {
            logger().error("Solve the problem first!");
            return;
        }

        const int actual_dim = mesh->dimension();
        result.resize(n_points, actual_dim * actual_dim);
        assert(!problem->is_scalar());

        int index = 0;
        const auto &sampler = RefElementSampler::sampler();

        Eigen::MatrixXi vis_faces_poly;
        Eigen::MatrixXd local_val;
        const auto &gbases = iso_parametric() ? bases : geom_bases;

        for (int i = 0; i < int(bases.size()); ++i)
        {
            if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
                continue;

            const ElementBases &bs = bases[i];
            const ElementBases &gbs = gbases[i];
            Eigen::MatrixXd local_pts;

            if (mesh->is_simplex(i))
                local_pts = sampler.simplex_points();
            else if (mesh->is_cube(i))
                local_pts = sampler.cube_points();
            else
            {
                if (mesh->is_volume())
                    sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, local_pts, vis_faces_poly);
                else
                    sampler.sample_polygon(polys[i], local_pts, vis_faces_poly);
            }

            assembler.compute_tensor_value(formulation(), i, bs, gbs, local_pts, fun, local_val);

            result.block(index, 0, local_val.rows(), local_val.cols()) = local_val;
            index += local_val.rows();
        }
    }
} // namespace polyfem