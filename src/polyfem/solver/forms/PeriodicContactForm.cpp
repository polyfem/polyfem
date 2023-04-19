#include "PeriodicContactForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>
#include <polyfem/State.hpp>

#include <iostream>

// #include <polyfem/utils/AutodiffTypes.hpp>

// Eigen::VectorXd random_coeffs;

namespace polyfem::solver
{
    // namespace {
	// 	typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> Diff;

	// 	template <typename T>
	// 	Eigen::Matrix<T, Eigen::Dynamic, 1> vector_func(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x, const Eigen::VectorXd &u, const Eigen::VectorXi &tiled_to_single, const ipc::CollisionMesh &collision_mesh)
	// 	{
    //         const int dim = collision_mesh.dim();
    //         const int n_single_dof_ = tiled_to_single.maxCoeff() + 1;

    //         // single to tiled u
    //         Eigen::Matrix<T, Eigen::Dynamic, 1> tiled_u;
    //         tiled_u.setZero(tiled_to_single.size() * dim);
    //         for (int i = 0; i < collision_mesh.num_vertices(); i++)
    //         {
    //             const int i_full = collision_mesh.to_full_vertex_id(i);
    //             for (int d = 0; d < dim; d++)
    //                 tiled_u(i_full * dim + d) += u(tiled_to_single(i_full) * dim + d);
                
    //             for (int p = 0; p < dim; p++)
    //                 for (int q = 0; q < dim; q++)
    //                     tiled_u(i_full * dim + p) += u(n_single_dof_ * dim + p * dim + q) * x(i * dim + q);
    //         }

    //         // tiled vec
    //         Eigen::Matrix<T, Eigen::Dynamic, 1> tiled_vec = tiled_u.array() * random_coeffs.cast<T>().array();

    //         // tiled to single
	// 		Eigen::Matrix<T, Eigen::Dynamic, 1> vec;
    //         vec.setZero(u.size());
    //         for (int i = 0; i < collision_mesh.num_vertices(); i++)
    //         {
    //             const int i_full = collision_mesh.to_full_vertex_id(i);
    //             for (int d = 0; d < dim; d++)
    //                 vec(tiled_to_single(i_full) * dim + d) += tiled_vec(i_full * dim + d);
                
    //             for (int p = 0; p < dim; p++)
    //                 for (int q = 0; q < dim; q++)
    //                     vec(n_single_dof_ * dim + p * dim + q) += x(i * dim + q) * tiled_vec(i_full * dim + p);
    //         }

    //         return vec;
	// 	}

	// 	Eigen::MatrixXd vector_func_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &u, const Eigen::VectorXi &tiled_to_single, const ipc::CollisionMesh &collision_mesh)
	// 	{
	// 		DiffScalarBase::setVariableCount(x.size());
	// 		Eigen::Matrix<Diff, Eigen::Dynamic, 1> shape(x.size());
	// 		for (int i = 0; i < x.size(); i++)
	// 			shape(i) = Diff(i, x(i));
	// 		auto shape_deriv = vector_func(shape, u, tiled_to_single, collision_mesh);

	// 		Eigen::MatrixXd grad(u.size(), x.size());
    //         for (int j = 0; j < x.size(); ++j)
	// 		    for (int i = 0; i < u.size(); ++i)
	// 			    grad(i, j) = shape_deriv(i).getGradient()(j);

	// 		return grad;
	// 	}
    // }
	PeriodicContactForm::PeriodicContactForm(const ipc::CollisionMesh &periodic_collision_mesh,
                        const Eigen::VectorXi &tiled_to_single,
                        const double dhat,
                        const double avg_mass,
                        const bool use_convergent_formulation,
                        const bool use_adaptive_barrier_stiffness,
                        const bool is_time_dependent,
                        const ipc::BroadPhaseMethod broad_phase_method,
                        const double ccd_tolerance,
                        const int ccd_max_iterations) : ContactForm(periodic_collision_mesh, dhat, avg_mass, use_convergent_formulation, use_adaptive_barrier_stiffness, is_time_dependent, broad_phase_method, ccd_tolerance, ccd_max_iterations), tiled_to_single_(tiled_to_single), n_single_dof_(tiled_to_single_.maxCoeff() + 1)
    {
        assert(tiled_to_single_.size() == collision_mesh_.full_num_vertices());

        update_projection();

        // const int dim = collision_mesh_.dim();
        // random_coeffs.setZero(proj.cols());
        // for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        // {
        //     const int i_full = collision_mesh_.to_full_vertex_id(i);
        //     // random_coeffs.segment(i_full * dim, dim).setRandom();
        //     if (i == 0)
        //         random_coeffs(i_full * dim + 1) = 1;
        // }

        // const Eigen::MatrixXd displaced = collision_mesh_.displace_vertices(
        //     Eigen::MatrixXd::Zero(collision_mesh_.full_num_vertices(), collision_mesh_.dim()));

        // io::OBJWriter::write(
        //     "tiled.obj", displaced,
        //     collision_mesh_.edges(), collision_mesh_.faces());
    }

    void PeriodicContactForm::update_projection() const
    {
        const int dim = collision_mesh_.dim();
        const auto &boundary_vertices = collision_mesh_.rest_positions();

        std::vector<Eigen::Triplet<double>> entries;
        for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        {
            const int i_full = collision_mesh_.to_full_vertex_id(i);
            for (int d = 0; d < dim; d++)
                entries.emplace_back(tiled_to_single_(i_full) * dim + d, i_full * dim + d, 1);
            
            for (int p = 0; p < dim; p++)
                for (int q = 0; q < dim; q++)
                    entries.emplace_back(n_single_dof_ * dim + p * dim + q, i_full * dim + p, boundary_vertices(i, q));
        }

        proj.resize(n_single_dof_ * dim + dim * dim, tiled_to_single_.size() * dim);
        proj.setZero();
        proj.setFromTriplets(entries.begin(), entries.end());
    }

    Eigen::VectorXd PeriodicContactForm::single_to_tiled(const Eigen::VectorXd &x) const
    {
        const int dim = collision_mesh_.dim();
        const auto &boundary_vertices = collision_mesh_.rest_positions();
        assert(x.size() == n_single_dof_ * dim + dim * dim);

        Eigen::VectorXd tiled_x;
        tiled_x.setZero(tiled_to_single_.size() * dim);
        for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        {
            const int i_full = collision_mesh_.to_full_vertex_id(i);
            for (int d = 0; d < dim; d++)
                tiled_x(i_full * dim + d) += x(tiled_to_single_(i_full) * dim + d);
            
            for (int p = 0; p < dim; p++)
                for (int q = 0; q < dim; q++)
                    tiled_x(i_full * dim + p) += x(n_single_dof_ * dim + p * dim + q) * boundary_vertices(i, q);
        }

        return tiled_x;
    }
    Eigen::VectorXd PeriodicContactForm::tiled_to_single_grad(const Eigen::VectorXd &grad) const
    {
        const int dim = collision_mesh_.dim();
        const auto &boundary_vertices = collision_mesh_.rest_positions();
        assert(grad.size() == tiled_to_single_.size() * dim);

        Eigen::VectorXd reduced_grad;
        reduced_grad.setZero(n_single_dof_ * dim + dim * dim);
        for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        {
            const int i_full = collision_mesh_.to_full_vertex_id(i);
            for (int d = 0; d < dim; d++)
                reduced_grad(tiled_to_single_(i_full) * dim + d) += grad(i_full * dim + d);
            
            for (int p = 0; p < dim; p++)
                for (int q = 0; q < dim; q++)
                    reduced_grad(n_single_dof_ * dim + p * dim + q) += boundary_vertices(i, q) * grad(i_full * dim + p);
        }

        return reduced_grad;
    }

    void PeriodicContactForm::init(const Eigen::VectorXd &x)
    {
        ContactForm::init(single_to_tiled(x));
    }

    void PeriodicContactForm::force_periodic_shape_derivative(const State& state, const ipc::CollisionConstraints &contact_set, const Eigen::VectorXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term)
    {
        const int dim = collision_mesh_.dim();
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(single_to_tiled(solution));

        Eigen::VectorXd tiled_term;

        {
		    StiffnessMatrix dq_h = collision_mesh_.to_full_dof(contact_set.compute_shape_derivative(collision_mesh_, displaced_surface, dhat_));
		    update_projection();
            tiled_term = dq_h.transpose() * single_to_tiled(adjoint_sol);
        }

        {
            Eigen::VectorXd force;
            force = contact_set.compute_potential_gradient(collision_mesh_, displaced_surface, dhat_);
            force = collision_mesh_.to_full_dof(force);
            Eigen::MatrixXd adjoint_affine = utils::unflatten(adjoint_sol.tail(dim * dim), dim);
            for (int k = 0; k < collision_mesh_.num_vertices(); k++)
            {
                const int k_full = collision_mesh_.to_full_vertex_id(k);
                tiled_term.segment(k_full * dim, dim) += adjoint_affine.transpose() * force.segment(k_full * dim, dim);
            }
        }

        {
            StiffnessMatrix hessian_full;
            ContactForm::second_derivative_unweighted(single_to_tiled(solution), hessian_full);
            Eigen::VectorXd tmp = single_to_tiled(adjoint_sol).transpose() * hessian_full;
            Eigen::MatrixXd sol_affine = utils::unflatten(solution.tail(dim * dim), dim);
            for (int k = 0; k < collision_mesh_.num_vertices(); k++)
            {
                const int k_full = collision_mesh_.to_full_vertex_id(k);
                tiled_term.segment(k_full * dim, dim) += sol_affine.transpose() * tmp.segment(k_full * dim, dim);
            }
        }

        // chain rule from tiled to periodic
        Eigen::VectorXd scale_term, unit_term;
        scale_term.setZero(dim);
        unit_term.setZero(tiled_term.size());
        Eigen::VectorXd scale = state.periodic_mesh_representation.tail(dim);
        for (int k = 0; k < collision_mesh_.num_vertices(); k++)
        {
            const int k_full = collision_mesh_.to_full_vertex_id(k);
            scale_term.array() += tiled_term.segment(k_full * dim, dim).array() * collision_mesh_.rest_positions().row(k).transpose().array();
            unit_term.segment(k_full * dim, dim).array() += tiled_term.segment(k_full * dim, dim).array() * scale.array();
        }

        Eigen::VectorXd single_term = state.down_sampling_mat * proj.topRows(dim * n_single_dof_) * unit_term;
        single_term = utils::flatten(utils::unflatten(single_term, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));

        term.setZero(state.periodic_mesh_representation.size());
        for (int i = 0; i < state.n_geom_bases; i++)
            term.segment(state.periodic_mesh_map->full_to_periodic(i) * dim, dim).array() += single_term.segment(i * dim, dim).array();
        term.tail(dim) = scale_term;

        term *= weight();

        // harder test
        // Eigen::VectorXd tiled_term = (adjoint_sol.dot(tiled_to_single_grad(random_coeffs.array() * single_to_tiled(solution).array()))) * collision_mesh_.to_full_dof(utils::flatten(Eigen::MatrixXd::Zero(collision_mesh_.rest_positions().rows(), collision_mesh_.rest_positions().cols())));

        // Eigen::VectorXd force = random_coeffs.array() * single_to_tiled(solution).array();
        // Eigen::MatrixXd adjoint_affine = utils::unflatten(adjoint_sol.tail(dim * dim), dim);
        // for (int k = 0; k < collision_mesh_.num_vertices(); k++)
        // {
        //     const int k_full = collision_mesh_.to_full_vertex_id(k);
        //     tiled_term.segment(k_full * dim, dim) += adjoint_affine.transpose() * force.segment(k_full * dim, dim);
        // }

        // Eigen::MatrixXd hess = random_coeffs.asDiagonal();
        // Eigen::VectorXd tmp = single_to_tiled(adjoint_sol).transpose() * hess;
        // for (int k = 0; k < collision_mesh_.num_vertices(); k++)
        // {
        //     const int k_full = collision_mesh_.to_full_vertex_id(k);
        //     for (int d = 0; d < dim; d++)
        //         for (int a = 0; a < dim; a++)
        //             tiled_term(k_full * dim + d) += tmp(k_full * dim + a) * solution(solution.size() - dim * dim + a * dim + d);
        // }

        // // auto grad
        // // Eigen::VectorXd tiled_term = collision_mesh_.to_full_dof(adjoint_sol.transpose() * vector_func_grad(utils::flatten(collision_mesh_.rest_positions()), solution, tiled_to_single_, collision_mesh_));

        // Eigen::VectorXd scale_term, unit_term;
        // scale_term.setZero(dim);
        // unit_term.setZero(tiled_term.size());
        // Eigen::VectorXd scale = state.periodic_mesh_representation.tail(dim);
        // for (int k = 0; k < collision_mesh_.num_vertices(); k++)
        // {
        //     const int k_full = collision_mesh_.to_full_vertex_id(k);
        //     scale_term.array() += tiled_term.segment(k_full * dim, dim).array() * collision_mesh_.rest_positions().row(k).transpose().array();
        //     unit_term.segment(k_full * dim, dim).array() += tiled_term.segment(k_full * dim, dim).array() * scale.array();
        // }

        // Eigen::VectorXd single_term = state.down_sampling_mat * proj.topRows(dim * n_single_dof_) * unit_term;
        // single_term = utils::flatten(utils::unflatten(single_term, state.mesh->dimension())(state.primitive_to_node(), Eigen::all));

        // term.setZero(state.periodic_mesh_representation.size());
        // for (int i = 0; i < state.n_geom_bases; i++)
        //     term.segment(state.periodic_mesh_map->full_to_periodic(i) * dim, dim).array() += single_term.segment(i * dim, dim).array();
        // term.tail(dim) = scale_term;

        // term *= weight();
    }

    double PeriodicContactForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        return ContactForm::value_unweighted(single_to_tiled(x));
    }

    void PeriodicContactForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        ContactForm::first_derivative_unweighted(single_to_tiled(x), gradv);
        gradv = tiled_to_single_grad(gradv);

        // gradv = tiled_to_single_grad(random_coeffs.array() * single_to_tiled(x).array());
        // gradv = vector_func<double>(utils::flatten(collision_mesh_.rest_positions()), x, tiled_to_single_, collision_mesh_);
    }

    void PeriodicContactForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        StiffnessMatrix hessian_full;
        ContactForm::second_derivative_unweighted(single_to_tiled(x), hessian_full);
        
        update_projection();
        hessian = proj * hessian_full * proj.transpose();
    }

    void PeriodicContactForm::update_quantities(const double t, const Eigen::VectorXd &x) 
    {
        ContactForm::update_quantities(t, single_to_tiled(x));
    }

    double PeriodicContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const 
    {
        return ContactForm::max_step_size(single_to_tiled(x0), single_to_tiled(x1));
    }

    void PeriodicContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) 
    {
        ContactForm::line_search_begin(single_to_tiled(x0), single_to_tiled(x1));
    }

    void PeriodicContactForm::solution_changed(const Eigen::VectorXd &new_x) 
    {
        ContactForm::solution_changed(single_to_tiled(new_x));
    }

    void PeriodicContactForm::post_step(const int iter_num, const Eigen::VectorXd &x) 
    {
        ContactForm::post_step(iter_num, single_to_tiled(x));
    }

    bool PeriodicContactForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const 
    {
        return ContactForm::is_step_collision_free(single_to_tiled(x0), single_to_tiled(x1));
    }

    void PeriodicContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) 
    {
        ContactForm::update_barrier_stiffness(single_to_tiled(x), single_to_tiled(grad_energy));
    }
}