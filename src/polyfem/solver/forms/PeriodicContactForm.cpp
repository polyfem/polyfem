#include "PeriodicContactForm.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/io/OBJWriter.hpp>

namespace polyfem::solver
{
	PeriodicContactForm::PeriodicContactForm(const ipc::CollisionMesh &periodic_collision_mesh,
                        const Eigen::VectorXi &tiled_to_periodic,
                        const double dhat,
                        const double avg_mass,
                        const bool use_convergent_formulation,
                        const bool use_adaptive_barrier_stiffness,
                        const bool is_time_dependent,
                        const ipc::BroadPhaseMethod broad_phase_method,
                        const double ccd_tolerance,
                        const int ccd_max_iterations) : ContactForm(periodic_collision_mesh, dhat, avg_mass, use_convergent_formulation, use_adaptive_barrier_stiffness, is_time_dependent, broad_phase_method, ccd_tolerance, ccd_max_iterations), tiled_to_periodic_(tiled_to_periodic)
    {
        assert(tiled_to_periodic_.size() == collision_mesh_.full_num_vertices());

        const int dim = collision_mesh_.dim();
        const auto &boundary_vertices = collision_mesh_.vertices_at_rest();
        proj.resize((tiled_to_periodic_.maxCoeff() + 1) * dim + dim * dim, tiled_to_periodic_.size() * dim);

        std::vector<Eigen::Triplet<double>> entries;
        for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        {
            const int i_full = collision_mesh_.to_full_vertex_id(i);
            for (int d = 0; d < dim; d++)
                entries.emplace_back(tiled_to_periodic_(i_full) * dim + d, i_full * dim + d, 1);
            
            for (int p = 0; p < dim; p++)
                for (int q = 0; q < dim; q++)
                    entries.emplace_back((tiled_to_periodic_.maxCoeff() + 1) * dim + p * dim + q, i_full * dim + p, boundary_vertices(i, q));
        }

        proj.setFromTriplets(entries.begin(), entries.end());
    }

    Eigen::VectorXd PeriodicContactForm::periodic_to_full(const Eigen::VectorXd &x) const
    {
        const int dim = collision_mesh_.dim();
        assert(x.size() == (tiled_to_periodic_.maxCoeff() + 1) * dim + dim * dim);

        Eigen::VectorXd full;
        full.setZero(tiled_to_periodic_.size() * dim);

        Eigen::MatrixXd affine;
        affine = utils::unflatten(x.tail(dim * dim), dim);

        const auto &boundary_vertices = collision_mesh_.vertices_at_rest();

        for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        {
            const int i_full = collision_mesh_.to_full_vertex_id(i);
            for (int d = 0; d < dim; d++)
                full(i_full * dim + d) = x(tiled_to_periodic_(i_full) * dim + d);

            full.segment(i_full * dim, dim) += boundary_vertices.row(i) * affine.transpose();
        }
        
        return full;
    }
    Eigen::VectorXd PeriodicContactForm::full_to_periodic_grad(const Eigen::VectorXd &grad) const
    {
        const int dim = collision_mesh_.dim();
        assert(grad.size() == tiled_to_periodic_.size() * dim);

        Eigen::VectorXd reduced;
        reduced.setZero((tiled_to_periodic_.maxCoeff() + 1) * dim + dim * dim);

        const auto &boundary_vertices = collision_mesh_.vertices_at_rest();

        for (int i = 0; i < collision_mesh_.num_vertices(); i++)
        {
            const int i_full = collision_mesh_.to_full_vertex_id(i);
            for (int d = 0; d < dim; d++)
                reduced(tiled_to_periodic_(i_full) * dim + d) += grad(i_full * dim + d);

            reduced.tail(dim * dim) += utils::flatten(grad.segment(i_full * dim, dim) * boundary_vertices.row(i));
        }
        
        return reduced;
    }

    void PeriodicContactForm::init(const Eigen::VectorXd &x)
    {
        ContactForm::init(periodic_to_full(x));
    }

    void PeriodicContactForm::force_shape_derivative(const ipc::Constraints &contact_set, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &adjoint_sol, Eigen::VectorXd &term)
    {
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(periodic_to_full(solution));

		StiffnessMatrix dq_h = collision_mesh_.to_full_dof(ipc::compute_barrier_shape_derivative(collision_mesh_, displaced_surface, contact_set, dhat_));
		term = -barrier_stiffness() * full_to_periodic_grad(dq_h.transpose() * periodic_to_full(adjoint_sol));
    }

    double PeriodicContactForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        return ContactForm::value_unweighted(periodic_to_full(x));
    }

    void PeriodicContactForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        ContactForm::first_derivative_unweighted(periodic_to_full(x), gradv);
        gradv = full_to_periodic_grad(gradv);
    }

    void PeriodicContactForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        StiffnessMatrix hessian_full;
        ContactForm::second_derivative_unweighted(periodic_to_full(x), hessian_full);
        
        hessian = proj * hessian_full * proj.transpose();
    }

    void PeriodicContactForm::update_quantities(const double t, const Eigen::VectorXd &x) 
    {
        ContactForm::update_quantities(t, periodic_to_full(x));
    }

    double PeriodicContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const 
    {
        return ContactForm::max_step_size(periodic_to_full(x0), periodic_to_full(x1));
    }

    void PeriodicContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) 
    {
        ContactForm::line_search_begin(periodic_to_full(x0), periodic_to_full(x1));
    }

    void PeriodicContactForm::solution_changed(const Eigen::VectorXd &new_x) 
    {
        ContactForm::solution_changed(periodic_to_full(new_x));
    }

    void PeriodicContactForm::post_step(const int iter_num, const Eigen::VectorXd &x) 
    {
        ContactForm::post_step(iter_num, periodic_to_full(x));
    }

    bool PeriodicContactForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const 
    {
        return ContactForm::is_step_collision_free(periodic_to_full(x0), periodic_to_full(x1));
    }

    void PeriodicContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) 
    {
        ContactForm::update_barrier_stiffness(periodic_to_full(x), periodic_to_full(grad_energy));
    }
}