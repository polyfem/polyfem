#include "Objective.hpp"
#include "AdjointForm.hpp"

namespace polyfem::solver
{
    Objective::Objective(const json &args)
    {
        set_weight(args["weight"]);
    }

    Eigen::VectorXd Objective::compute_adjoint_term(const State& state, const Parameter &param)
    {
        // TODO: call AdjointForm function
        return Eigen::VectorXd();
    }

    StressObjective::StressObjective(const State &state, const std::shared_ptr<const ElasticParameter> &elastic_param, const json &args): state_(state), elastic_param_(elastic_param), Objective(args)
    {
        formulation_ = state.formulation();
        power_ = args["power"];
        transient_integral_type_ = args["transient_integral_type"];
        const auto &interested_bodies = args["volume_selection"].get<std::vector<int>>();
        interested_ids_ = std::set(interested_bodies.begin(), interested_bodies.end());

        j.set_j([formulation_, power_](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
            val.setZero(grad_u.rows(), 1);
            for (int q = 0; q < grad_u.rows(); q++)
            {
                Eigen::MatrixXd grad_u_q, stress;
                vector2matrix(grad_u.row(q), grad_u_q);
                if (formulation_ == "LinearElasticity")
                {
                    stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
                }
                else if (formulation_ == "NeoHookean")
                {
                    Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
                    Eigen::MatrixXd FmT = def_grad.inverse().transpose();
                    stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
                }
                else
                    logger().error("Unknown formulation!");
                val(q) = pow(stress.squaredNorm(), power_ / 2.);
            }
        });

        j.set_dj_dgradu([formulation_, power_](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
            val.setZero(grad_u.rows(), grad_u.cols());
            const int dim = sqrt(grad_u.cols());
            for (int q = 0; q < grad_u.rows(); q++)
            {
                Eigen::MatrixXd grad_u_q, stress, stress_dstress;
                vector2matrix(grad_u.row(q), grad_u_q);
                if (formulation_ == "LinearElasticity")
                {
                    stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
                    stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
                }
                else if (formulation_ == "NeoHookean")
                {
                    Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
                    Eigen::MatrixXd FmT = def_grad.inverse().transpose();
                    stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
                    stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
                }
                else
                    logger().error("Unknown formulation!");

                const double coef = power_ * pow(stress.squaredNorm(), power_ / 2. - 1.);
                for (int i = 0; i < dim; i++)
                    for (int l = 0; l < dim; l++)
                        val(q, i * dim + l) = coef * stress_dstress(i, l);
            }
        });
    }

    double StressObjective::value() const
    {
        return AdjointForm::value(state_, j_, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type_);
    }

    Eigen::VectorXd StressObjective::compute_adjoint_rhs(const State& state) const
    {
        if (&state != &state_)
            return Eigen::VectorXd::Zero(state.ndof());
        
        Eigen::VectorXd rhs;
        if (state.problem->is_time_dependent())
        {
            AdjointForm::dJ_du_transient(state, j_, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, transient_integral_type_, rhs);
        }
        else
        {
            dJ_du_step(state, j_, state.diff_cached[0].u, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, 0, rhs);
        }

        return rhs;
    }

    Eigen::VectorXd StressObjective::compute_partial_gradient(const Parameter &param) const
    {
        if (&param == elastic_param_.get())
        {
            // TODO: differentiate stress wrt. lame param
            log_and_throw_error("Not implemented!");
        }
        else
            return Eigen::VectorXd::Zero(/*TODO*/);
    }

    SumObjective::SumObjective(const json &args)
    {
        if (args.is_array())
        {

        }
        else
        {

        }
    }

    Eigen::VectorXd SumObjective::compute_adjoint_rhs(const State& state) const
    {
        Eigen::VectorXd rhs;
        rhs.setZero(state.ndof());
        for (const auto &obj : objs)
        {
            rhs += obj.compute_adjoint_rhs(state);
        }
        return rhs;
    }

    Eigen::VectorXd SumObjective::compute_partial_gradient(const Parameter &param) const
    {
        Eigen::VectorXd grad;
        grad.setZero(/*TODO*/);
        for (const auto &obj : objs)
        {
            grad += obj.compute_partial_gradient(param);
        }
        return grad;
    }

    double SumObjective::value() const
    {
        double val = 0;
        for (const auto &obj : objs)
        {
            val += obj.value();
        }
        return val;
    }

    void BoundarySmoothingObjective::init(const std::shared_ptr<const ShapeParameter> shape_param)
    {
        shape_param_ = shape_param;

        shape_param_->get_full_mesh(V, F);

        const int dim = V.cols();
        const int n_verts = V.rows();

        Eigen::MatrixXi boundary_edges = shape_param_->get_boundary_edges();
        active_mask = shape_param_->get_active_vertex_mask();
        boundary_nodes = shape_param_->get_boundary_nodes();

        adj.setZero();
        adj.resize(n_verts, n_verts);
        std::vector<Eigen::Triplet<bool>> T_adj;
        for (int e = 0; e < boundary_edges.rows(); e++)
        {
            T_adj.emplace_back(boundary_edges(e, 0), boundary_edges(e, 1), true);
            T_adj.emplace_back(boundary_edges(e, 1), boundary_edges(e, 0), true);
        }
        adj.setFromTriplets(T_adj.begin(), T_adj.end());

        std::vector<int> degrees(n_verts, 0);
        for (int k = 0; k < adj.outerSize(); ++k)
            for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
                degrees[k]++;
                
        L.setZero();
        L.resize(n_verts, n_verts);
        if (!args_["scale_invariant"])
        {
            std::vector<Eigen::Triplet<double>> T_L;
            for (int k = 0; k < adj.outerSize(); ++k)
            {
                if (degrees[k] == 0 || !active_mask[k])
                    continue;
                T_L.emplace_back(k, k, degrees[k]);
                for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, k); it; ++it)
                {
                    assert(it.row() == k);
                    T_L.emplace_back(it.row(), it.col(), -1);
                }
            }
            L.setFromTriplets(T_L.begin(), T_L.end());
            L.prune([](int i, int j, double val) { return abs(val) > 1e-12; });
        }
    }

    BoundarySmoothingObjective::BoundarySmoothingObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args): args_(args)
    {
        init(shape_param);
    }

    double BoundarySmoothingObjective::value() const
    {
        const int dim = V.cols();
        const int n_verts = V.rows();
        const int power = args_["power"];

        double val = 0;
        if (args_["scale_invariant"])
        {
			for (int b : boundary_nodes)
			{
				if (!active_mask[b])
					continue;
				polyfem::RowVectorNd s;
				s.setZero(V.cols());
				double sum_norm = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					s += V.row(b) - V.row(it.col());
					sum_norm += (V.row(b) - V.row(it.col())).norm();
				}
				s = s / sum_norm;
				val += pow(s.norm(), power);
			}
        }
        else
            val = (L * V).eval().squaredNorm();

        return val;
    }

    Eigen::VectorXd BoundarySmoothingObjective::compute_adjoint_rhs(const State& state) const
    {
        return Eigen::VectorXd::Zero(state.ndof());
    }

    Eigen::VectorXd BoundarySmoothingObjective::compute_partial_gradient(const Parameter &param) const
    {
        const int dim = V.cols();
        const int n_verts = V.rows();
        const int power = args_["power"];

        if (args_["scale_invariant"])
        {
            Eigen::VectorXd grad;
			grad.setZero(V.size());
			for (int b : boundary_nodes)
			{
				if (!active_mask[b])
					continue;
				polyfem::RowVectorNd s;
				s.setZero(dim);
				double sum_norm = 0;
				auto sum_normalized = s;
				int valence = 0;
				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					assert(it.col() != b);
					auto x = V.row(b) - V.row(it.col());
					s += x;
					sum_norm += x.norm();
					sum_normalized += x.normalized();
					valence += 1;
				}
				s = s / sum_norm;

				for (int d = 0; d < dim; d++)
				{
					grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * p * pow(s.norm(), p - 2.) / sum_norm;
				}

				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					for (int d = 0; d < dim; d++)
					{
						grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (V(it.col(), d) - V(b, d)) / (V.row(b) - V.row(it.col())).norm()) * power * pow(s.norm(), p - 2.) / sum_norm;
					}
				}
			}
            return grad;
        }
        else
            return 2 * (L.transpose() * (L * V));
    }
}