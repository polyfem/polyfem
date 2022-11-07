#include "Objective.hpp"

using namespace polyfem::utils;

namespace polyfem::solver
{
    Eigen::VectorXd Objective::compute_adjoint_term(const State& state, const Parameter &param)
    {
        Eigen::VectorXd term;
        term.setZero(param.full_dim());

        if (param.contains_state(state))
        {
            assert(state.adjoint_solved);
            AdjointForm::compute_adjoint_term(state, param.name(), term);
        }
        
        return term;
    }

    StressObjective::StressObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const std::shared_ptr<const ElasticParameter> &elastic_param, const json &args, bool has_integral_sqrt): state_(state), shape_param_(shape_param), elastic_param_(elastic_param)
    {
        formulation_ = state.formulation();
        in_power_ = args["power"];
        out_sqrt_ = has_integral_sqrt;
        auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
        interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

        j_.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
            val.setZero(grad_u.rows(), 1);
            Eigen::MatrixXd grad_u_q, stress;
            for (int q = 0; q < grad_u.rows(); q++)
            {
                if (this->formulation_ == "Laplacian")
                {
                    stress = grad_u.row(q);
                }
                else if (this->formulation_ == "LinearElasticity")
                {
                    vector2matrix(grad_u.row(q), grad_u_q);
                    stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
                }
                else if (this->formulation_ == "NeoHookean")
                {
                    vector2matrix(grad_u.row(q), grad_u_q);
                    Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
                    Eigen::MatrixXd FmT = def_grad.inverse().transpose();
                    stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
                }
                else
                    log_and_throw_error("Unknown formulation!");
                val(q) = pow(stress.squaredNorm(), this->in_power_ / 2.);
            }
        });

        j_.set_dj_dgradu([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
            val.setZero(grad_u.rows(), grad_u.cols());
            const int dim = sqrt(grad_u.cols());
            const int actual_dim = (this->formulation_ == "Laplacian") ? 1 : dim;
            Eigen::MatrixXd grad_u_q, stress, stress_dstress;
            for (int q = 0; q < grad_u.rows(); q++)
            {
                if (this->formulation_ == "Laplacian")
                {
                    stress = grad_u.row(q);
                    stress_dstress = 2 * stress;
                }
                else if (this->formulation_ == "LinearElasticity")
                {
                    vector2matrix(grad_u.row(q), grad_u_q);
                    stress = mu(q) * (grad_u_q + grad_u_q.transpose()) + lambda(q) * grad_u_q.trace() * Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols());
                    stress_dstress = mu(q) * (stress + stress.transpose()) + lambda(q) * stress.trace() * Eigen::MatrixXd::Identity(stress.rows(), stress.cols());
                }
                else if (this->formulation_ == "NeoHookean")
                {
                    vector2matrix(grad_u.row(q), grad_u_q);
                    Eigen::MatrixXd def_grad = Eigen::MatrixXd::Identity(grad_u_q.rows(), grad_u_q.cols()) + grad_u_q;
                    Eigen::MatrixXd FmT = def_grad.inverse().transpose();
                    stress = mu(q) * (def_grad - FmT) + lambda(q) * std::log(def_grad.determinant()) * FmT;
                    stress_dstress = mu(q) * stress + FmT * stress.transpose() * FmT * (mu(q) - lambda(q) * std::log(def_grad.determinant())) + (lambda(q) * (FmT.array() * stress.array()).sum()) * FmT;
                }
                else
                    logger().error("Unknown formulation!");

                const double coef = this->in_power_ * pow(stress.squaredNorm(), this->in_power_ / 2. - 1.);
                for (int i = 0; i < actual_dim; i++)
                    for (int l = 0; l < dim; l++)
                        val(q, i * dim + l) = coef * stress_dstress(i, l);
            }
        });
    }

    double StressObjective::value() const
    {
        assert(time_step_ < state_.diff_cached.size());
        double val = AdjointForm::integrate_objective(state_, j_, state_.diff_cached[time_step_].u, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, time_step_);
        if (out_sqrt_)
            return pow(val, 1. / in_power_);
        else
            return val;
    }

    Eigen::VectorXd StressObjective::compute_adjoint_rhs_step(const State& state) const
    {
        if (&state != &state_)
            return Eigen::VectorXd::Zero(state.ndof());
        
        assert(time_step_ < state_.diff_cached.size());
        
        Eigen::VectorXd rhs;
        AdjointForm::dJ_du_step(state, j_, state.diff_cached[time_step_].u, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, time_step_, rhs);

        if (out_sqrt_)
        {
            double val = AdjointForm::integrate_objective(state_, j_, state_.diff_cached[time_step_].u, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, time_step_);
            if (std::abs(val) < 1e-12)
                logger().warn("stress integral too small, may result in NAN grad!");
            return (pow(val, 1. / in_power_ - 1) / in_power_) * rhs;
        }
        else
            return rhs;
    }

    Eigen::VectorXd StressObjective::compute_partial_gradient(const Parameter &param) const
    {
        Eigen::VectorXd term;
        term.setZero(param.full_dim());
        if (&param == elastic_param_.get())
        {
            // TODO: differentiate stress wrt. lame param
            log_and_throw_error("Not implemented!");
        }
        else if (&param == shape_param_.get())
        {
            assert(time_step_ < state_.diff_cached.size());
            AdjointForm::compute_shape_derivative_functional_term(state_, state_.diff_cached[time_step_].u, j_, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, term, time_step_);
        }
        
        if (out_sqrt_)
        {
            double val = AdjointForm::integrate_objective(state_, j_, state_.diff_cached[time_step_].u, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, time_step_);
            if (std::abs(val) < 1e-12)
                logger().warn("stress integral too small, may result in NAN grad!");
            return (pow(val, 1. / in_power_ - 1) / in_power_) * term;
        }
        else
            return term;
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

    Eigen::MatrixXd SumObjective::compute_adjoint_rhs(const State& state) const
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
        grad.setZero(param.full_dim());
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

    Eigen::MatrixXd BoundarySmoothingObjective::compute_adjoint_rhs(const State& state) const
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
					grad(b * dim + d) += (s(d) * valence - s.squaredNorm() * sum_normalized(d)) * power * pow(s.norm(), power - 2.) / sum_norm;
				}

				for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(adj, b); it; ++it)
				{
					for (int d = 0; d < dim; d++)
					{
						grad(it.col() * dim + d) -= (s(d) + s.squaredNorm() * (V(it.col(), d) - V(b, d)) / (V.row(b) - V.row(it.col())).norm()) * power * pow(s.norm(), power - 2.) / sum_norm;
					}
				}
			}
            return grad;
        }
        else
            return 2 * (L.transpose() * (L * V));
    }

    VolumeObjective::VolumeObjective(const std::shared_ptr<const ShapeParameter> shape_param, const json &args): shape_param_(shape_param)
    {
        auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
        interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
    }

    double VolumeObjective::value() const
    {
		IntegrableFunctional j;
		j.set_j([](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setOnes(u.rows(), 1);
		});

        const State &state = shape_param_->get_state();

        return AdjointForm::integrate_objective(state, j, Eigen::MatrixXd::Zero(state.ndof(), 1), interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, 0);
    }

    Eigen::MatrixXd VolumeObjective::compute_adjoint_rhs(const State& state) const
    {
        return Eigen::VectorXd::Zero(state.ndof()); // Important: it's state, not state_
    }

    Eigen::VectorXd VolumeObjective::compute_partial_gradient(const Parameter &param) const
    {
        if (&param == shape_param_.get())
        {
            IntegrableFunctional j;
            j.set_j([](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
                val.setOnes(u.rows(), 1);
            });

            const State &state = shape_param_->get_state();
            Eigen::VectorXd term;
            AdjointForm::compute_shape_derivative_functional_term(state, Eigen::MatrixXd::Zero(state.ndof(), 1), j, interested_ids_, AdjointForm::SpatialIntegralType::VOLUME, term, 0);
            return term;
        }
        else
            return Eigen::VectorXd::Zero(param.full_dim());
    }

    PositionObjective::PositionObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args): state_(state), shape_param_(shape_param)
    {
        auto tmp_ids = args["volume_selection"].get<std::vector<int>>();
        interested_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());
    }
    
    double PositionObjective::value() const
    {
		IntegrableFunctional j;
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = u.col(this->dim_) + pts.col(this->dim_);
		});

        assert(time_step_ < state_.diff_cached.size());
        return AdjointForm::integrate_objective(state_, j, state_.diff_cached[time_step_].u, interested_ids_, integral_type_, time_step_);
    }

    Eigen::VectorXd PositionObjective::compute_adjoint_rhs_step(const State& state) const
    {
        if (&state != &state_)
            return Eigen::VectorXd::Zero(state.ndof());

		IntegrableFunctional j;
		j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val = u.col(this->dim_) + pts.col(this->dim_);
		});

		j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(u.rows(), u.cols());
			val.col(this->dim_).setOnes();
		});

		j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
			val.setZero(pts.rows(), pts.cols());
			val.col(this->dim_).setOnes();
		});

        Eigen::VectorXd term;
        assert(time_step_ < state.diff_cached.size());
        AdjointForm::dJ_du_step(state, j, state.diff_cached[time_step_].u, interested_ids_, integral_type_, time_step_, term);

        return term;
    }

    Eigen::MatrixXd StaticObjective::compute_adjoint_rhs(const State& state) const
    {
        Eigen::MatrixXd term(state.ndof(), state.diff_cached.size());
        term.col(time_step_) = compute_adjoint_rhs_step(state);

        return term;
    }

    Eigen::VectorXd PositionObjective::compute_partial_gradient(const Parameter &param) const
    {
        if (&param == shape_param_.get())
        {
            IntegrableFunctional j;
            j.set_j([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
                val = u.col(this->dim_) + pts.col(this->dim_);
            });

            j.set_dj_du([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
                val.setZero(u.rows(), u.cols());
                val.col(this->dim_).setOnes();
            });

            j.set_dj_dx([this](const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &pts, const Eigen::MatrixXd &u, const Eigen::MatrixXd &grad_u, const Eigen::MatrixXd &lambda, const Eigen::MatrixXd &mu, const json &params, Eigen::MatrixXd &val) {
                val.setZero(pts.rows(), pts.cols());
                val.col(this->dim_).setOnes();
            });

            Eigen::VectorXd term;
            assert(time_step_ < state_.diff_cached.size());
            AdjointForm::compute_shape_derivative_functional_term(state_, state_.diff_cached[time_step_].u, j, interested_ids_, integral_type_, term, time_step_);
            return term;
        }
        else
            return Eigen::VectorXd::Zero(param.full_dim());
    }

    BarycenterTargetObjective::BarycenterTargetObjective(const State &state, const std::shared_ptr<const ShapeParameter> shape_param, const json &args, const Eigen::MatrixXd &target)
    {
        dim_ = state.mesh->dimension();
        target_ = target;

        objv = std::make_shared<VolumeObjective>(shape_param, args);
        objp.resize(dim_);
        for (int d = 0; d < dim_; d++)
        {
            objp[d] = std::make_shared<PositionObjective>(state, shape_param, args);
            objp[d]->set_dim(d);
        }
    }
    
    Eigen::VectorXd BarycenterTargetObjective::get_target() const
    {
        assert(target_.cols() == dim_);
        if (target_.rows() > 1)
            return target_.row(get_time_step());
        else
            return target_;
    }

    void BarycenterTargetObjective::set_time_step(int time_step)
    {
        StaticObjective::set_time_step(time_step);
        for (auto &obj : objp)
            obj->set_time_step(time_step);
    }

    double BarycenterTargetObjective::value() const
    {
        return (get_barycenter() - get_target()).squaredNorm();
    }
    Eigen::VectorXd BarycenterTargetObjective::compute_partial_gradient(const Parameter &param) const
    {
        Eigen::VectorXd term;
        term.setZero(param.full_dim());

        Eigen::VectorXd target = get_target();
        
        const double volume = objv->value();
        Eigen::VectorXd center(dim_);
        for (int d = 0; d < dim_; d++)
            center(d) = objp[d]->value() / volume;

        double coeffv = 0;
        for (int d = 0; d < dim_; d++)
            coeffv += 2 * (center(d) - target(d)) * (-center(d) / volume);
        
        term += coeffv * objv->compute_partial_gradient(param);

        for (int d = 0; d < dim_; d++)
            term += (2.0 / volume * (center(d) - target(d))) * objp[d]->compute_partial_gradient(param);
        
        return term;
    }
    Eigen::VectorXd BarycenterTargetObjective::compute_adjoint_rhs_step(const State& state) const
    {
        Eigen::VectorXd term;
        term.setZero(state.ndof());

        Eigen::VectorXd target = get_target();
        
        const double volume = objv->value();
        Eigen::VectorXd center(dim_);
        for (int d = 0; d < dim_; d++)
            center(d) = objp[d]->value() / volume;

        for (int d = 0; d < dim_; d++)
            term += (2.0 / volume * (center(d) - target(d))) * objp[d]->compute_adjoint_rhs_step(state);
        
        return term;
    }

    Eigen::VectorXd BarycenterTargetObjective::get_barycenter() const
    {
        const double volume = objv->value();
        Eigen::VectorXd center(dim_);
        for (int d = 0; d < dim_; d++)
            center(d) = objp[d]->value() / volume;

        return center;
    }

    TransientObjective::TransientObjective(const int time_steps, const double dt, const std::string &transient_integral_type, const std::shared_ptr<StaticObjective> &obj)
    {
        time_steps_ = time_steps;
        dt_ = dt;
        transient_integral_type_ = transient_integral_type;
        obj_ = obj;
    }

    std::vector<double> TransientObjective::get_transient_quadrature_weights() const
    {
        std::vector<double> weights;
        weights.assign(time_steps_ + 1, dt_);
        if (transient_integral_type_ == "uniform")
        {
            weights[0] = 0;
        }
        else if (transient_integral_type_ == "trapezoidal")
        {
            weights[0] = dt_ / 2.;
            weights[weights.size() - 1] = dt_ / 2.;
        }
        else if (transient_integral_type_ == "simpson")
        {
            weights[0] = dt_ / 3.;
            weights[weights.size() - 1] = dt_ / 3.;
            for (int i = 1; i < weights.size() - 1; i++)
            {
                if (i % 2)
                    weights[i] = dt_ * 4. / 3.;
                else
                    weights[i] = dt_ * 2. / 4.;
            }
        }
        else if (transient_integral_type_ == "final")
        {
            weights.assign(time_steps_ + 1, 0);
            weights[time_steps_] = 1;
        }
        else if (transient_integral_type_.find("step_") != std::string::npos)
        {
            weights.assign(time_steps_ + 1, 0);
            int step = std::stoi(transient_integral_type_.substr(5));
            assert(step > 0 && step < weights.size());
            weights[step] = 1;
        }
        else
            assert(false);

        return weights;
    }

    double TransientObjective::value() const
    {
        double value = 0;
		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
            obj_->set_time_step(i);
            value += weights[i] * obj_->value();
		}
        return value;
    }

    Eigen::MatrixXd TransientObjective::compute_adjoint_rhs(const State& state) const
    {
        Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), time_steps_ + 1);

		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
            obj_->set_time_step(i);
            terms.col(i) = weights[i] * obj_->compute_adjoint_rhs_step(state);
		}

        return terms;
    }

    Eigen::VectorXd TransientObjective::compute_partial_gradient(const Parameter &param) const
    {
        Eigen::VectorXd term;
        term.setZero(param.full_dim());

        std::vector<double> weights = get_transient_quadrature_weights();
        for (int i = 0; i <= time_steps_; i++)
        {
            obj_->set_time_step(i);
            term += weights[i] * obj_->compute_partial_gradient(param);
        }

        return term;
    }

}