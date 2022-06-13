
namespace polyfem
{
	namespace solver
	{
		LaggedRegForm::LaggedRegForm()
		{
			_lagged_damping_weight = is_time_dependent ? 0 : state.args["solver"]["contact"]["lagged_damping_weight"].get<double>();
		}

		double LaggedRegForm::value(const Eigen::VectorXd &x)
		{
			return _lagged_damping_weight * (full - x_lagged).squaredNorm();
		}

		void LaggedRegForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			_lagged_damping_weight *(full - x_lagged);
		}

		void LaggedRegForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			THessian lagged_damping_hessian = _lagged_damping_weight * sparse_identity(full.size(), full.size());
		}

		void LaggedRegForm::update_lagging(const Eigen::VectorXd &x)
		{
			x_lagged = x;
		};

	} // namespace solver
} // namespace polyfem