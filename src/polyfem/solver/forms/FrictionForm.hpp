#pragma once

#include <polyfem/utils/Types.hpp>

#include <ipc/collision_mesh.hpp>

namespace polyfem
{
	namespace solver
	{
		class FrictionForm
		{
		public:
			FrictionForm(const double epsv, const double mu, const ipc::CollisionMesh &collision_mesh);

			virtual void init(const Eigen::VectorXd &displacement) {}

			virtual double value(const Eigen::VectorXd &x) = 0;
			virtual void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) = 0;
			virtual void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) { hessian.resize(0, 0); }

			virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return true; }
			virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return 1; }

			virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}
			virtual void line_search_end() {}

			virtual void post_step(const int iter_num, const Eigen::VectorXd &x) {}

			virtual void solution_changed(const Eigen::VectorXd &newX){};

			virtual void update_quantities(const double t, const Eigen::VectorXd &x) = 0;

			//more than one step?
			virtual void init_lagging(const Eigen::VectorXd &x){};
			virtual void update_lagging(const Eigen::VectorXd &x){};

			void set_project_to_psd(bool val) { project_to_psd_ = val; }
			bool is_project_to_psd() const { return project_to_psd_; }

		private:
			const double epsv_;
			const double mu_;
			const ipc::CollisionMesh &collision_mesh_;
			ipc::FrictionConstraints friction_constraint_set_;
		};
	} // namespace solver
} // namespace polyfem
