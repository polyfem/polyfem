#pragma once

#include <Eigen/Core>

namespace polyfem::output
{
	class Evaluator
	{
	private:
		Evaluator() {}

	public:
		/// evaluates the function fun at the vertices on the mesh
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] basis basis function
		/// @param[in] fun function to interpolate
		/// @param[out] result output
		void compute_vertex_values(int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, Eigen::MatrixXd &result);
		/// compute von mises stress at quadrature points for the function fun, also compute the interpolated function
		/// @param[in] fun function to used
		/// @param[out] result output displacement
		/// @param[out] von_mises output von mises
		void compute_stress_at_quadrature_points(const MatrixXd &fun, Eigen::MatrixXd &result, Eigen::VectorXd &von_mises);
		/// interpolate the function fun.
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void interpolate_function(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);
		/// interpolate the function fun.
		/// @param[in] n_points is the size of the output.
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] basis basis function
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void interpolate_function(const int n_points, const int actual_dim, const std::vector<ElementBases> &basis, const MatrixXd &fun, MatrixXd &result, const bool use_sampler, const bool boundary_only);

		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] el_index element index
		/// @param[in] local_pts points in the reference element
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		void interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, MatrixXd &result, MatrixXd &result_grad);
		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] el_index element index
		/// @param[in] local_pts points in the reference element
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		void interpolate_at_local_vals(const int el_index, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad);
		/// interpolate the function fun and its gradient at in element el_index for the local_pts in the reference element using bases bases
		/// interpolate solution and gradient at element (calls interpolate_at_local_vals with sol)
		/// @param[in] el_index element index
		/// @param[in] actual_dim is the size of the problem (e.g., 1 for Laplace, dim for elasticity)
		/// @param[in] bases basis function
		/// @param[in] local_pts points in the reference element
		/// @param[in] fun function to used
		/// @param[out] result output
		/// @param[out] result_grad output gradients
		void interpolate_at_local_vals(const int el_index, const int actual_dim, const std::vector<ElementBases> &bases, const MatrixXd &local_pts, const MatrixXd &fun, MatrixXd &result, MatrixXd &result_grad);

		/// checks if mises are not nan
		/// @param[in] fun function to used
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		/// @return if mises are nan
		bool check_scalar_value(const Eigen::MatrixXd &fun, const bool use_sampler, const bool boundary_only);
		/// computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result scalar value
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void compute_scalar_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);
		/// computes scalar quantity of funtion (ie von mises for elasticity and norm of velocity for fluid)
		/// the scalar value is averaged around every node to make it continuos
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result_scalar scalar value
		/// @param[out] result_tensor tensor value
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void average_grad_based_function(const int n_points, const MatrixXd &fun, MatrixXd &result_scalar, MatrixXd &result_tensor, const bool use_sampler, const bool boundary_only);
		/// compute tensor quantity (ie stress tensor or velocy)
		/// @param[in] n_points is the size of the output.
		/// @param[in] fun function to used
		/// @param[out] result resulting tensor
		/// @param[in] use_sampler uses the sampler or not
		/// @param[in] boundary_only interpolates only at boundary elements
		void compute_tensor_value(const int n_points, const Eigen::MatrixXd &fun, Eigen::MatrixXd &result, const bool use_sampler, const bool boundary_only);

		/// computes integrated solution (fun) per surface face. pts and faces are the boundary are the boundary on the rest configuration
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		void interpolate_boundary_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result);
		/// computes integrated solution (fun) per surface face vertex. pts and faces are the boundary are the boundary on the rest configuration
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[out] result resulting value
		void interpolate_boundary_function_at_vertices(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, MatrixXd &result);
		/// computes traction foces for fun (tensor * surface normal) result, stress tensor, and von mises, per surface face. pts and faces are the boundary on the rest configuration.
		/// disp is the displacement of the surface vertices
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] disp displacement to deform mesh
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		/// @param[out] stresses resulting stresses
		/// @param[out] mises resulting mises
		/// @param[in] skip_orientation skip reorientation of surface
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const MatrixXd &disp, const bool compute_avg, MatrixXd &result, MatrixXd &stresses, MatrixXd &mises, const bool skip_orientation = false);
		/// same as interpolate_boundary_tensor_function with disp=0
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[in] fun function to used
		/// @param[in] compute_avg if compute the average across elements
		/// @param[out] result resulting value
		/// @param[out] stresses resulting stresses
		/// @param[out] mises resulting mises
		/// @param[in] skip_orientation skip reorientation of surface
		void interpolate_boundary_tensor_function(const MatrixXd &pts, const MatrixXi &faces, const MatrixXd &fun, const bool compute_avg, MatrixXd &result, MatrixXd &stresses, MatrixXd &mises, const bool skip_orientation = false);

		/// returns a triangulated representation of the sideset. sidesets contains integers mapping to faces
		/// @param[in] pts boundary points
		/// @param[in] faces boundary faces
		/// @param[out] sidesets resulting sidesets
		void get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets);

		/// samples to solution on the visualization mesh and return the vis mesh (points and tets) and the interpolated values (fun)
		void get_sampled_solution(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd discr;
			// Eigen::MatrixXi el_id;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// interpolate_function(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// samples to stess tensor on the visualization mesh and return them (fun)
		void get_stresses(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd points;
			// Eigen::MatrixXi tets;
			// Eigen::MatrixXi el_id;
			// Eigen::MatrixXd discr;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// compute_tensor_value(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// samples to von mises stesses on the visualization mesh and return them (fun)
		void get_sampled_mises(Eigen::MatrixXd &fun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd points;
			// Eigen::MatrixXi tets;
			// Eigen::MatrixXi el_id;
			// Eigen::MatrixXd discr;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// compute_scalar_value(points.rows(), sol, fun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}

		/// samples to averaged von mises stesses on the visualization mesh and return them (fun)
		void get_sampled_mises_avg(Eigen::MatrixXd &fun, Eigen::MatrixXd &tfun, bool boundary_only = false)
		{
			// TODO: fix me TESEO
			// Eigen::MatrixXd points;
			// Eigen::MatrixXi tets;
			// Eigen::MatrixXi el_id;
			// Eigen::MatrixXd discr;
			// const bool tmp = args["export"]["vis_boundary_only"];
			// args["export"]["vis_boundary_only"] = boundary_only;

			// build_vis_mesh(points, tets, el_id, discr);
			// average_grad_based_function(points.rows(), sol, fun, tfun, false, boundary_only);

			// args["export"]["vis_boundary_only"] = tmp;
		}
	}
} // namespace polyfem::output