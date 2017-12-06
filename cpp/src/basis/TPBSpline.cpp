#include "TPBSpline.hpp"

#include "BSpline.hpp"



namespace poly_fem {
	TensorIndex::TensorIndex()
	{
		dims_[0]=dims_[1]=dims_[2]=0;
	}


	TensorIndex::TensorIndex(const int dim1, const int dim2, const int dim3)
	{
		init(dim1, dim2, dim3);
	}


	void TensorIndex::init(const int dim1, const int dim2, const int dim3)
	{
		dims_[0]=dim1;
		dims_[1]=dim2;
		dims_[2]=dim3;
	}


	int TensorIndex::index_for(const int i, const int j, const int k) const
	{
		assert(dims_[0] > 0);
		assert(dims_[1] > 0);
		assert(dims_[2] > 0);

		assert(i>=0 && i<dims_[0]);
		assert(j>=0 && j<dims_[1]);
		assert(k>=0 && k<dims_[2]);

		return (i*dims_[1]+j)*dims_[2]+k;
	}

	int TensorIndex::size() const
	{
		assert(dims_[0] > 0);
		assert(dims_[1] > 0);
		assert(dims_[2] > 0);

		return dims_[0]*dims_[1]*dims_[2];
	}

	int TensorIndex::operator[](const int i) const
	{
		assert(i>=0 && i < 3);
		assert(dims_[0] > 0);
		assert(dims_[1] > 0);
		assert(dims_[2] > 0);


		return dims_[i];
	}



	void TensorProductBSpline::init(const std::vector<double> &knots_u, const std::vector<double> &knots_v, const std::vector<double> &control_points, const int n_control_u, const int n_control_v, const int dim)
	{
		init(int(knots_u.size())-n_control_u-1, int(knots_v.size())-n_control_v-1, knots_u, knots_v, control_points, n_control_u, n_control_v, dim);
	}

	void TensorProductBSpline::init(const int degree_u, const int degree_v, const std::vector<double> &knots_u, const std::vector<double> &knots_v, const std::vector<double> &control_points, const int n_control_u, const int n_control_v, const int dim)
	{
		degree_u_ = degree_u;
		degree_v_ = degree_v;

		tensor_index_.init(n_control_u, n_control_v, dim);

		knots_u_ = knots_u;
		knots_v_ = knots_v;

		control_points_ = control_points;

		assert(degree_u_>0);
		assert(degree_v_>0);
		assert(degree_u_ == int(knots_u_.size()) - tensor_index_[0] - 1);
		assert(degree_v_ == int(knots_v_.size()) - tensor_index_[1] - 1);
	}

	void TensorProductBSpline::interpolate(const Eigen::MatrixXd &ts, Eigen::MatrixXd &result) const
	{
		const int n_t = int(ts.rows());
		assert(ts.cols() == 2);

		result.resize(n_t, dim());

		std::vector<double> temp;

		for(int i = 0; i < n_t; ++i)
		{
			interpolate(ts(i,0), ts(i,1), temp);

			for(int j = 0; j < dim(); ++j)
				result(i, j) = temp[j];
		}
	}

	void TensorProductBSpline::interpolate(const double u, const double v, std::vector<double> &result) const
	{
		BSpline bspline;

		std::vector<double> tmp;
		std::vector<double> tmp_control_u(dim() * tensor_index_[0]);
		std::vector<double> tmp_control_v(dim() * tensor_index_[1]);


		for(int i = 0; i < tensor_index_[0]; ++i)
		{
			for(int j = 0; j < tensor_index_[1]; ++j){
				for(int k=0; k < dim(); ++k)
					tmp_control_v[j*dim()+k] = ctrl_pts_at(i, j, k);
			}

			bspline.init(degree_v_, knots_v_, tmp_control_v);
			bspline.interpolate(v, tmp);

			for(int k=0; k < dim(); ++k)
				tmp_control_u[i * dim() + k] = tmp[k];
		}

		bspline.init(degree_u_, knots_u_, tmp_control_u);
		bspline.interpolate(u, result);
	}


	void TensorProductBSpline::derivative(TensorProductBSpline &dx, TensorProductBSpline &dy) const
	{
		std::vector<double> knot_dx(knots_u_.size()-2);
		for(std::size_t i = 1; i < knots_u_.size()-1; ++i)
			knot_dx[i-1] = knots_u_[i];

		const TensorIndex ti_dx(tensor_index_[0]-1, tensor_index_[1], dim());

		std::vector<double> ctrl_dx( ti_dx.size() );

		for(int j = 0; j < ti_dx[1]; ++j)
		{
			for(int i = 0; i < ti_dx[0]; ++i)
			{
				for(int k=0; k < dim(); ++k)
					ctrl_dx[ti_dx.index_for(i,j,k)]=degree_u_/(knots_u_[i+degree_u_+1]-knots_u_[i+1]) * (ctrl_pts_at(i+1, j, k) - ctrl_pts_at(i, j, k));
			}
		}
		dx.init(degree_u_ - 1, degree_v_, knot_dx, knots_v_, ctrl_dx, ti_dx[0], ti_dx[1], ti_dx[2]);




		std::vector<double> knot_dy(knots_v_.size()-2);
		for(std::size_t i = 1; i < knots_v_.size()-1; ++i)
			knot_dy[i-1] = knots_v_[i];

		const TensorIndex ti_dy(tensor_index_[0], tensor_index_[1] - 1, dim());

		std::vector<double> ctrl_dy(ti_dy.size());
		for(int i = 0; i < ti_dy[0]; ++i)
		{
			for(int j = 0; j < ti_dy[1]; ++j)
			{
				for(int k = 0; k < dim(); ++k)
					ctrl_dy[ti_dy.index_for(i,j,k)]=degree_v_/(knots_v_[j+degree_v_+1]-knots_v_[j+1]) * (ctrl_pts_at(i, j+1, k) - ctrl_pts_at(i, j, k));
			}
		}
		dy.init(degree_u_, degree_v_ - 1, knots_u_, knot_dy, ctrl_dy, ti_dy[0], ti_dy[1], ti_dy[2]);
	}
}
