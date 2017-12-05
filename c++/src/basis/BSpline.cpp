#include "BSpline.hpp"

#include <cmath>

namespace poly_fem {
	void BSpline::init(const std::vector<double> &knots, const std::vector<double> &control_points, int dim)
	{
		init(int(knots.size())-int(control_points.size())/dim-1, knots, control_points, dim);
	}

	void BSpline::init(const int degree, const std::vector<double> &knots, const std::vector<double> &control_points, int dim)
	{
		assert(dim > 0);

		degree_ = degree;
		knots_ = knots;
		control_points_ = control_points;
		dim_ = dim;

		assert(degree_>0);
		assert(degree_==int(knots_.size()-control_points_.size()/dim)-1);
	}

	void BSpline::interpolate(const std::vector<double> &ts, std::vector<double> &result) const
	{
		result.resize(dim_ * ts.size());
		std::vector<double> temp(dim_);

		for(std::size_t i = 0; i < ts.size(); ++i) {
			interpolate(ts[i], temp);

			for(int j=0; j < dim_; ++j)
				result[i*dim_ + j] = temp[j];
		}
	}

	void BSpline::interpolate(const double t, std::vector<double> &result) const
	{
		std::vector<double> edges, new_edges;
		std::vector<double> weigths;

		const int interval=find_interval(t);
		int start=first_interval(interval, degree_);
		find_edges(start, degree_, edges);


		for(int degree=degree_; degree>0;--degree)
		{
			start=first_interval(interval,degree);

			compute_weigth(t, start, degree, weigths);
			create_new_edges(weigths, edges, new_edges);
			edges=new_edges;
		}

		assert(edges.size()==std::size_t(dim_));

		result=edges;
	}

	void BSpline::derivative(BSpline &result) const
	{
		const int new_n_control=int(control_points_.size())/dim_ - 1;
		std::vector<double> new_control(dim_*new_n_control, 0);

		for(int i=0; i<new_n_control; ++i){
			for(int j=0; j < dim_; ++j)
				new_control[i*dim_+j] = degree_*(control_points_[(i+1)*dim_+j]-control_points_[i*dim_+j])/(knots_[i+degree_+1]-knots_[i+1]);
		}

		std::vector<double> new_knots(knots_.size()-2);
		for(std::size_t i = 1; i < knots_.size()-1; ++i)
			new_knots[i-1] = knots_[i];

		result.init(degree_-1, new_knots, new_control, dim_);
	}


	int BSpline::find_interval(const double t) const
	{
		for(std::size_t i=0;i<knots_.size()-1;++i)
		{
			if(t>=knots_[i] && t<=knots_[i+1])
			{
				std::size_t j;
				for(j=i;j<knots_.size();++j){
					if(fabs(knots_[i]-knots_[j])>1e-8)
						break;
				}
				--j;

				if(j==knots_.size()-2)
					j=i;

				return int(j);
			}
		}

		assert(false);
		return -1;
	}

	int BSpline::first_interval(const int interval, const int degree) const
	{
		int res=interval-degree+1;

		assert(res>0);
		assert(interval+degree<int(knots_.size())-1);
		return res;
	}

	void BSpline::find_edges(const int first_interval, const int degree, std::vector<double> &edges) const
	{
		edges.resize(dim_*(degree+1));

		for(int i=0; i<=degree; ++i){
			for(int j=0; j < dim_; ++j)
				edges[i*dim_+j] = control_points_[(first_interval+i-1)*dim_+j];
		}
	}

	void BSpline::create_new_edges(const std::vector<double> &weigths, const std::vector<double> &edges, std::vector<double> &new_edges) const
	{
		const int n_edges=int(edges.size())/dim_;

		new_edges.resize(dim_*(n_edges-1));

		for(int i=0;i<n_edges-1;++i)
		{
			for(int j=0; j < dim_; ++j)
				new_edges[i*dim_+j]= (1-weigths[i]) * edges[i*dim_+j] + weigths[i]*edges[(i+1)*dim_+j];
		}
	}

	void BSpline::compute_weigth(const double t, const int first_interval, const int degree, std::vector<double> &weigths) const
	{
		weigths.resize(degree);

		for(int i=0; i<degree; ++i)
		{
			const double distTot=knots_[first_interval+i+degree]-knots_[first_interval+i];
			const double dist=t-knots_[first_interval+i];
			weigths[i]=dist/distTot;
			assert(dist/distTot>=0);
			assert(dist/distTot<=1);
		}
	}
}


