#include "PointTriangleDistance.hpp"

#include <cmath>

namespace polyfem::mesh
{
	namespace
	{
		// Plain 3D vector with the arithmetic of geogram's vec3 (same operation
		// order).
		struct Vec3
		{
			double x, y, z;
		};

		inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
		inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
		inline Vec3 operator*(double s, const Vec3 &v) { return {s * v.x, s * v.y, s * v.z}; }
		inline double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
		inline double length2(const Vec3 &v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
		inline double distance2(const Vec3 &a, const Vec3 &b)
		{
			const double dx = a.x - b.x;
			const double dy = a.y - b.y;
			const double dz = a.z - b.z;
			return dx * dx + dy * dy + dz * dz;
		}

		double point_segment_squared_distance(const Vec3 &point, const Vec3 &v0, const Vec3 &v1)
		{
			const double l2 = distance2(v0, v1);
			const double t = dot(point - v0, v1 - v0);
			if (t <= 0.0 || l2 == 0.0)
				return distance2(point, v0);
			else if (t > l2)
				return distance2(point, v1);
			const double lambda1 = t / l2;
			const double lambda0 = 1.0 - lambda1;
			const Vec3 closest_point = lambda0 * v0 + lambda1 * v1;
			return distance2(point, closest_point);
		}

		double point_triangle_squared_distance(const Vec3 &point, const Vec3 &V0, const Vec3 &V1, const Vec3 &V2)
		{
			Vec3 diff = V0 - point;
			Vec3 edge0 = V1 - V0;
			Vec3 edge1 = V2 - V0;
			double a00 = length2(edge0);
			double a01 = dot(edge0, edge1);
			double a11 = length2(edge1);
			double b0 = dot(diff, edge0);
			double b1 = dot(diff, edge1);
			double c = length2(diff);
			double det = ::fabs(a00 * a11 - a01 * a01);
			double s = a01 * b1 - a11 * b0;
			double t = a01 * b0 - a00 * b1;
			double sqrDistance;

			// If the triangle is degenerate
			if (det < 1e-30)
			{
				double result = point_segment_squared_distance(point, V0, V1);
				double cur_dist = point_segment_squared_distance(point, V0, V2);
				if (cur_dist < result)
					result = cur_dist;
				cur_dist = point_segment_squared_distance(point, V1, V2);
				if (cur_dist < result)
					result = cur_dist;
				return result;
			}

			if (s + t <= det)
			{
				if (s < 0.0)
				{
					if (t < 0.0)
					{ // region 4
						if (b0 < 0.0)
						{
							t = 0.0;
							if (-b0 >= a00)
							{
								s = 1.0;
								sqrDistance = a00 + 2.0 * b0 + c;
							}
							else
							{
								s = -b0 / a00;
								sqrDistance = b0 * s + c;
							}
						}
						else
						{
							s = 0.0;
							if (b1 >= 0.0)
							{
								t = 0.0;
								sqrDistance = c;
							}
							else if (-b1 >= a11)
							{
								t = 1.0;
								sqrDistance = a11 + 2.0 * b1 + c;
							}
							else
							{
								t = -b1 / a11;
								sqrDistance = b1 * t + c;
							}
						}
					}
					else
					{ // region 3
						s = 0.0;
						if (b1 >= 0.0)
						{
							t = 0.0;
							sqrDistance = c;
						}
						else if (-b1 >= a11)
						{
							t = 1.0;
							sqrDistance = a11 + 2.0 * b1 + c;
						}
						else
						{
							t = -b1 / a11;
							sqrDistance = b1 * t + c;
						}
					}
				}
				else if (t < 0.0)
				{ // region 5
					t = 0.0;
					if (b0 >= 0.0)
					{
						s = 0.0;
						sqrDistance = c;
					}
					else if (-b0 >= a00)
					{
						s = 1.0;
						sqrDistance = a00 + 2.0 * b0 + c;
					}
					else
					{
						s = -b0 / a00;
						sqrDistance = b0 * s + c;
					}
				}
				else
				{ // region 0
					// minimum at interior point
					double invDet = double(1.0) / det;
					s *= invDet;
					t *= invDet;
					sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c;
				}
			}
			else
			{
				double tmp0, tmp1, numer, denom;

				if (s < 0.0)
				{ // region 2
					tmp0 = a01 + b0;
					tmp1 = a11 + b1;
					if (tmp1 > tmp0)
					{
						numer = tmp1 - tmp0;
						denom = a00 - 2.0 * a01 + a11;
						if (numer >= denom)
						{
							s = 1.0;
							t = 0.0;
							sqrDistance = a00 + 2.0 * b0 + c;
						}
						else
						{
							s = numer / denom;
							t = 1.0 - s;
							sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c;
						}
					}
					else
					{
						s = 0.0;
						if (tmp1 <= 0.0)
						{
							t = 1.0;
							sqrDistance = a11 + 2.0 * b1 + c;
						}
						else if (b1 >= 0.0)
						{
							t = 0.0;
							sqrDistance = c;
						}
						else
						{
							t = -b1 / a11;
							sqrDistance = b1 * t + c;
						}
					}
				}
				else if (t < 0.0)
				{ // region 6
					tmp0 = a01 + b1;
					tmp1 = a00 + b0;
					if (tmp1 > tmp0)
					{
						numer = tmp1 - tmp0;
						denom = a00 - 2.0 * a01 + a11;
						if (numer >= denom)
						{
							t = 1.0;
							s = 0.0;
							sqrDistance = a11 + 2.0 * b1 + c;
						}
						else
						{
							t = numer / denom;
							s = 1.0 - t;
							sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c;
						}
					}
					else
					{
						t = 0.0;
						if (tmp1 <= 0.0)
						{
							s = 1.0;
							sqrDistance = a00 + 2.0 * b0 + c;
						}
						else if (b0 >= 0.0)
						{
							s = 0.0;
							sqrDistance = c;
						}
						else
						{
							s = -b0 / a00;
							sqrDistance = b0 * s + c;
						}
					}
				}
				else
				{ // region 1
					numer = a11 + b1 - a01 - b0;
					if (numer <= 0.0)
					{
						s = 0.0;
						t = 1.0;
						sqrDistance = a11 + 2.0 * b1 + c;
					}
					else
					{
						denom = a00 - 2.0 * a01 + a11;
						if (numer >= denom)
						{
							s = 1.0;
							t = 0.0;
							sqrDistance = a00 + 2.0 * b0 + c;
						}
						else
						{
							s = numer / denom;
							t = 1.0 - s;
							sqrDistance = s * (a00 * s + a01 * t + 2.0 * b0) + t * (a01 * s + a11 * t + 2.0 * b1) + c;
						}
					}
				}
			}

			// Account for numerical round-off error.
			if (sqrDistance < 0.0)
			{
				sqrDistance = 0.0;
			}

			return sqrDistance;
		}
	} // namespace

	double point_triangle_squared_distance(const double *p, const double *v0, const double *v1, const double *v2)
	{
		return point_triangle_squared_distance(
			Vec3{p[0], p[1], p[2]}, Vec3{v0[0], v0[1], v0[2]}, Vec3{v1[0], v1[1], v1[2]}, Vec3{v2[0], v2[1], v2[2]});
	}
} // namespace polyfem::mesh
