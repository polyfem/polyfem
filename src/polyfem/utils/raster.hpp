#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <functional>
#include <vector>

namespace polyfem
{
	namespace renderer
	{
		struct Material
		{
			Eigen::Vector3d diffuse_color;
			Eigen::Vector3d specular_color;
			double specular_exponent;
		};
		class VertexAttributes
		{
		public:
			VertexAttributes(double x = 0, double y = 0, double z = 0, double w = 1)
			{
				position << x, y, z, w;
			}

			static VertexAttributes interpolate(
				const VertexAttributes &a,
				const VertexAttributes &b,
				const VertexAttributes &c,
				const double alpha,
				const double beta,
				const double gamma)
			{
				VertexAttributes r;
				r.position = alpha * (a.position / a.position(3)) + beta * (b.position / b.position(3)) + gamma * (c.position / c.position(3));
				r.color = alpha * a.color + beta * b.color + gamma * c.color;
				return r;
			}

			Eigen::Vector4d position;
			Eigen::Vector3d normal;
			Eigen::Vector3d color;
			Material material;
		};

		class FragmentAttributes
		{
		public:
			FragmentAttributes(double r = 0, double g = 0, double b = 0, double a = 1)
			{
				color << r, g, b, a;
			}
			double depth;
			Eigen::Vector4d color;
		};

		class FrameBufferAttributes
		{
		public:
			FrameBufferAttributes(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 255)
			{
				color << r, g, b, a;
				depth = 2;
			}
			double depth;
			Eigen::Matrix<uint8_t, 4, 1> color;
		};

		class UniformAttributes
		{
		public:
			Eigen::Matrix4d M_cam;

			Eigen::Matrix4d M_orth;
			Eigen::Matrix4d M;
		};

		class Program
		{
		public:
			std::function<VertexAttributes(const VertexAttributes &, const UniformAttributes &)> VertexShader;
			std::function<FragmentAttributes(const VertexAttributes &, const UniformAttributes &)> FragmentShader;
			std::function<FrameBufferAttributes(const FragmentAttributes &, const FrameBufferAttributes &)> BlendingShader;
		};

		std::vector<uint8_t> render(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &faces, const Eigen::MatrixXi &faces_id,
									int width, int height,
									const Eigen::Vector3d &camera_position, const double camera_fov, const double camera_near, const double camera_far, const bool is_perspective, const Eigen::Vector3d &lookat, const Eigen::Vector3d &up,
									const Eigen::Vector3d &ambient_light, const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &lights,
									std::vector<Material> &materials);
	} // namespace renderer
} // namespace polyfem