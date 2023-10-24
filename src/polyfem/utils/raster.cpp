#include "raster.hpp"

#include <igl/per_vertex_normals.h>
#include <iostream>

namespace polyfem
{
	namespace renderer
	{
		namespace
		{
			void rasterize_triangle(const Program &program, const UniformAttributes &uniform, const VertexAttributes &v1, const VertexAttributes &v2, const VertexAttributes &v3, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
			{
				Eigen::Matrix<double, 3, 4> p;
				p.row(0) = v1.position.array() / v1.position[3];
				p.row(1) = v2.position.array() / v2.position[3];
				p.row(2) = v3.position.array() / v3.position[3];

				p.col(0) = ((p.col(0).array() + 1.0) / 2.0) * frameBuffer.rows();
				p.col(1) = ((p.col(1).array() + 1.0) / 2.0) * frameBuffer.cols();

				int lx = std::floor(p.col(0).minCoeff());
				int ly = std::floor(p.col(1).minCoeff());
				int ux = std::ceil(p.col(0).maxCoeff());
				int uy = std::ceil(p.col(1).maxCoeff());

				lx = std::min(std::max(lx, int(0)), int(frameBuffer.rows() - 1));
				ly = std::min(std::max(ly, int(0)), int(frameBuffer.cols() - 1));
				ux = std::max(std::min(ux, int(frameBuffer.rows() - 1)), int(0));
				uy = std::max(std::min(uy, int(frameBuffer.cols() - 1)), int(0));

				Eigen::Matrix3d A;
				A.col(0) = p.row(0).segment(0, 3);
				A.col(1) = p.row(1).segment(0, 3);
				A.col(2) = p.row(2).segment(0, 3);
				A.row(2) << 1.0, 1.0, 1.0;

				Eigen::Matrix3d Ai = A.inverse();

				for (unsigned i = lx; i <= ux; i++)
				{
					for (unsigned j = ly; j <= uy; j++)
					{

						Eigen::Vector3d pixel(i + 0.5, j + 0.5, 1);
						Eigen::Vector3d b = Ai * pixel;
						if (b.minCoeff() >= 0)
						{
							VertexAttributes va = VertexAttributes::interpolate(v1, v2, v3, b[0], b[1], b[2]);

							if (va.position[2] >= -1 && va.position[2] <= 1)
							{
								FragmentAttributes frag = program.FragmentShader(va, uniform);
								frameBuffer(i, j) = program.BlendingShader(frag, frameBuffer(i, j));
							}
						}
					}
				}
			}

			void rasterize_triangles(const Program &program, const UniformAttributes &uniform, const std::vector<VertexAttributes> &vertices, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
			{
				std::vector<VertexAttributes> v(vertices.size());
				for (unsigned i = 0; i < vertices.size(); i++)
					v[i] = program.VertexShader(vertices[i], uniform);

				for (unsigned i = 0; i < vertices.size() / 3; i++)
					rasterize_triangle(program, uniform, v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2], frameBuffer);
			}

			void rasterize_line(const Program &program, const UniformAttributes &uniform, const VertexAttributes &v1, const VertexAttributes &v2, double line_thickness, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
			{
				Eigen::Matrix<double, 2, 4> p;
				p.row(0) = v1.position.array() / v1.position[3];
				p.row(1) = v2.position.array() / v2.position[3];

				p.col(0) = ((p.col(0).array() + 1.0) / 2.0) * frameBuffer.rows();
				p.col(1) = ((p.col(1).array() + 1.0) / 2.0) * frameBuffer.cols();

				int lx = std::floor(p.col(0).minCoeff() - line_thickness);
				int ly = std::floor(p.col(1).minCoeff() - line_thickness);
				int ux = std::ceil(p.col(0).maxCoeff() + line_thickness);
				int uy = std::ceil(p.col(1).maxCoeff() + line_thickness);

				lx = std::min(std::max(lx, int(0)), int(frameBuffer.rows() - 1));
				ly = std::min(std::max(ly, int(0)), int(frameBuffer.cols() - 1));
				ux = std::max(std::min(ux, int(frameBuffer.rows() - 1)), int(0));
				uy = std::max(std::min(uy, int(frameBuffer.cols() - 1)), int(0));

				Eigen::Vector2f l1(p(0, 0), p(0, 1));
				Eigen::Vector2f l2(p(1, 0), p(1, 1));

				double t = -1;
				double ll = (l1 - l2).squaredNorm();

				for (unsigned i = lx; i <= ux; i++)
				{
					for (unsigned j = ly; j <= uy; j++)
					{

						Eigen::Vector2f pixel(i + 0.5, j + 0.5);

						if (ll == 0.0)
							t = 0;
						else
						{
							t = (pixel - l1).dot(l2 - l1) / ll;
							t = std::fmax(0, std::fmin(1, t));
						}

						Eigen::Vector2f pixel_p = l1 + t * (l2 - l1);

						if ((pixel - pixel_p).squaredNorm() < (line_thickness * line_thickness))
						{
							VertexAttributes va = VertexAttributes::interpolate(v1, v2, v1, 1 - t, t, 0);
							FragmentAttributes frag = program.FragmentShader(va, uniform);
							frameBuffer(i, j) = program.BlendingShader(frag, frameBuffer(i, j));
						}
					}
				}
			}

			void rasterize_lines(const Program &program, const UniformAttributes &uniform, const std::vector<VertexAttributes> &vertices, double line_thickness, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
			{
				std::vector<VertexAttributes> v(vertices.size());
				for (unsigned i = 0; i < vertices.size(); i++)
					v[i] = program.VertexShader(vertices[i], uniform);

				for (unsigned i = 0; i < vertices.size() / 2; i++)
					rasterize_line(program, uniform, v[i * 2 + 0], v[i * 2 + 1], line_thickness, frameBuffer);
			}

			void framebuffer_to_uint8(const Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer, std::vector<uint8_t> &image)
			{
				const int w = frameBuffer.rows();
				const int h = frameBuffer.cols();
				const int comp = 4;
				const int stride_in_bytes = w * comp;
				image.resize(w * h * comp, 0);

				for (unsigned wi = 0; wi < w; ++wi)
				{
					for (unsigned hi = 0; hi < h; ++hi)
					{
						unsigned hif = h - 1 - hi;
						image[(hi * w * 4) + (wi * 4) + 0] = frameBuffer(wi, hif).color[0];
						image[(hi * w * 4) + (wi * 4) + 1] = frameBuffer(wi, hif).color[1];
						image[(hi * w * 4) + (wi * 4) + 2] = frameBuffer(wi, hif).color[2];
						image[(hi * w * 4) + (wi * 4) + 3] = frameBuffer(wi, hif).color[3];
					}
				}
			}
		} // namespace

		std::vector<uint8_t> render(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &faces, const Eigen::MatrixXi &faces_id,
									int width, int height,
									const Eigen::Vector3d &camera_position, const double camera_fov, const double camera_near, const double camera_far, const bool is_perspective, const Eigen::Vector3d &lookat, const Eigen::Vector3d &up,
									const Eigen::Vector3d &ambient_light, const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &lights,
									std::vector<Material> &materials)
		{
			using namespace renderer;
			using namespace Eigen;

			Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(width, height);
			UniformAttributes uniform;

			const Vector3d gaze = lookat - camera_position;
			const Vector3d w = -gaze.normalized();
			const Vector3d u = up.cross(w).normalized();
			const Vector3d v = w.cross(u);

			Matrix4d M_cam_inv;
			M_cam_inv << u(0), v(0), w(0), camera_position(0),
				u(1), v(1), w(1), camera_position(1),
				u(2), v(2), w(2), camera_position(2),
				0, 0, 0, 1;

			uniform.M_cam = M_cam_inv.inverse();

			{
				const double camera_ar = double(width) / height;
				const double tan_angle = tan(camera_fov / 2);
				const double n = -camera_near;
				const double f = -camera_far;
				const double t = tan_angle * n;
				const double b = -t;
				const double r = t * camera_ar;
				const double l = -r;

				uniform.M_orth << 2 / (r - l), 0, 0, -(r + l) / (r - l),
					0, 2 / (t - b), 0, -(t + b) / (t - b),
					0, 0, 2 / (n - f), -(n + f) / (n - f),
					0, 0, 0, 1;
				Matrix4d P;
				if (is_perspective)
				{
					P << n, 0, 0, 0,
						0, n, 0, 0,
						0, 0, n + f, -f * n,
						0, 0, 1, 0;
				}
				else
				{
					P << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
				}

				uniform.M = uniform.M_orth * P * uniform.M_cam;
			}

			Program program;
			program.VertexShader = [&](const VertexAttributes &va, const UniformAttributes &uniform) {
				VertexAttributes out;
				out.position = uniform.M * va.position;
				Vector3d color = ambient_light;

				Vector3d hit(va.position(0), va.position(1), va.position(2));
				for (const auto &l : lights)
				{
					Vector3d Li = (l.first - hit).normalized();
					Vector3d N = va.normal;
					Vector3d diffuse = va.material.diffuse_color * std::max(Li.dot(N), 0.0);
					Vector3d H;
					if (is_perspective)
					{
						H = (Li + (camera_position - hit).normalized()).normalized();
					}
					else
					{
						H = (Li - gaze.normalized()).normalized();
					}
					const Vector3d specular = va.material.specular_color * std::pow(std::max(N.dot(H), 0.0), va.material.specular_exponent);
					const Vector3d D = l.first - hit;
					color += (diffuse + specular).cwiseProduct(l.second) / D.squaredNorm();
				}
				out.color = color;
				return out;
			};

			program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
				FragmentAttributes out(va.color(0), va.color(1), va.color(2));
				out.depth = -va.position(2);
				return out;
			};

			program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
				if (fa.depth < previous.depth)
				{
					FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
					out.depth = fa.depth;
					return out;
				}
				else
				{
					return previous;
				}
			};

			Eigen::MatrixXd vnormals;
			igl::per_vertex_normals(vertices, faces, vnormals);
			const Material material = materials.front();

			std::vector<VertexAttributes> vertex_attributes;
			for (int i = 0; i < faces.rows(); ++i)
			{
				const auto &mat = faces_id.size() <= 0 ? material : materials[faces_id(i)];
				for (int j = 0; j < 3; j++)
				{
					int vid = faces(i, j);
					VertexAttributes va(vertices(vid, 0), vertices(vid, 1), vertices(vid, 2));
					va.material = mat;
					va.normal = vnormals.row(vid).normalized();
					vertex_attributes.push_back(va);
				}
			}

			rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);

			std::vector<uint8_t> image;
			framebuffer_to_uint8(frameBuffer, image);

			return image;
		}
	} // namespace renderer
} // namespace polyfem