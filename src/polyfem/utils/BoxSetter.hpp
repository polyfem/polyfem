#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/Common.hpp>

namespace polyfem
{
	class Selection
	{
	public:
		typedef std::array<RowVectorNd, 2> BBox;

		virtual bool inside(const RowVectorNd &p) const = 0;
		static std::shared_ptr<Selection> build(const json &selection, const BBox &mesh_bbox);
		virtual ~Selection() {}
	};

	class Box : public Selection
	{
	public:
		Box(const json &selection, const BBox &mesh_bbox);
		bool inside(const RowVectorNd &p) const override;

	private:
		BBox bbox_;
	};

	class Sphere : public Selection
	{
	public:
		Sphere(const json &selection, const BBox &mesh_bbox);
		bool inside(const RowVectorNd &p) const override;

	private:
		RowVectorNd center_;
		double radius2_;
	};

	class AxisPlane : public Selection
	{
	public:
		AxisPlane(const json &selection, const BBox &mesh_bbox);
		bool inside(const RowVectorNd &p) const override;

	private:
		int axis_;
		double position_;
	};

	class Plane : public Selection
	{
	public:
		Plane(const json &selection, const BBox &mesh_bbox);
		bool inside(const RowVectorNd &p) const override;

	private:
		RowVectorNd normal_;
		RowVectorNd point_;
	};

	class BoxSetter
	{
	public:
		static void set_sidesets(const json &args, mesh::Mesh &mesh);
	};
} // namespace polyfem
