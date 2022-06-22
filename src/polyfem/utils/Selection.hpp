#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/Common.hpp>

namespace polyfem
{
	namespace utils
	{
		class Selection
		{
		public:
			typedef std::array<RowVectorNd, 2> BBox;

			Selection(
				const int id,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());

			virtual ~Selection() {}

			virtual bool inside(
				const size_t element_id, const RowVectorNd &p) const;
			virtual int id(const size_t element_id) const { return id_; }

			static std::shared_ptr<Selection> build(
				const json &selection,
				const BBox &mesh_bbox,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());

		protected:
			size_t id_;
			size_t start_element_id_;
			size_t end_element_id_;
		};

		///////////////////////////////////////////////////////////////////////

		class BoxSelection : public Selection
		{
		public:
			BoxSelection(
				const json &selection,
				const BBox &mesh_bbox,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());
			bool inside(const size_t element_id, const RowVectorNd &p) const override;

		protected:
			BBox bbox_;
		};

		///////////////////////////////////////////////////////////////////////

		class SphereSelection : public Selection
		{
		public:
			SphereSelection(
				const json &selection,
				const BBox &mesh_bbox,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());
			bool inside(const size_t element_id, const RowVectorNd &p) const override;

		protected:
			RowVectorNd center_;
			double radius2_;
		};

		///////////////////////////////////////////////////////////////////////

		class AxisPlaneSelection : public Selection
		{
		public:
			AxisPlaneSelection(
				const json &selection,
				const BBox &mesh_bbox,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());
			bool inside(const size_t element_id, const RowVectorNd &p) const override;

		protected:
			int axis_;
			double position_;
		};

		///////////////////////////////////////////////////////////////////////

		class PlaneSelection : public Selection
		{
		public:
			PlaneSelection(
				const json &selection,
				const BBox &mesh_bbox,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());
			bool inside(const size_t element_id, const RowVectorNd &p) const override;

		protected:
			RowVectorNd normal_;
			RowVectorNd point_;
		};

		///////////////////////////////////////////////////////////////////////

		class UniformSelection : public Selection
		{
		public:
			UniformSelection(
				const int id,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max())
				: Selection(id, start_element_id, end_element_id) {}
		};

		///////////////////////////////////////////////////////////////////////

		class SpecifiedSelection : public Selection
		{
		public:
			SpecifiedSelection(
				const std::vector<int> &ids,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max());
			int id(const size_t element_id) const override;

		protected:
			std::vector<int> ids_;
		};

		///////////////////////////////////////////////////////////////////////

		class FileSelection : public SpecifiedSelection
		{
		public:
			FileSelection(
				const std::string &file_path,
				const size_t start_element_id = 0,
				const size_t end_element_id = std::numeric_limits<size_t>::max(),
				const int id_offset = 0);
		};
	} // namespace utils
} // namespace polyfem
