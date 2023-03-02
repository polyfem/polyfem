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

			Selection(const int id) : id_(id) {}

			virtual ~Selection() {}

			virtual bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const = 0;

			virtual int id(const size_t element_id, const std::vector<int> &vs, const RowVectorNd &p) const
			{
				return id_;
			}

			/// @brief Build a selection objects from a JSON selection.
			/// @param j_selections JSON object of selection(s).
			/// @param mesh_bbox    Bounding box of the mesh.
			/// @param root_path    Root path of the JSON file.
			/// @return Shared pointer to selection object.
			static std::shared_ptr<Selection> build(
				const json &j_selections,
				const BBox &mesh_bbox,
				const std::string &root_path = "");

			/// @brief Build a vector of selection objects from a JSON selection(s).
			/// @param j_selections JSON object of selection(s).
			/// @param mesh_bbox    Bounding box of the mesh.
			/// @param root_path    Root path of the JSON file.
			/// @return Vector of selection objects.
			static std::vector<std::shared_ptr<utils::Selection>> build_selections(
				const json &j_selections,
				const BBox &mesh_bbox,
				const std::string &root_path = "");

		protected:
			Selection() {}

			size_t id_;
		};

		// --------------------------------------------------------------------

		class BoxSelection : public Selection
		{
		public:
			BoxSelection(
				const json &selection,
				const BBox &mesh_bbox);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			BBox bbox_;
		};

		// --------------------------------------------------------------------

		class BoxSideSelection : public Selection
		{
		public:
			BoxSideSelection(
				const json &selection,
				const BBox &mesh_bbox);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override
			{
				return true;
			}

			int id(const size_t element_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			BBox mesh_bbox_;
			double tolerance_;
			int id_offset_;
		};

		// --------------------------------------------------------------------

		class SphereSelection : public Selection
		{
		public:
			SphereSelection(
				const json &selection,
				const BBox &mesh_bbox);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			RowVectorNd center_;
			double radius2_;
		};

		// --------------------------------------------------------------------

		class CylinderSelection : public Selection
		{
		public:
			CylinderSelection(
				const json &selection,
				const BBox &mesh_bbox);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			RowVectorNd axis_;
			RowVectorNd point_;
			double radius2_;
			double height_;
		};

		// --------------------------------------------------------------------

		class AxisPlaneSelection : public Selection
		{
		public:
			AxisPlaneSelection(
				const json &selection,
				const BBox &mesh_bbox);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			int axis_;
			double position_;
		};

		// --------------------------------------------------------------------

		class PlaneSelection : public Selection
		{
		public:
			PlaneSelection(
				const json &selection,
				const BBox &mesh_bbox);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			RowVectorNd normal_;
			RowVectorNd point_;
		};

		// --------------------------------------------------------------------

		class UniformSelection : public Selection
		{
		public:
			UniformSelection(const int id)
				: Selection(id) {}

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override { return true; }
		};

		// --------------------------------------------------------------------

		class SpecifiedSelection : public Selection
		{
		public:
			SpecifiedSelection(
				const std::vector<int> &ids);

			virtual bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override { return true; }

			virtual int id(const size_t element_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		protected:
			SpecifiedSelection() {}

			std::vector<int> ids_;
		};

		// --------------------------------------------------------------------

		class FileSelection : public SpecifiedSelection
		{
		public:
			FileSelection(
				const std::string &file_path,
				const int id_offset = 0);

			bool inside(const size_t p_id, const std::vector<int> &vs, const RowVectorNd &p) const override;
			int id(const size_t element_id, const std::vector<int> &vs, const RowVectorNd &p) const override;

		private:
			std::vector<std::pair<int, std::vector<int>>> data_;
		};
	} // namespace utils
} // namespace polyfem
