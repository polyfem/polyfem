#pragma once

#include <Eigen/Dense>

#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace polyfem
{
	namespace mesh
	{
		/// Contiguous view of one row of a JaggedArray (like std::span).
		template <typename T>
		class ArrayRow
		{
		public:
			ArrayRow(T *begin, T *end) : begin_(begin), end_(end) {}

			T *begin() const { return begin_; }
			T *end() const { return end_; }
			size_t size() const { return end_ - begin_; }
			bool empty() const { return begin_ == end_; }
			T &operator[](const size_t i) const
			{
				assert(i < size());
				return begin_[i];
			}
			T &front() const { return *begin_; }
			T &back() const { return *(end_ - 1); }

			/// Copy of the row.
			std::vector<std::remove_const_t<T>> to_vector() const { return {begin_, end_}; }

		private:
			T *begin_;
			T *end_;
		};

		/// Rows of different lengths stored contiguously (compressed rows): row i
		/// is data[offsets[i], offsets[i + 1]).
		template <typename T>
		class JaggedArray
		{
		public:
			/// Number of rows.
			int size() const { return int(offsets_.size()) - 1; }
			bool empty() const { return size() == 0; }
			/// Total number of entries.
			size_t n_entries() const { return data_.size(); }

			ArrayRow<T> operator[](const int i)
			{
				assert(i >= 0 && i < size());
				return {data_.data() + offsets_[i], data_.data() + offsets_[i + 1]};
			}
			ArrayRow<const T> operator[](const int i) const
			{
				assert(i >= 0 && i < size());
				return {data_.data() + offsets_[i], data_.data() + offsets_[i + 1]};
			}
			int row_size(const int i) const { return int(offsets_[i + 1] - offsets_[i]); }

			void clear()
			{
				offsets_.assign(1, 0);
				data_.clear();
			}
			void reserve(const size_t n_rows, const size_t n_entries)
			{
				offsets_.reserve(n_rows + 1);
				data_.reserve(n_entries);
			}

			/// Appends a row.
			template <typename It>
			void push_back(It begin, It end)
			{
				data_.insert(data_.end(), begin, end);
				push_offset();
			}
			template <typename Range>
			void push_back(const Range &row) { push_back(std::begin(row), std::end(row)); }
			void push_back(std::initializer_list<T> row) { push_back(row.begin(), row.end()); }
			/// Appends a row of n copies of value.
			void push_back_filled(const size_t n, const T &value)
			{
				data_.resize(data_.size() + n, value);
				push_offset();
			}

			/// Replaces the content with rows of the given sizes, filled with value.
			void assign_sizes(const std::vector<uint32_t> &sizes, const T &value = T())
			{
				offsets_.resize(sizes.size() + 1);
				offsets_[0] = 0;
				for (size_t i = 0; i < sizes.size(); ++i)
					offsets_[i + 1] = offsets_[i] + sizes[i];
				data_.assign(offsets_.back(), value);
			}

			/// Appends the rows of other, adding shift to every entry.
			void append(const JaggedArray &other, const T shift)
			{
				const uint32_t n = data_.size();
				offsets_.reserve(offsets_.size() + other.size());
				for (int i = 0; i < other.size(); ++i)
					offsets_.push_back(n + other.offsets_[i + 1]);
				data_.reserve(n + other.data_.size());
				for (const T &v : other.data_)
					data_.push_back(v + shift);
			}

			size_t memory() const { return offsets_.capacity() * sizeof(uint32_t) + data_.capacity() * sizeof(T); }

		private:
			void push_offset()
			{
				assert(data_.size() <= std::numeric_limits<uint32_t>::max());
				offsets_.push_back(data_.size());
			}

			std::vector<uint32_t> offsets_ = {0};
			std::vector<T> data_;
		};

		enum class MeshType
		{
			TRI = 0,
			QUA,
			H_SUR,
			TET,
			HYB,
			HEX
		};

		/// Topology of a conforming 3D mesh (used by CMesh3D), stored as flat
		/// arrays indexed by entity id.
		///
		/// The input topology is: points, face_vs (vertices of each face, in order
		/// around it), cell_fs and cell_fs_flag (faces of each cell and whether
		/// each one is oriented inward), cell_hex, cell_input_vs and cell_kernel
		/// (and cell_vs for the HEX type). MeshProcessing3D::build_connectivity
		/// derives everything else:
		/// - edges sorted by (min, max) vertex id, edge_vs = (min, max);
		/// - face_es[f][j] = edge (face_vs[f][j], face_vs[f][j + 1]);
		/// - cell_vs: hexes in navigation order (a face, then the opposite
		///   vertices), other cells sorted; cell_es sorted;
		/// - every incidence list (vertex_*, edge_fs/hs, face_hs) sorted by id,
		///   vertex_vs[v][k] being the other vertex of the edge vertex_es[v][k];
		/// - the boundary flags.
		/// Face ids are the creation order and are never renumbered.
		class Mesh3DStorage
		{
		public:
			MeshType type = MeshType::HYB;
			/// Vertex positions, 3 x #vertices.
			Eigen::MatrixXd points;

			// Vertices: incident vertices, edges, faces and cells.
			JaggedArray<uint32_t> vertex_vs, vertex_es, vertex_fs, vertex_hs;
			std::vector<uint8_t> vertex_boundary;

			// Edges: endpoints (min, max), incident faces and cells.
			std::vector<std::array<uint32_t, 2>> edge_vs;
			JaggedArray<uint32_t> edge_fs, edge_hs;
			std::vector<uint8_t> edge_boundary;

			// Faces: vertices in order, edges aligned with them, incident cells and
			// the local index of the face in each of these cells.
			JaggedArray<uint32_t> face_vs, face_es, face_hs;
			JaggedArray<uint8_t> face_hs_slot;
			std::vector<uint8_t> face_boundary;

			// Cells.
			JaggedArray<uint32_t> cell_vs;       ///< navigation order (see above)
			JaggedArray<uint32_t> cell_input_vs; ///< input order (rows may be empty)
			JaggedArray<uint32_t> cell_es, cell_fs;
			JaggedArray<uint8_t> cell_fs_flag;
			std::vector<uint8_t> cell_hex;
			/// A point from which the cell is star-shaped.
			std::vector<std::array<double, 3>> cell_kernel;

			int n_vertices() const { return points.cols(); }
			int n_edges() const { return edge_vs.size(); }
			int n_faces() const { return face_vs.size(); }
			int n_cells() const { return cell_fs.size(); }

			// Fixed-size accessors used by the tetrahedral navigation.
			int EV(const int k, const int e) const { return edge_vs[e][k]; }
			int FV(const int k, const int f) const { return face_vs[f][k]; }
			int FE(const int k, const int f) const { return face_es[f][k]; }
			int HF(const int k, const int c) const { return cell_fs[c][k]; }
			int FH(const int k, const int f) const { return k < face_hs.row_size(f) ? int(face_hs[f][k]) : -1; }
			int FHi(const int k, const int f) const { return k < face_hs.row_size(f) ? int(face_hs_slot[f][k]) : -1; }

			/// Appends the input topology of other (shifting its ids). The
			/// connectivity must then be rebuilt (Navigation3D::prepare_mesh).
			void append(const Mesh3DStorage &other);

			/// Bytes allocated by the storage.
			size_t memory() const;
		};

		struct Mesh_Quality
		{
			std::string Name;
			double min_Jacobian;
			double ave_Jacobian;
			double deviation_Jacobian;
			Eigen::VectorXd V_Js;
			Eigen::VectorXd H_Js;
			Eigen::VectorXd Num_Js;
			int32_t V_num, H_num;
		};
	} // namespace mesh
} // namespace polyfem
