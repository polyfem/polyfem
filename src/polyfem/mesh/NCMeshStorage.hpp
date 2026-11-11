#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <vector>

namespace polyfem::mesh
{
	/// Small vector stored inline (no heap allocation), with the VectorXi /
	/// VectorXd API; used for the per-entity data of the non-conforming meshes.
	template <typename T, int MaxSize>
	using InlineVector = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor, MaxSize, 1>;

	/// Set of ids stored as a sorted vector: iterates in increasing order like
	/// std::set<int>, but uses a fraction of its memory for the short element
	/// lists of the non-conforming meshes.
	class SortedIds
	{
	public:
		using const_iterator = std::vector<int>::const_iterator;

		void insert(const int id)
		{
			const auto it = std::lower_bound(ids_.begin(), ids_.end(), id);
			if (it == ids_.end() || *it != id)
				ids_.insert(it, id);
		}

		/// Removes id; returns the number of removed ids (0 or 1).
		size_t erase(const int id)
		{
			const auto it = std::lower_bound(ids_.begin(), ids_.end(), id);
			if (it == ids_.end() || *it != id)
				return 0;
			ids_.erase(it);
			return 1;
		}

		size_t size() const { return ids_.size(); }
		bool empty() const { return ids_.empty(); }
		void clear() { ids_.clear(); }

		const_iterator begin() const { return ids_.begin(); }
		const_iterator end() const { return ids_.end(); }
		const_iterator cbegin() const { return ids_.cbegin(); }
		const_iterator cend() const { return ids_.cend(); }

	private:
		std::vector<int> ids_;
	};
} // namespace polyfem::mesh
