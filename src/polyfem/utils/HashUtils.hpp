#pragma once

#include <cstddef> // size_t
#include <vector>

namespace polyfem::utils
{
	struct HashPair
	{
		template <typename T1, typename T2>
		size_t operator()(const std::pair<T1, T2> &p) const noexcept
		{
			auto hash1 = std::hash<T1>{}(p.first);
			auto hash2 = std::hash<T2>{}(p.second);
			if (hash1 != hash2)
				return hash1 ^ hash2;
			// If hash1 == hash2, their XOR is zero.
			return hash1;
		}
	};

	template <>
	inline size_t HashPair::operator()<int, int>(const std::pair<int, int> &p) const noexcept
	{
		size_t h = (size_t(p.first) << 32) + size_t(p.second);
		h *= 1231231557ull; // "random" uneven integer
		h ^= (h >> 32);
		return h;
	}

	struct HashVector
	{
		template <typename T>
		size_t operator()(const std::vector<T> &v) const
		{
			std::hash<T> hasher;
			size_t hash = 0;
			for (int i : v)
				hash ^= hasher(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
			return hash;
		}
	};

	// https://github.com/ethz-asl/map_api/blob/master/map-api-common/include/map-api-common/eigen-hash.h
	struct HashMatrix
	{
		// https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
		template <typename Scalar, int Rows, int Cols>
		size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols> &matrix) const
		{
			size_t seed = 0;
			for (size_t i = 0; i < matrix.size(); ++i)
			{
				Scalar elem = *(matrix.data() + i);
				seed ^=
					std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};

} // namespace polyfem::utils