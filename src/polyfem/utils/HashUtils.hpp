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

} // namespace polyfem::utils