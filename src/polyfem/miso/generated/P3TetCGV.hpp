#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P3TetCGV {
		public:
		static constexpr unsigned dimension = 4;
		static constexpr unsigned numVertices = 8;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = true;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P3TetCGV(
			const RealVector<20> &p0x,
			const RealVector<20> &p0y,
			const RealVector<20> &p0z,
			const RealVector<20> &p1x,
			const RealVector<20> &p1y,
			const RealVector<20> &p1z
		);

		P3TetCGV() = default;

		unsigned schemeSize() const {
			auto s = (scheme < schemes.size()) ? scheme : 0;
			return schemes.at(s);
		}

		const Real width() const {
			return std::max({
				x0.inclusion().width(),
				x1.inclusion().width(),
				x2.inclusion().width(),
				x3.inclusion().width()
			});
		}

		RealVector<dimension> getVertex(unsigned i) const {
			return {
				x0[i],
				x1[i],
				x2[i],
				x3[i]
			};
		}

		static constexpr std::array<unsigned, 2> schemes = {16, 2};

		RealInterval inclusion(unsigned i) const;
		RealVector<numVertices> sample(unsigned i) const;
		template<typename F> void split(unsigned scheme, F &&f) const {
			if (scheme == 1) { for (auto &&_c : split_impl<1>()) f(std::move(_c)); }
			else { for (auto &&_c : split_impl<0>()) f(std::move(_c)); }
		}
		void inherit(const P3TetCGV &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P3TetCGV &s);

		private:

		RealVector<336> m0;
		RealVector<8> x0;
		RealVector<8> x1;
		RealVector<8> x2;
		RealVector<8> x3;

		P3TetCGV(const P3TetCGV &parent, unsigned q,
			const RealVector<336> &m0,
			const RealVector<8> &x0,
			const RealVector<8> &x1,
			const RealVector<8> &x2,
			const RealVector<8> &x3
		);

		static RealVector<336> CL_m0(const RealVector<20> &p0x, const RealVector<20> &p0y, const RealVector<20> &p0z, const RealVector<20> &p1x, const RealVector<20> &p1y, const RealVector<20> &p1z);
		static RealVector<336> LB_3p6_1p3(const RealVector<336> &_l);
		static std::array<RealVector<8>, 16> subdiv_0_3p1_1p1(const RealVector<8> &_b);
		static std::array<RealVector<8>, 2> subdiv_1_3p1_1p1(const RealVector<8> &_b);
		static std::array<RealVector<336>, 16> subdiv_0_3p6_1p3(const RealVector<336> &_b);
		static std::array<RealVector<336>, 2> subdiv_1_3p6_1p3(const RealVector<336> &_b);
		template<unsigned SI=0> std::array<P3TetCGV, schemes[SI]> split_impl() const;
	};

	template<> std::array<P3TetCGV, P3TetCGV::schemes[0]> P3TetCGV::split_impl<0>() const;
	template<> std::array<P3TetCGV, P3TetCGV::schemes[1]> P3TetCGV::split_impl<1>() const;
}
