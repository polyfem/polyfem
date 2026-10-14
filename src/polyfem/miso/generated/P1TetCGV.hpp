#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P1TetCGV {
		public:
		static constexpr unsigned dimension = 1;
		static constexpr unsigned numVertices = 2;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = true;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P1TetCGV(
			const RealVector<4> &rx,
			const RealVector<4> &ry,
			const RealVector<4> &rz,
			const RealVector<4> &u0x,
			const RealVector<4> &u0y,
			const RealVector<4> &u0z,
			const RealVector<4> &u1x,
			const RealVector<4> &u1y,
			const RealVector<4> &u1z
		);

		P1TetCGV() = default;

		unsigned schemeSize() const {
			auto s = (scheme < schemes.size()) ? scheme : 0;
			return schemes.at(s);
		}

		const Real width() const {
			return std::max({
				x0.inclusion().width()
			});
		}

		RealVector<dimension> getVertex(unsigned i) const {
			return {
				x0[i]
			};
		}

		static constexpr std::array<unsigned, 1> schemes = {2};

		RealInterval inclusion(unsigned i) const;
		RealVector<numVertices> sample(unsigned i) const;
		template<typename F> void split(unsigned scheme, F &&f) const {
			for (auto &&_c : split_impl<0>()) f(std::move(_c));
		}
		void inherit(const P1TetCGV &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P1TetCGV &s);

		private:

		RealVector<4> m0;
		RealVector<2> x0;

		P1TetCGV(const P1TetCGV &parent, unsigned q,
			const RealVector<4> &m0,
			const RealVector<2> &x0
		);

		static RealVector<4> CL_m0(const RealVector<4> &rx, const RealVector<4> &ry, const RealVector<4> &rz, const RealVector<4> &u0x, const RealVector<4> &u0y, const RealVector<4> &u0z, const RealVector<4> &u1x, const RealVector<4> &u1y, const RealVector<4> &u1z);
		static RealVector<4> LB_1p3(const RealVector<4> &_l);
		static std::array<RealVector<2>, 2> subdiv_0_1p1(const RealVector<2> &_b);
		static std::array<RealVector<4>, 2> subdiv_0_1p3(const RealVector<4> &_b);
		template<unsigned SI=0> std::array<P1TetCGV, schemes[SI]> split_impl() const;
	};

	template<> std::array<P1TetCGV, P1TetCGV::schemes[0]> P1TetCGV::split_impl<0>() const;
}
