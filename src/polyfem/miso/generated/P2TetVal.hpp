#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P2TetVal {
		public:
		static constexpr unsigned dimension = 3;
		static constexpr unsigned numVertices = 4;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = false;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P2TetVal(
			const RealVector<10> &px,
			const RealVector<10> &py,
			const RealVector<10> &pz
		);

		P2TetVal() = default;

		unsigned schemeSize() const {
			auto s = (scheme < schemes.size()) ? scheme : 0;
			return schemes.at(s);
		}

		const Real width() const {
			return std::max({
				x0.inclusion().width(),
				x1.inclusion().width(),
				x2.inclusion().width()
			});
		}

		RealVector<dimension> getVertex(unsigned i) const {
			return {
				x0[i],
				x1[i],
				x2[i]
			};
		}

		static constexpr std::array<unsigned, 1> schemes = {8};

		RealInterval inclusion(unsigned i) const;
		RealVector<numVertices> sample(unsigned i) const;
		template<typename F> void split(unsigned scheme, F &&f) const {
			for (auto &&_c : split_impl<0>()) f(std::move(_c));
		}
		void inherit(const P2TetVal &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P2TetVal &s);

		private:

		RealVector<20> m0;
		RealVector<4> x0;
		RealVector<4> x1;
		RealVector<4> x2;

		P2TetVal(const P2TetVal &parent, unsigned q,
			const RealVector<20> &m0,
			const RealVector<4> &x0,
			const RealVector<4> &x1,
			const RealVector<4> &x2
		);

		static RealVector<20> CL_m0(const RealVector<10> &px, const RealVector<10> &py, const RealVector<10> &pz);
		static RealVector<20> LB_3p3(const RealVector<20> &_l);
		static std::array<RealVector<4>, 8> subdiv_0_3p1(const RealVector<4> &_b);
		static std::array<RealVector<20>, 8> subdiv_0_3p3(const RealVector<20> &_b);
		template<unsigned SI=0> std::array<P2TetVal, schemes[SI]> split_impl() const;
	};

	template<> std::array<P2TetVal, P2TetVal::schemes[0]> P2TetVal::split_impl<0>() const;
}
