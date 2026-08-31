#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P2TriCGV {
		public:
		static constexpr unsigned dimension = 3;
		static constexpr unsigned numVertices = 6;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = true;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P2TriCGV(
			const RealVector<6> &p0x,
			const RealVector<6> &p0y,
			const RealVector<6> &p1x,
			const RealVector<6> &p1y
		);

		P2TriCGV() = default;

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

		static constexpr std::array<unsigned, 2> schemes = {8, 2};

		RealInterval inclusion(unsigned i) const;
		RealVector<numVertices> sample(unsigned i) const;
		template<typename F> void split(unsigned scheme, F &&f) const {
			if (scheme == 1) { for (auto &&_c : split_impl<1>()) f(std::move(_c)); }
			else { for (auto &&_c : split_impl<0>()) f(std::move(_c)); }
		}
		void inherit(const P2TriCGV &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P2TriCGV &s);

		private:

		RealVector<18> m0;
		RealVector<6> x0;
		RealVector<6> x1;
		RealVector<6> x2;

		P2TriCGV(const P2TriCGV &parent, unsigned q,
			const RealVector<18> &m0,
			const RealVector<6> &x0,
			const RealVector<6> &x1,
			const RealVector<6> &x2
		);

		static RealVector<18> CL_m0(const RealVector<6> &p0x, const RealVector<6> &p0y, const RealVector<6> &p1x, const RealVector<6> &p1y);
		static RealVector<18> LB_2p2_1p2(const RealVector<18> &_l);
		static std::array<RealVector<6>, 8> subdiv_0_2p1_1p1(const RealVector<6> &_b);
		static std::array<RealVector<6>, 2> subdiv_1_2p1_1p1(const RealVector<6> &_b);
		static std::array<RealVector<18>, 8> subdiv_0_2p2_1p2(const RealVector<18> &_b);
		static std::array<RealVector<18>, 2> subdiv_1_2p2_1p2(const RealVector<18> &_b);
		template<unsigned SI=0> std::array<P2TriCGV, schemes[SI]> split_impl() const;
	};

	template<> std::array<P2TriCGV, P2TriCGV::schemes[0]> P2TriCGV::split_impl<0>() const;
	template<> std::array<P2TriCGV, P2TriCGV::schemes[1]> P2TriCGV::split_impl<1>() const;
}
