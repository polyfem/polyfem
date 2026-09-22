#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P2TriVal {
		public:
		static constexpr unsigned dimension = 2;
		static constexpr unsigned numVertices = 3;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = false;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P2TriVal(
			const RealVector<6> &px,
			const RealVector<6> &py
		);

		P2TriVal() = default;

		unsigned schemeSize() const {
			auto s = (scheme < schemes.size()) ? scheme : 0;
			return schemes.at(s);
		}

		const Real width() const {
			return std::max({
				x0.inclusion().width(),
				x1.inclusion().width()
			});
		}

		RealVector<dimension> getVertex(unsigned i) const {
			return {
				x0[i],
				x1[i]
			};
		}

		static constexpr std::array<unsigned, 1> schemes = {4};

		RealInterval inclusion(unsigned i) const;
		RealVector<numVertices> sample(unsigned i) const;
		template<typename F> void split(unsigned scheme, F &&f) const {
			for (auto &&_c : split_impl<0>()) f(std::move(_c));
		}
		void inherit(const P2TriVal &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P2TriVal &s);

		private:

		RealVector<6> m0;
		RealVector<3> x0;
		RealVector<3> x1;

		P2TriVal(const P2TriVal &parent, unsigned q,
			const RealVector<6> &m0,
			const RealVector<3> &x0,
			const RealVector<3> &x1
		);

		static RealVector<6> CL_m0(const RealVector<6> &px, const RealVector<6> &py);
		static RealVector<6> LB_2p2(const RealVector<6> &_l);
		static std::array<RealVector<3>, 4> subdiv_0_2p1(const RealVector<3> &_b);
		static std::array<RealVector<6>, 4> subdiv_0_2p2(const RealVector<6> &_b);
		template<unsigned SI=0> std::array<P2TriVal, schemes[SI]> split_impl() const;
	};

	template<> std::array<P2TriVal, P2TriVal::schemes[0]> P2TriVal::split_impl<0>() const;
}
