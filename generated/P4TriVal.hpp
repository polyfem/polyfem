#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P4TriVal {
		public:
		static constexpr unsigned dimension = 2;
		static constexpr unsigned numVertices = 3;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = false;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P4TriVal(
			const RealVector<15> &px,
			const RealVector<15> &py
		);

		P4TriVal() = default;

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
		void inherit(const P4TriVal &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P4TriVal &s);

		private:

		RealVector<28> m0;
		RealVector<3> x0;
		RealVector<3> x1;

		P4TriVal(const P4TriVal &parent, unsigned q,
			const RealVector<28> &m0,
			const RealVector<3> &x0,
			const RealVector<3> &x1
		);

		static RealVector<28> CL_m0(const RealVector<15> &px, const RealVector<15> &py);
		static RealVector<28> LB_2p6(const RealVector<28> &_l);
		static std::array<RealVector<3>, 4> subdiv_0_2p1(const RealVector<3> &_b);
		static std::array<RealVector<28>, 4> subdiv_0_2p6(const RealVector<28> &_b);
		template<unsigned SI=0> std::array<P4TriVal, schemes[SI]> split_impl() const;
	};

	template<> std::array<P4TriVal, P4TriVal::schemes[0]> P4TriVal::split_impl<0>() const;
}
