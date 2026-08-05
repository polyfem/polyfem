#pragma once
#include "core/RealVector.hpp"
#include "core/SubdivHistory.hpp"

namespace miso {
	class P3TriVal {
		public:
		static constexpr unsigned dimension = 2;
		static constexpr unsigned numVertices = 3;
		static constexpr unsigned numConstraints = 1;
		static constexpr bool hasObjective = false;
		unsigned scheme = 0;
		unsigned depth = 0;
		std::shared_ptr<const SubdivHistory> history;

		P3TriVal(
			const RealVector<10> &px,
			const RealVector<10> &py
		);

		P3TriVal() = default;

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
		void inherit(const P3TriVal &parent, unsigned q);
		friend std::ostream &operator<<(std::ostream &out, const P3TriVal &s);

		private:

		RealVector<15> m0;
		RealVector<3> x0;
		RealVector<3> x1;

		P3TriVal(const P3TriVal &parent, unsigned q,
			const RealVector<15> &m0,
			const RealVector<3> &x0,
			const RealVector<3> &x1
		);

		static RealVector<15> CL_m0(const RealVector<10> &px, const RealVector<10> &py);
		static RealVector<15> LB_2p4(const RealVector<15> &_l);
		static std::array<RealVector<3>, 4> subdiv_0_2p1(const RealVector<3> &_b);
		static std::array<RealVector<15>, 4> subdiv_0_2p4(const RealVector<15> &_b);
		template<unsigned SI=0> std::array<P3TriVal, schemes[SI]> split_impl() const;
	};

	template<> std::array<P3TriVal, P3TriVal::schemes[0]> P3TriVal::split_impl<0>() const;
}
