#include "../P2TriVal.hpp"
namespace miso {
	P2TriVal::P2TriVal(const P2TriVal &parent, unsigned q,
		const RealVector<6> &m0,
		const RealVector<3> &x0,
		const RealVector<3> &x1
	) :
	m0(m0),
	x0(x0),
	x1(x1)
	{ inherit(parent, q); }

	P2TriVal::P2TriVal(
		const RealVector<6> &px,
		const RealVector<6> &py
	) :
	m0(LB_2p2(CL_m0(px, py))),
	x0{0, 0, 1},
	x1{0, 1, 0}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P2TriVal::inherit(const P2TriVal &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P2TriVal::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P2TriVal::numVertices> P2TriVal::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,2,5>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P2TriVal, P2TriVal::schemes[0]> P2TriVal::split_impl<0>() const {
		auto _r_m0 = subdiv_0_2p2(m0);
		auto _r_x0 = subdiv_0_2p1(x0);
		auto _r_x1 = subdiv_0_2p1(x1);
		return {
			P2TriVal{*this, 0,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0]
			},
			P2TriVal{*this, 1,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1]
			},
			P2TriVal{*this, 2,
				_r_m0[2], 
				_r_x0[2], 
				_r_x1[2]
			},
			P2TriVal{*this, 3,
				_r_m0[3], 
				_r_x0[3], 
				_r_x1[3]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P2TriVal &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
