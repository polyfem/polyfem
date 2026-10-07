#include "../P4TriVal.hpp"
namespace miso {
	P4TriVal::P4TriVal(const P4TriVal &parent, unsigned q,
		const RealVector<28> &m0,
		const RealVector<3> &x0,
		const RealVector<3> &x1
	) :
	m0(m0),
	x0(x0),
	x1(x1)
	{ inherit(parent, q); }

	P4TriVal::P4TriVal(
		const RealVector<15> &px,
		const RealVector<15> &py
	) :
	m0(LB_2p6(CL_m0(px, py))),
	x0{0, 0, 1},
	x1{0, 1, 0}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P4TriVal::inherit(const P4TriVal &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P4TriVal::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P4TriVal::numVertices> P4TriVal::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,6,27>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P4TriVal, P4TriVal::schemes[0]> P4TriVal::split_impl<0>() const {
		auto _r_m0 = subdiv_0_2p6(m0);
		auto _r_x0 = subdiv_0_2p1(x0);
		auto _r_x1 = subdiv_0_2p1(x1);
		return {
			P4TriVal{*this, 0,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0]
			},
			P4TriVal{*this, 1,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1]
			},
			P4TriVal{*this, 2,
				_r_m0[2], 
				_r_x0[2], 
				_r_x1[2]
			},
			P4TriVal{*this, 3,
				_r_m0[3], 
				_r_x0[3], 
				_r_x1[3]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P4TriVal &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
