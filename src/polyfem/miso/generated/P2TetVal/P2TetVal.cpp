#include "../P2TetVal.hpp"
namespace miso {
	P2TetVal::P2TetVal(const P2TetVal &parent, unsigned q,
		const RealVector<20> &m0,
		const RealVector<4> &x0,
		const RealVector<4> &x1,
		const RealVector<4> &x2
	) :
	m0(m0),
	x0(x0),
	x1(x1),
	x2(x2)
	{ inherit(parent, q); }

	P2TetVal::P2TetVal(
		const RealVector<10> &px,
		const RealVector<10> &py,
		const RealVector<10> &pz
	) :
	m0(LB_3p3(CL_m0(px, py, pz))),
	x0{0, 0, 0, 1},
	x1{0, 0, 1, 0},
	x2{0, 1, 0, 0}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P2TetVal::inherit(const P2TetVal &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P2TetVal::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P2TetVal::numVertices> P2TetVal::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,3,9,19>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P2TetVal, P2TetVal::schemes[0]> P2TetVal::split_impl<0>() const {
		auto _r_m0 = subdiv_0_3p3(m0);
		auto _r_x0 = subdiv_0_3p1(x0);
		auto _r_x1 = subdiv_0_3p1(x1);
		auto _r_x2 = subdiv_0_3p1(x2);
		return {
			P2TetVal{*this, 0,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0], 
				_r_x2[0]
			},
			P2TetVal{*this, 1,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1], 
				_r_x2[1]
			},
			P2TetVal{*this, 2,
				_r_m0[2], 
				_r_x0[2], 
				_r_x1[2], 
				_r_x2[2]
			},
			P2TetVal{*this, 3,
				_r_m0[3], 
				_r_x0[3], 
				_r_x1[3], 
				_r_x2[3]
			},
			P2TetVal{*this, 4,
				_r_m0[4], 
				_r_x0[4], 
				_r_x1[4], 
				_r_x2[4]
			},
			P2TetVal{*this, 5,
				_r_m0[5], 
				_r_x0[5], 
				_r_x1[5], 
				_r_x2[5]
			},
			P2TetVal{*this, 6,
				_r_m0[6], 
				_r_x0[6], 
				_r_x1[6], 
				_r_x2[6]
			},
			P2TetVal{*this, 7,
				_r_m0[7], 
				_r_x0[7], 
				_r_x1[7], 
				_r_x2[7]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P2TetVal &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
