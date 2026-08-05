#include "../P3TetVal.hpp"
namespace miso {
	P3TetVal::P3TetVal(const P3TetVal &parent, unsigned q,
		const RealVector<84> &m0,
		const RealVector<4> &x0,
		const RealVector<4> &x1,
		const RealVector<4> &x2
	) :
	m0(m0),
	x0(x0),
	x1(x1),
	x2(x2)
	{ inherit(parent, q); }

	P3TetVal::P3TetVal(
		const RealVector<20> &px,
		const RealVector<20> &py,
		const RealVector<20> &pz
	) :
	m0(LB_3p6(CL_m0(px, py, pz))),
	x0{0, 0, 0, 1},
	x1{0, 0, 1, 0},
	x2{0, 1, 0, 0}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P3TetVal::inherit(const P3TetVal &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P3TetVal::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P3TetVal::numVertices> P3TetVal::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,6,27,83>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P3TetVal, P3TetVal::schemes[0]> P3TetVal::split_impl<0>() const {
		auto _r_m0 = subdiv_0_3p6(m0);
		auto _r_x0 = subdiv_0_3p1(x0);
		auto _r_x1 = subdiv_0_3p1(x1);
		auto _r_x2 = subdiv_0_3p1(x2);
		return {
			P3TetVal{*this, 0,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0], 
				_r_x2[0]
			},
			P3TetVal{*this, 1,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1], 
				_r_x2[1]
			},
			P3TetVal{*this, 2,
				_r_m0[2], 
				_r_x0[2], 
				_r_x1[2], 
				_r_x2[2]
			},
			P3TetVal{*this, 3,
				_r_m0[3], 
				_r_x0[3], 
				_r_x1[3], 
				_r_x2[3]
			},
			P3TetVal{*this, 4,
				_r_m0[4], 
				_r_x0[4], 
				_r_x1[4], 
				_r_x2[4]
			},
			P3TetVal{*this, 5,
				_r_m0[5], 
				_r_x0[5], 
				_r_x1[5], 
				_r_x2[5]
			},
			P3TetVal{*this, 6,
				_r_m0[6], 
				_r_x0[6], 
				_r_x1[6], 
				_r_x2[6]
			},
			P3TetVal{*this, 7,
				_r_m0[7], 
				_r_x0[7], 
				_r_x1[7], 
				_r_x2[7]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P3TetVal &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
