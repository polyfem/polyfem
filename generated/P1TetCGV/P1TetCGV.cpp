#include "../P1TetCGV.hpp"
namespace miso {
	P1TetCGV::P1TetCGV(const P1TetCGV &parent, unsigned q,
		const RealVector<4> &m0,
		const RealVector<8> &x0,
		const RealVector<8> &x1,
		const RealVector<8> &x2,
		const RealVector<8> &x3
	) :
	m0(m0),
	x0(x0),
	x1(x1),
	x2(x2),
	x3(x3)
	{ inherit(parent, q); }

	P1TetCGV::P1TetCGV(
		const RealVector<4> &p0x,
		const RealVector<4> &p0y,
		const RealVector<4> &p0z,
		const RealVector<4> &p1x,
		const RealVector<4> &p1y,
		const RealVector<4> &p1z
	) :
	m0(LB_1p3(CL_m0(p0x, p0y, p0z, p1x, p1y, p1z))),
	x0{0, 0, 0, 0, 0, 0, 1, 1},
	x1{0, 0, 0, 0, 1, 1, 0, 0},
	x2{0, 0, 1, 1, 0, 0, 0, 0},
	x3{0, 1, 0, 1, 0, 1, 0, 1}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P1TetCGV::inherit(const P1TetCGV &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P1TetCGV::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			case 1: return x3.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P1TetCGV::numVertices> P1TetCGV::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,3,0,3,0,3,0,3>();
			case 1: return x3.sample<0,1,2,3,4,5,6,7>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P1TetCGV, P1TetCGV::schemes[0]> P1TetCGV::split_impl<0>() const {
		auto _r_m0 = subdiv_0_1p3(m0);
		auto _r_x0 = subdiv_0_3p1_1p1(x0);
		auto _r_x1 = subdiv_0_3p1_1p1(x1);
		auto _r_x2 = subdiv_0_3p1_1p1(x2);
		auto _r_x3 = subdiv_0_3p1_1p1(x3);
		return {
			P1TetCGV{*this, 0,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0], 
				_r_x2[0], 
				_r_x3[0]
			},
			P1TetCGV{*this, 1,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1], 
				_r_x2[1], 
				_r_x3[1]
			},
			P1TetCGV{*this, 2,
				_r_m0[0], 
				_r_x0[2], 
				_r_x1[2], 
				_r_x2[2], 
				_r_x3[2]
			},
			P1TetCGV{*this, 3,
				_r_m0[1], 
				_r_x0[3], 
				_r_x1[3], 
				_r_x2[3], 
				_r_x3[3]
			},
			P1TetCGV{*this, 4,
				_r_m0[0], 
				_r_x0[4], 
				_r_x1[4], 
				_r_x2[4], 
				_r_x3[4]
			},
			P1TetCGV{*this, 5,
				_r_m0[1], 
				_r_x0[5], 
				_r_x1[5], 
				_r_x2[5], 
				_r_x3[5]
			},
			P1TetCGV{*this, 6,
				_r_m0[0], 
				_r_x0[6], 
				_r_x1[6], 
				_r_x2[6], 
				_r_x3[6]
			},
			P1TetCGV{*this, 7,
				_r_m0[1], 
				_r_x0[7], 
				_r_x1[7], 
				_r_x2[7], 
				_r_x3[7]
			},
			P1TetCGV{*this, 8,
				_r_m0[0], 
				_r_x0[8], 
				_r_x1[8], 
				_r_x2[8], 
				_r_x3[8]
			},
			P1TetCGV{*this, 9,
				_r_m0[1], 
				_r_x0[9], 
				_r_x1[9], 
				_r_x2[9], 
				_r_x3[9]
			},
			P1TetCGV{*this, 10,
				_r_m0[0], 
				_r_x0[10], 
				_r_x1[10], 
				_r_x2[10], 
				_r_x3[10]
			},
			P1TetCGV{*this, 11,
				_r_m0[1], 
				_r_x0[11], 
				_r_x1[11], 
				_r_x2[11], 
				_r_x3[11]
			},
			P1TetCGV{*this, 12,
				_r_m0[0], 
				_r_x0[12], 
				_r_x1[12], 
				_r_x2[12], 
				_r_x3[12]
			},
			P1TetCGV{*this, 13,
				_r_m0[1], 
				_r_x0[13], 
				_r_x1[13], 
				_r_x2[13], 
				_r_x3[13]
			},
			P1TetCGV{*this, 14,
				_r_m0[0], 
				_r_x0[14], 
				_r_x1[14], 
				_r_x2[14], 
				_r_x3[14]
			},
			P1TetCGV{*this, 15,
				_r_m0[1], 
				_r_x0[15], 
				_r_x1[15], 
				_r_x2[15], 
				_r_x3[15]
			},
		};
	}
	template<> std::array<P1TetCGV, P1TetCGV::schemes[1]> P1TetCGV::split_impl<1>() const {
		auto _r_m0 = subdiv_1_1p3(m0);
		auto _r_x0 = subdiv_1_3p1_1p1(x0);
		auto _r_x1 = subdiv_1_3p1_1p1(x1);
		auto _r_x2 = subdiv_1_3p1_1p1(x2);
		auto _r_x3 = subdiv_1_3p1_1p1(x3);
		return {
			P1TetCGV{*this, 16,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0], 
				_r_x2[0], 
				_r_x3[0]
			},
			P1TetCGV{*this, 17,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1], 
				_r_x2[1], 
				_r_x3[1]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P1TetCGV &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
