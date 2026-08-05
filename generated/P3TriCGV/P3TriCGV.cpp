#include "../P3TriCGV.hpp"
namespace miso {
	P3TriCGV::P3TriCGV(const P3TriCGV &parent, unsigned q,
		const RealVector<45> &m0,
		const RealVector<6> &x0,
		const RealVector<6> &x1,
		const RealVector<6> &x2
	) :
	m0(m0),
	x0(x0),
	x1(x1),
	x2(x2)
	{ inherit(parent, q); }

	P3TriCGV::P3TriCGV(
		const RealVector<10> &p0x,
		const RealVector<10> &p0y,
		const RealVector<10> &p1x,
		const RealVector<10> &p1y
	) :
	m0(LB_2p4_1p2(CL_m0(p0x, p0y, p1x, p1y))),
	x0{0, 0, 0, 0, 1, 1},
	x1{0, 0, 1, 1, 0, 0},
	x2{0, 1, 0, 1, 0, 1}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P3TriCGV::inherit(const P3TriCGV &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P3TriCGV::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			case 1: return x2.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P3TriCGV::numVertices> P3TriCGV::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,2,12,14,42,44>();
			case 1: return x2.sample<0,1,2,3,4,5>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P3TriCGV, P3TriCGV::schemes[0]> P3TriCGV::split_impl<0>() const {
		auto _r_m0 = subdiv_0_2p4_1p2(m0);
		auto _r_x0 = subdiv_0_2p1_1p1(x0);
		auto _r_x1 = subdiv_0_2p1_1p1(x1);
		auto _r_x2 = subdiv_0_2p1_1p1(x2);
		return {
			P3TriCGV{*this, 0,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0], 
				_r_x2[0]
			},
			P3TriCGV{*this, 1,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1], 
				_r_x2[1]
			},
			P3TriCGV{*this, 2,
				_r_m0[2], 
				_r_x0[2], 
				_r_x1[2], 
				_r_x2[2]
			},
			P3TriCGV{*this, 3,
				_r_m0[3], 
				_r_x0[3], 
				_r_x1[3], 
				_r_x2[3]
			},
			P3TriCGV{*this, 4,
				_r_m0[4], 
				_r_x0[4], 
				_r_x1[4], 
				_r_x2[4]
			},
			P3TriCGV{*this, 5,
				_r_m0[5], 
				_r_x0[5], 
				_r_x1[5], 
				_r_x2[5]
			},
			P3TriCGV{*this, 6,
				_r_m0[6], 
				_r_x0[6], 
				_r_x1[6], 
				_r_x2[6]
			},
			P3TriCGV{*this, 7,
				_r_m0[7], 
				_r_x0[7], 
				_r_x1[7], 
				_r_x2[7]
			},
		};
	}
	template<> std::array<P3TriCGV, P3TriCGV::schemes[1]> P3TriCGV::split_impl<1>() const {
		auto _r_m0 = subdiv_1_2p4_1p2(m0);
		auto _r_x0 = subdiv_1_2p1_1p1(x0);
		auto _r_x1 = subdiv_1_2p1_1p1(x1);
		auto _r_x2 = subdiv_1_2p1_1p1(x2);
		return {
			P3TriCGV{*this, 8,
				_r_m0[0], 
				_r_x0[0], 
				_r_x1[0], 
				_r_x2[0]
			},
			P3TriCGV{*this, 9,
				_r_m0[1], 
				_r_x0[1], 
				_r_x1[1], 
				_r_x2[1]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P3TriCGV &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
