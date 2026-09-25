#include "../P1TetCGV.hpp"
namespace miso {
	P1TetCGV::P1TetCGV(const P1TetCGV &parent, unsigned q,
		const RealVector<4> &m0,
		const RealVector<2> &x0
	) :
	m0(m0),
	x0(x0)
	{ inherit(parent, q); }

	P1TetCGV::P1TetCGV(
		const RealVector<4> &rx,
		const RealVector<4> &ry,
		const RealVector<4> &rz,
		const RealVector<4> &u0x,
		const RealVector<4> &u0y,
		const RealVector<4> &u0z,
		const RealVector<4> &u1x,
		const RealVector<4> &u1y,
		const RealVector<4> &u1z
	) :
	m0(LB_1p3(CL_m0(rx, ry, rz, u0x, u0y, u0z, u1x, u1y, u1z))),
	x0{0, 1}
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
			case 1: return x0.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P1TetCGV::numVertices> P1TetCGV::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,3>();
			case 1: return x0.sample<0,1>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P1TetCGV, P1TetCGV::schemes[0]> P1TetCGV::split_impl<0>() const {
		auto _r_m0 = subdiv_0_1p3(m0);
		auto _r_x0 = subdiv_0_1p1(x0);
		return {
			P1TetCGV{*this, 0,
				_r_m0[0], 
				_r_x0[0]
			},
			P1TetCGV{*this, 1,
				_r_m0[1], 
				_r_x0[1]
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
