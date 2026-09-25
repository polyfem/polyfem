#include "../P1TriCGV.hpp"
namespace miso {
	P1TriCGV::P1TriCGV(const P1TriCGV &parent, unsigned q,
		const RealVector<3> &m0,
		const RealVector<2> &x0
	) :
	m0(m0),
	x0(x0)
	{ inherit(parent, q); }

	P1TriCGV::P1TriCGV(
		const RealVector<3> &rx,
		const RealVector<3> &ry,
		const RealVector<3> &u0x,
		const RealVector<3> &u0y,
		const RealVector<3> &u1x,
		const RealVector<3> &u1y
	) :
	m0(LB_1p2(CL_m0(rx, ry, u0x, u0y, u1x, u1y))),
	x0{0, 1}
	{
		history = std::make_shared<SubdivHistory>();
	}

	void P1TriCGV::inherit(const P1TriCGV &parent, unsigned q) {
		depth = parent.depth + 1;
		history = std::make_shared<SubdivHistory>(q, parent.history);
	}
	RealInterval P1TriCGV::inclusion(unsigned i) const {
		switch (i) {
			case 0: return m0.inclusion();
			case 1: return x0.inclusion();
			default: throw std::logic_error("Undefined inclusion function");
		}
	}
	RealVector<P1TriCGV::numVertices> P1TriCGV::sample(unsigned i) const {
		switch (i) {
			case 0: return m0.sample<0,2>();
			case 1: return x0.sample<0,1>();
			default: throw std::logic_error("Undefined sampling function");
		}
	}
	template<> std::array<P1TriCGV, P1TriCGV::schemes[0]> P1TriCGV::split_impl<0>() const {
		auto _r_m0 = subdiv_0_1p2(m0);
		auto _r_x0 = subdiv_0_1p1(x0);
		return {
			P1TriCGV{*this, 0,
				_r_m0[0], 
				_r_x0[0]
			},
			P1TriCGV{*this, 1,
				_r_m0[1], 
				_r_x0[1]
			},
		};
	}
	std::ostream &operator<<(std::ostream &out, const P1TriCGV &s) {
		for (unsigned i=0; i<s.numVertices; ++i) {
			out << s.getVertex(i) << ' ';
		}
		return out;
	}
}
