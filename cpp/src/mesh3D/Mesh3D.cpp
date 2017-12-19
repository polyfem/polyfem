#include "Mesh3D.hpp"

namespace poly_fem{
	bool Mesh3D::load(Mesh &mesh, const std::string &path){
		string filename = path + ".HYBRID";
		FILE *f = fopen(filename.data(), "rt");
		if (!f) return false;

		int nv, np, nh;
		fscanf(f, "%d %d %d", &nv, &np, &nh);
		nh /= 3;

		mesh.V.resize(3, nv); mesh.Vs.resize(nv);
		for (int i = 0; i<nv; i++) {
			double x, y, z;
			fscanf(f, "%lf %lf %lf", &x, &y, &z);
			mesh.V(0, i) = x;
			mesh.V(1, i) = y;
			mesh.V(2, i) = z;
			Hybrid_V v;
			v.id = i;
			mesh.Vs[i] = v;
		}
		mesh.Fs.resize(np);
		for (int i = 0; i<np; i++) {
			Hybrid_F &p = mesh.Fs[i];
			p.id = i;
			int nw;

			fscanf(f, "%d", &nw);
			p.vs.resize(nw);
			for (int j = 0; j<nw; j++) {
				fscanf(f, "%d", &(p.vs[j]));
			}
		}
		mesh.Hs.resize(nh);
		for (int i = 0; i<nh; i++) {
			Hybrid &h = mesh.Hs[i];
			h.id = i;

			int nf;
			fscanf(f, "%d", &nf);
			h.fs.resize(nf);

			for (int j = 0; j<nf; j++) {
				fscanf(f, "%d", &(h.fs[j]));
			}

			for (auto fid : h.fs)h.vs.insert(h.vs.end(), mesh.Fs[fid].vs.begin(), mesh.Fs[fid].vs.end());
			sort(h.vs.begin(), h.vs.end()); h.vs.erase(unique(h.vs.begin(), h.vs.end()), h.vs.end());

			int tmp; fscanf(f, "%d", &tmp);
			for (int j = 0; j<nf; j++) {
				int s;
				fscanf(f, "%d", &s);
				h.fs_flag.push_back(s);
			}
		}
		for (int i = 0; i<nh; i++) {
			int tmp;
			fscanf(f, "%d", &(mesh.Hs[i].hex));
		}

		fclose(f);

		Navigation3D::prepare_mesh(mesh);
		return true;
	}

	bool Mesh3D::save(Mesh &mesh, const std::string &path) const{

		std::fstream f(path, std::ios::out);

		f << mesh.V.cols() << " " << mesh.Fs.size() << " " << 3 * mesh.Hs.size() << std::endl;
		for (int i = 0; i<mesh.V.cols(); i++)
			f << mesh.V(0, i) << " " << mesh.V(1, i) << " " << mesh.V(2, i) << std::endl;

		for (auto f_ : mesh.Fs) {
			f << f_.vs.size() << " ";
			for (auto vid : f_.vs)
				f << vid << " ";
			f << std::endl;
		}

		for (uint32_t i = 0; i < mesh.Hs.size(); i++) {
			f << mesh.Hs[i].fs.size() << " ";
			for (auto fid : mesh.Hs[i].fs)
				f << fid << " ";
			f << std::endl;
			f << mesh.Hs[i].fs_flag.size() << " ";
			for (auto f_flag : mesh.Hs[i].fs_flag)
				f << f_flag << " ";
			f << std::endl;
		}

		for (uint32_t i = 0; i < mesh.Hs.size(); i++) {
			f << mesh.Hs[i].hex << std::endl;
		}

		f.close();

		return true;
	}

}
