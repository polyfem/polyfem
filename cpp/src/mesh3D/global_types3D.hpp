#pragma once
#include <cstdlib>
#include <vector>
#include "Eigen/Dense"
using namespace Eigen;
using namespace std;

/*typedefs*/
#if defined(SINGLE_PRECISION)
typedef float Float;
#else
typedef double Float;
#endif

//-------------------------------------------------------------------
//---For Hybrid mesh-------------------------------------------------
struct Hybrid_V
{
	uint32_t id;
	vector<Float> v;
	vector<uint32_t> neighbor_vs;
	vector<uint32_t> neighbor_es;
	vector<uint32_t> neighbor_fs;
	vector<uint32_t> neighbor_hs;

	bool boundary;
};
struct Hybrid_E
{
	uint32_t id;
	vector<uint32_t> vs;
	vector<uint32_t> neighbor_fs;
	vector<uint32_t> neighbor_hs;
	
	bool boundary;
};
struct Hybrid_F
{
	uint32_t id;
	vector<uint32_t> vs;
	vector<uint32_t> es;
	vector<uint32_t> neighbor_hs;
	bool boundary;
};

struct Hybrid
{
	uint32_t id;
	vector<uint32_t> vs;
	vector<uint32_t> es;
	vector<uint32_t> fs;
	vector<bool> fs_flag;
	bool hex = false;
};
//-------------------------------------------------------------------
//-------------------------------------------------------------------
enum Mesh_type {
	Tri = 0,
	Qua,
	Tet,
	Hyb,
	Hex,
	PHr
};

struct Mesh
{
	short type;//Mesh_type
	MatrixXd V;
	vector<Hybrid_V> Vs;
	vector<Hybrid_E> Es;
	vector<Hybrid_F> Fs;
	vector<Hybrid> Hs;
};
