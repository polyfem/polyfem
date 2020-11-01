
// void State::compute_poly_basis_error(const std::string &path)
// {

// 	MatrixXd fun = MatrixXd::Zero(n_bases, 1);
// 	MatrixXd tmp, mapped;
// 	MatrixXd v_approx, v_exact;

// 	int poly_index = -1;

// 	for(size_t i = 0; i < bases.size(); ++i)
// 	{
// 		const ElementBases &basis = bases[i];
// 		if(!basis.has_parameterization){
// 			poly_index = i;
// 			continue;
// 		}

// 		for(std::size_t j = 0; j < basis.bases.size(); ++j)
// 		{
// 			for(std::size_t kk = 0; kk < basis.bases[j].global().size(); ++kk)
// 			{
// 				const Local2Global &l2g = basis.bases[j].global()[kk];
// 				const int g_index = l2g.index;

// 				const auto &node = l2g.node;
// 				problem->exact(node, tmp);

// 				fun(g_index) = tmp(0);
// 			}
// 		}
// 	}

// 	if(poly_index == -1)
// 		poly_index = 0;

// 	auto &poly_basis = bases[poly_index];
// 	ElementAssemblyValues vals;
// 	vals.compute(poly_index, true, poly_basis, poly_basis);

// 	// problem.exact(vals.val, v_exact);
// 	v_exact.resize(vals.val.rows(), vals.val.cols());
// 	dx(vals.val, tmp); v_exact.col(0) = tmp;
// 	dy(vals.val, tmp); v_exact.col(1) = tmp;
// 	dz(vals.val, tmp); v_exact.col(2) = tmp;

// 	v_approx = MatrixXd::Zero(v_exact.rows(), v_exact.cols());

// 	const int n_loc_bases=int(vals.basis_values.size());

// 	for(int i = 0; i < n_loc_bases; ++i)
// 	{
// 		auto &val=vals.basis_values[i];

// 		for(std::size_t ii = 0; ii < val.global.size(); ++ii)
// 		{
// 			// v_approx += val.global[ii].val * fun(val.global[ii].index) * val.val;
// 			v_approx += val.global[ii].val * fun(val.global[ii].index) * val.grad;
// 		}
// 	}

// 	const Eigen::MatrixXd err = (v_exact-v_approx).cwiseAbs();

// 	using json = nlohmann::json;
// 	json j;
// 	j["mesh_path"] = mesh_path;

// 	for(long c = 0; c < v_approx.cols();++c){
// 		double l2_err_interp = 0;
// 		double lp_err_interp = 0;

// 		l2_err_interp += (err.col(c).array() * err.col(c).array() * vals.det.array() * vals.quadrature.weights.array()).sum();
// 		lp_err_interp += (err.col(c).array().pow(8.) * vals.det.array() * vals.quadrature.weights.array()).sum();

// 		l2_err_interp = sqrt(fabs(l2_err_interp));
// 		lp_err_interp = pow(fabs(lp_err_interp), 1./8.);

// 		j["err_l2_"+std::to_string(c)] = l2_err_interp;
// 		j["err_lp_"+std::to_string(c)] = lp_err_interp;
// 	}

// 	std::ofstream out(path);
// 	out << j.dump(4) << std::endl;
// 	out.close();
// }