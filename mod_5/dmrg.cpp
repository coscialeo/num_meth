#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/KroneckerProduct>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/DavidsonSymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <algorithm>
#include <vector>
#include <map>
#include <string>

#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace Spectra;


struct bloc {
    int length, basis_size;
    SparseMatrix<double> ham, conn_sz, conn_sp, fixed_sz;
    vector<int> sz_of_idx;
};

struct result {
    bloc bl;
    double energy, truncation_error, entropy, correlation;
    int dist;
    MatrixXd transformation_matrix, gs_vector;
};

vector<double> energies, truncation_errors, entropies, correlations;

map<pair<string, int>, bloc> bloc_disk;
map<pair<string, int>, MatrixXd> trmat_disk;

double delta;
int num_sites, m, fixed_pos;

int num_sweeps, num_threads;
bool started = false;

bool verbose = true;
double total_ham_time = 0.0;
double total_rho_time = 0.0;

const int site_dim = 2;

const SparseMatrix<double> sz = [] {
    SparseMatrix<double> mat(2, 2);
    mat.insert(0, 0) = 1;
    mat.insert(1, 1) = -1;

    mat.makeCompressed();
    return mat;
}();

const SparseMatrix<double> sp = [] {
    SparseMatrix<double> mat(2, 2);
    mat.insert(0, 1) = 1;

    mat.makeCompressed();
    return mat;
}();

const SparseMatrix<double> site_ham = [] {
    SparseMatrix<double> mat(2, 2);

    mat.makeCompressed();
    return mat;
}();

const bloc initial_bloc = {
    1,
    site_dim,
    site_ham,
    sz,
    sp,
    sz,
    {1, -1}
};

const MatrixXd none(0, 0);

SparseMatrix<double> int_ham(const SparseMatrix<double>& sz_1, const SparseMatrix<double>& sp_1, const SparseMatrix<double>& sz_2, const SparseMatrix<double>& sp_2) {
    SparseMatrix<double> mat = - ( 2 * KroneckerProductSparse(sp_1, sp_2.transpose()) + 2 * KroneckerProductSparse(sp_1.transpose(), sp_2) + delta * KroneckerProductSparse(sz_1, sz_2));
    mat.makeCompressed();
    return mat;
}

bloc enlarge_bloc(const bloc& bl) {

    int m = bl.basis_size;
    SparseMatrix<double> id_m(m, m), id_site(site_dim, site_dim);
    id_site.setIdentity();
    id_m.setIdentity();

    SparseMatrix<double> enl_ham = KroneckerProductSparse(bl.ham, id_site) + KroneckerProductSparse(id_m, site_ham) + int_ham(bl.conn_sz, bl.conn_sp, sz, sp);
    enl_ham.makeCompressed();
    SparseMatrix<double> enl_conn_sz = KroneckerProductSparse(id_m, sz);
    enl_conn_sz.makeCompressed();
    SparseMatrix<double> enl_conn_sp = KroneckerProductSparse(id_m, sp);
    enl_conn_sp.makeCompressed();
    SparseMatrix<double> enl_fixed_sz = KroneckerProductSparse(bl.fixed_sz, id_site);
    enl_fixed_sz.makeCompressed();

    vector<int> sz_of_new_idx(m * site_dim);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < site_dim; ++j)
            sz_of_new_idx[i * site_dim + j] = bl.sz_of_idx[i] + sz.coeff(j, j);

    return {bl.length + 1, m * site_dim, enl_ham, enl_conn_sz, enl_conn_sp, enl_fixed_sz, sz_of_new_idx};
}

SparseMatrix<double> rot_trunc(const SparseMatrix<double>& op, const SparseMatrix<double>& transf) {
    SparseMatrix<double> mat = transf.transpose() * op * transf;
    mat.makeCompressed();
    return mat;
}

result dmrg_step(bloc sys, const bloc& env, int m, const string& sys_label, ostream& out_stream, bool finite = false, const MatrixXd& guess = none) {

    if (verbose) out_stream << endl << "Preprocessing blocks..." << endl;

    bloc e_sys = enlarge_bloc(sys);
    bloc e_env = enlarge_bloc(env);

    vector<int> sector_of_sb_idx(e_sys.basis_size * e_env.basis_size);
    vector<int> sector_idx_of_sb_idx(e_sys.basis_size * e_env.basis_size);

    map<int, vector<int>> sb_idxs_in_sector;

    for (int e_sys_idx = 0; e_sys_idx < e_sys.basis_size; ++e_sys_idx) {
        int e_sys_sz = e_sys.sz_of_idx[e_sys_idx];

        for (int e_env_idx = 0; e_env_idx < e_env.basis_size; ++e_env_idx) {
            int e_env_sz = e_env.sz_of_idx[e_env_idx];

            int sb_idx = e_env.basis_size * e_sys_idx + e_env_idx;
            int sector = e_sys_sz + e_env_sz;

            sector_of_sb_idx[sb_idx] = sector;
            sector_idx_of_sb_idx[sb_idx] = sb_idxs_in_sector[sector].size();

            sb_idxs_in_sector[sector].push_back(sb_idx);
        }
    }

    if (verbose) {
        out_stream << "Blocks preprocessed." << endl;
        out_stream << endl << "Creating super-block..." << endl;
    }

    int sb_dim = e_sys.basis_size * e_env.basis_size;

    SparseMatrix<double> id_e_sys(e_sys.basis_size, e_sys.basis_size);
    SparseMatrix<double> id_e_env(e_env.basis_size, e_env.basis_size);
    id_e_sys.setIdentity();
    id_e_env.setIdentity();

    SparseMatrix<double> super_ham = (KroneckerProductSparse(e_sys.ham, id_e_env) + KroneckerProductSparse(id_e_sys, e_env.ham) + int_ham(e_sys.conn_sz, e_sys.conn_sp, e_env.conn_sz, e_env.conn_sp)) / double(e_sys.length + e_env.length);
    super_ham.makeCompressed();

    map<int, SparseMatrix<double>> super_ham_of_sector;
    for (const auto& [sector, sb_idxs] : sb_idxs_in_sector) {
        super_ham_of_sector[sector].resize(sb_idxs.size(), sb_idxs.size());
        super_ham_of_sector[sector].reserve(VectorXi::Constant(sb_idxs.size(), 3));
    }

    for (int k = 0; k < super_ham.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(super_ham, k); it; ++it) {

            int sector = sector_of_sb_idx[it.row()];
            int sector_row = sector_idx_of_sb_idx[it.row()];
            int sector_col = sector_idx_of_sb_idx[it.col()];

            super_ham_of_sector[sector].insert(sector_row, sector_col) = it.value();
        }
    }

    if (verbose) {
        out_stream << "Super-block created." << endl;
        out_stream << endl << "Diagonalizing super-block Hamiltonian..." << endl;
    }

    auto ham_start = high_resolution_clock::now();

    map<int, double> sector_energies;
    map<int, MatrixXd> sector_gs;

    vector<int> sector_keys;
    for (const auto& kv : super_ham_of_sector)
        sector_keys.push_back(kv.first);

    vector<map<int, double>> thread_energies(num_threads);
    vector<map<int, MatrixXd>> thread_gs(num_threads);

    #pragma omp parallel for num_threads(num_threads)
    for (int idx = 0; idx < sector_keys.size(); ++idx) {
        int thread_id = omp_get_thread_num();
        int sector = sector_keys[idx];
        SparseMatrix<double>& sector_ham = super_ham_of_sector[sector];

        sector_ham.makeCompressed();
        int sector_dim = sector_ham.rows();

        MatrixXd sector_guess;
        if (finite) {
            sector_guess.resize(sector_dim, 1);
            for (int i = 0; i < sector_dim; ++i)
                sector_guess(i, 0) = guess(sb_idxs_in_sector[sector][i], 0);
        }

        if (sector_dim < 512) {
            SelfAdjointEigenSolver<SparseMatrix<double>> eigs_ham(sector_ham);
            thread_energies[thread_id][sector] = eigs_ham.eigenvalues()(0);
            thread_gs[thread_id][sector] = eigs_ham.eigenvectors().col(0);
        } else {
            SparseSymMatProd<double> op_ham(sector_ham);
            DavidsonSymEigsSolver<SparseSymMatProd<double>> eigs_ham(op_ham, 1);

            if (!finite)
                eigs_ham.compute(SortRule::SmallestAlge, 50000, 1e-10);
            else
                eigs_ham.compute_with_guess(sector_guess, SortRule::SmallestAlge, 50000, 1e-10);

            if (eigs_ham.info() != Spectra::CompInfo::Successful) {
                out_stream << "Spectra failed to converge in sector " << sector << ". Info: " << static_cast<int>(eigs_ham.info()) << endl;
                exit(EXIT_FAILURE);
            }
            thread_energies[thread_id][sector] = eigs_ham.eigenvalues()(0);
            thread_gs[thread_id][sector] = eigs_ham.eigenvectors().col(0);
        }
    }

    for (const auto& local_map : thread_energies)
        for (const auto& kv : local_map)
            sector_energies[kv.first] = kv.second;

    for (const auto& local_map : thread_gs)
        for (const auto& kv : local_map)
            sector_gs[kv.first] = kv.second;

    auto ham_end = high_resolution_clock::now();
    total_ham_time += duration<double>(ham_end - ham_start).count();

    if (verbose) {
        out_stream << "Super-block Hamiltonian diagonalized." << endl;
        out_stream << endl << "Creating reduced density matrix..." << endl;
    }
    
    int gs_sector = distance(sector_energies.begin(), min_element(sector_energies.begin(), sector_energies.end()));
    int gs_sector_dim = super_ham_of_sector[gs_sector].rows();
    int my_m = min(m, e_sys.basis_size);
    double gs_energy = sector_energies[gs_sector];

    SparseMatrix<double> gs_rho(e_sys.basis_size, e_sys.basis_size), gs_vect(e_sys.basis_size, e_env.basis_size);
    vector<Triplet<double>> tripletList;
    tripletList.reserve(gs_sector_dim);

    for (int i = 0; i < gs_sector_dim; ++i) {

        int sb_idx = sb_idxs_in_sector[gs_sector][i];
        int e_sys_idx = sb_idx / e_env.basis_size;
        int e_env_idx = sb_idx % e_env.basis_size;

        tripletList.push_back(Triplet<double>(e_sys_idx, e_env_idx, sector_gs[gs_sector](i, 0)));
    }

    gs_vect.setFromTriplets(tripletList.begin(), tripletList.end());
    gs_rho = gs_vect * gs_vect.transpose();
    gs_rho.makeCompressed();

    map<int, vector<int>> e_sys_idxs_with_sz;
    for (int e_sys_idx = 0; e_sys_idx < e_sys.basis_size; ++e_sys_idx) {
        int e_sys_sz = e_sys.sz_of_idx[e_sys_idx];
        e_sys_idxs_with_sz[e_sys_sz].push_back(e_sys_idx);
    }

    if (verbose) {
        out_stream << "Reduced density matrix created." << endl;
        out_stream << endl << "Diagonalizing reduced density matrix." << endl;
    }

    auto rho_start = high_resolution_clock::now();

    map<int, vector<double>> highest_lambdas_of_sz;
    map<int, vector<MatrixXd>> highest_evecs_of_sz;

    vector<int> sz_keys;
    for (const auto& kv : e_sys_idxs_with_sz)
        sz_keys.push_back(kv.first);

    vector<map<int, vector<double>>> thread_lambdas(num_threads);
    vector<map<int, vector<MatrixXd>>> thread_evecs(num_threads);

    #pragma omp parallel for num_threads(num_threads)
    for (int idx = 0; idx < sz_keys.size(); ++idx) {
        int thread_id = omp_get_thread_num();
        int e_sys_sz = sz_keys[idx];
        const vector<int>& e_sys_idxs = e_sys_idxs_with_sz[e_sys_sz];

        int sz_dim = e_sys_idxs.size();
        int sz_m = min(my_m, sz_dim);

        for (int i = 0; i < sz_m; ++i) {
            thread_lambdas[thread_id][e_sys_sz].push_back(0.0);
            thread_evecs[thread_id][e_sys_sz].push_back(MatrixXd::Zero(sz_dim, 1));
        }

        SparseMatrix<double> sub_rho(sz_dim, sz_dim);
        vector<Triplet<double>> tripletList_sz;
        tripletList_sz.reserve(sz_dim * sz_dim);
        for (int i = 0; i < sz_dim; ++i) {
            for (int j = 0; j < sz_dim; ++j) {
                double value = gs_rho.coeff(e_sys_idxs[i], e_sys_idxs[j]);
                if (value != 0)
                    tripletList_sz.push_back(Triplet<double>(i, j, value));
            }
        }
        sub_rho.setFromTriplets(tripletList_sz.begin(), tripletList_sz.end());
        sub_rho.makeCompressed();

        if (sz_dim > sz_m && sz_dim > 512) {
            SparseSymMatProd<double> op_sub_rho(sub_rho);
            SymEigsSolver<SparseSymMatProd<double>> eigs_rho(op_sub_rho, sz_m, min(6 * sz_m, sz_dim - 1));

            int nconv_rho = eigs_rho.compute(SortRule::LargestAlge);
            if (eigs_rho.info() != Spectra::CompInfo::Successful) {
                out_stream << "Spectra failed to converge for gs sector, sz " << e_sys_sz << ". Info: " << static_cast<int>(eigs_rho.info()) << endl;
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < sz_m; ++i) {
                thread_lambdas[thread_id][e_sys_sz][i] = eigs_rho.eigenvalues()(i);
                thread_evecs[thread_id][e_sys_sz][i] = eigs_rho.eigenvectors().col(i);
            }
        }
        else {
            SelfAdjointEigenSolver<MatrixXd> eigs_rho(sub_rho);

            for (int i = 0; i < sz_m; ++i) {
                thread_lambdas[thread_id][e_sys_sz][sz_m - i - 1] = eigs_rho.eigenvalues()(i);
                thread_evecs[thread_id][e_sys_sz][sz_m - i - 1] = eigs_rho.eigenvectors().col(i);
            } 
        }
    }
    for (const auto& local_lambdas : thread_lambdas) {
        for (const auto& kv : local_lambdas) {
            int sz = kv.first;
            if (highest_lambdas_of_sz.find(sz) == highest_lambdas_of_sz.end())
                highest_lambdas_of_sz[sz] = kv.second;
            else
                highest_lambdas_of_sz[sz].insert(highest_lambdas_of_sz[sz].end(), kv.second.begin(), kv.second.end());
        }
    }
    for (const auto& local_evecs : thread_evecs) {
        for (const auto& kv : local_evecs) {
            int sz = kv.first;
            if (highest_evecs_of_sz.find(sz) == highest_evecs_of_sz.end())
                highest_evecs_of_sz[sz] = kv.second;
            else
                highest_evecs_of_sz[sz].insert(highest_evecs_of_sz[sz].end(), kv.second.begin(), kv.second.end());
        }
    }

    auto rho_end = high_resolution_clock::now();
    total_rho_time += duration<double>(rho_end - rho_start).count();

    if (verbose) {
        out_stream << "Reduced density matrix diagonalized." << endl;
        out_stream << endl << "Processing kept states..." << endl;
    }

    vector< pair<int, int> > sz_and_pos_of_highest_evecs;
    for (const auto& [sz, lambdas] : highest_lambdas_of_sz) {
        for (int pos = 0; pos < lambdas.size(); ++pos) {

            if (sz_and_pos_of_highest_evecs.size() < my_m) {
                int i = 0;
                while (i < sz_and_pos_of_highest_evecs.size() && highest_lambdas_of_sz[sz][pos] <= highest_lambdas_of_sz[sz_and_pos_of_highest_evecs[i].first][sz_and_pos_of_highest_evecs[i].second]) {
                    i++;
                }
                sz_and_pos_of_highest_evecs.insert(sz_and_pos_of_highest_evecs.begin() + i, pair<int, int>(sz, pos));
            }
            else if (highest_lambdas_of_sz[sz][pos] > highest_lambdas_of_sz[sz_and_pos_of_highest_evecs.back().first][sz_and_pos_of_highest_evecs.back().second]) {

                sz_and_pos_of_highest_evecs.pop_back();

                int i = 0;
                while (i < sz_and_pos_of_highest_evecs.size() && highest_lambdas_of_sz[sz][pos] <= highest_lambdas_of_sz[sz_and_pos_of_highest_evecs[i].first][sz_and_pos_of_highest_evecs[i].second]) {
                    i++;
                }
                sz_and_pos_of_highest_evecs.insert(sz_and_pos_of_highest_evecs.begin() + i, pair<int, int>(sz, pos));
            }

        }
    }

    vector<double> eig_vals(my_m);
    MatrixXd eig_vects_in_e_sys_basis = MatrixXd::Zero(e_sys.basis_size, my_m);

    for (int i = 0; i < my_m; ++i) {

        int sz = sz_and_pos_of_highest_evecs[i].first;
        int pos = sz_and_pos_of_highest_evecs[i].second;

        eig_vals[i] = highest_lambdas_of_sz[sz][pos];
        MatrixXd eig_vect_in_sz_basis = highest_evecs_of_sz[sz][pos];

        for (int j = 0; j < eig_vect_in_sz_basis.rows(); ++j) {

            int e_sys_idx = e_sys_idxs_with_sz[sz][j];
            eig_vects_in_e_sys_basis(e_sys_idx, i) = eig_vect_in_sz_basis(j, 0);
        }
    }

    SparseMatrix<double> transf_mat = eig_vects_in_e_sys_basis.sparseView();
    transf_mat.makeCompressed();

    if (verbose) {
        out_stream << "Kept states processed." << endl;
        out_stream << endl << "Computing physical quantities..." << endl;
    }

    double trunc_err = 1.0, entropy = 0.0;
    for (int i = 0; i < my_m; ++i) {
        trunc_err -= eig_vals[i];
        if (eig_vals[i] > 1e-12)
            entropy -= eig_vals[i] * log(eig_vals[i]);
    }

    if (trunc_err > 1e-5)
        out_stream << "Warning: High truncation error = " << trunc_err << endl;

    int dist = e_sys.length - fixed_pos;
    double correlation = 0.0;

    if (finite && sys_label == "r") {
        
        if (dist == 1) {
            e_sys.fixed_sz = e_sys.conn_sz;
            started = true;
        }

        if (dist > 0 && started) {
            SparseMatrix<double> correlator = KroneckerProductSparse(e_sys.fixed_sz, e_env.conn_sz);
            MatrixXd correlator_mat = correlator.toDense();
            MatrixXd gs = MatrixXd::Zero(e_sys.basis_size * e_env.basis_size, 1);

            for (int e_sys_idx = 0; e_sys_idx < e_sys.basis_size; e_sys_idx++)
                for (int e_env_idx = 0; e_env_idx < e_env.basis_size; e_env_idx++)
                    gs(e_sys_idx * e_env.basis_size + e_env_idx, 0) = gs_vect.coeff(e_sys_idx, e_env_idx);

            correlation = (gs.transpose() * correlator_mat * gs)(0, 0);
        }
    }
    else
        started = false;

    SparseMatrix<double> new_ham = rot_trunc(e_sys.ham, transf_mat);
    SparseMatrix<double> new_conn_sz = rot_trunc(e_sys.conn_sz, transf_mat);
    SparseMatrix<double> new_conn_sp = rot_trunc(e_sys.conn_sp, transf_mat);
    SparseMatrix<double> new_fixed_sz = rot_trunc(e_sys.fixed_sz, transf_mat);

    result res;

    res.bl = {e_sys.length, my_m, new_ham, new_conn_sz, new_conn_sp, new_fixed_sz, vector<int>(my_m)};
    for (int i = 0; i < my_m; ++i) {
        res.bl.sz_of_idx[i] = sz_and_pos_of_highest_evecs[i].first;
    }
    
    res.energy = gs_energy;
    res.truncation_error = trunc_err;
    res.entropy = entropy;
    res.dist = dist;
    res.correlation = correlation;
    res.transformation_matrix = eig_vects_in_e_sys_basis;
    res.gs_vector = gs_vect;

    if (verbose) out_stream << "Physical quantities computed." << endl;
    
    return res;
}

string graphic(const bloc& sys_bloc, const bloc& env_bloc, const string& sys_label = "l", const int sweep = 0) {
    
    string graphic = string(sys_bloc.length, '=') + "**" + string(env_bloc.length, '-');

    if (sys_label == "r") {
        reverse(graphic.begin(), graphic.end());
    }

    graphic = to_string(sweep) + ") " + graphic;

    return graphic;
}

void finite_system_algorithm(ostream& out_stream) {
    out_stream << endl << "#################################" << endl;
    out_stream << "Start of infinite-size algorithm." << endl;
    out_stream << "#################################" << endl;

    bloc sys_bloc = initial_bloc, env_bloc;
    string sys_label = "l", env_label = "r";

    MatrixXd last_gs;
    MatrixXd sys_trmat, env_trmat;

    bloc_disk[{"l", sys_bloc.length}] = sys_bloc;
    bloc_disk[{"r", sys_bloc.length}] = sys_bloc;

    while (2 * sys_bloc.length < num_sites) {

        out_stream << endl << graphic(sys_bloc, sys_bloc) << endl;
        result res = dmrg_step(sys_bloc, sys_bloc, m, sys_label, out_stream);

        sys_bloc = res.bl;
        sys_trmat = res.transformation_matrix;
        last_gs = res.gs_vector;
        
    if (verbose) out_stream << endl << "Saving to disk..." << endl;

        bloc_disk[{"l", sys_bloc.length}] = sys_bloc;
        bloc_disk[{"r", sys_bloc.length}] = sys_bloc;

        trmat_disk[{"l", sys_bloc.length}] = sys_trmat;
        trmat_disk[{"r", sys_bloc.length}] = sys_trmat;

    if (verbose) out_stream << "Saved to disk." << endl;
    }
    
    out_stream << endl << "###############################" << endl;
    out_stream << "Start of finite-size algorithm." << endl;
    out_stream << "###############################" << endl;

    for (int sweep = 1; sweep < num_sweeps + 1; ++sweep) {
        while (true) {
            if (verbose) out_stream << endl << "Fetching from disk..." << endl;

            int env_length = num_sites - sys_bloc.length - 2;
            env_bloc = bloc_disk[{env_label, env_length}];
            env_trmat = trmat_disk[{env_label, env_length + 1}];

            if (verbose) {
                out_stream << "Fetched from disk." << endl;
                out_stream << endl << "Obtaining guess..." << endl;
            }

            MatrixXd temp1 = sys_trmat.transpose() * last_gs;
            MatrixXd temp2(temp1.rows() * site_dim, temp1.cols() / site_dim);
            for (int i = 0; i < temp2.rows(); ++i) {
                for (int j = 0; j < temp2.cols(); ++j) {
                    temp2(i, j) = temp1(i / site_dim, j * site_dim + i % site_dim);
                }
            }

            MatrixXd temp3 = temp2 * env_trmat.transpose();
            MatrixXd guess(temp3.rows() * temp3.cols(), 1);
            for (int i = 0; i < temp3.rows(); ++i) {
                for (int j = 0; j < temp3.cols(); ++j) {
                    guess(i * temp3.cols() + j, 0) = temp3(i, j);
                }
            }

            if (env_bloc.length == 1) {
                swap(sys_bloc, env_bloc);
                swap(sys_label, env_label);

                MatrixXd temp = guess;
                for (int se = 0; se < site_dim; se++)
                    for (int e = 0; e < env_bloc.basis_size; e++)
                        for (int ss = 0; ss < site_dim; ss++)
                            for (int s = 0; s < sys_bloc.basis_size; s++)
                                guess(ss + site_dim * (s + sys_bloc.basis_size * (se + site_dim * e)), 0) = temp(se + site_dim * (e + env_bloc.basis_size * (ss + site_dim * s)), 0);
            }

            if (verbose) out_stream << "Guess obtained." << endl;

            out_stream << endl << graphic(sys_bloc, env_bloc, sys_label, sweep) << endl;
            result res = dmrg_step(sys_bloc, env_bloc, m, sys_label, out_stream, true, guess);

            energies[(sweep - 1) * (num_sites - 4) + sys_bloc.length - 1] = res.energy;
            truncation_errors[(sweep - 1) * (num_sites - 4) + sys_bloc.length - 1] = res.truncation_error;
            entropies[(sweep - 1) * (num_sites - 4) + sys_bloc.length - 1] = res.entropy;

            if (sys_label == "r" && res.dist > 0)
                correlations[(sweep - 1) * (num_sites - fixed_pos - 3) + sys_bloc.length - fixed_pos] = res.correlation;

            sys_bloc = res.bl;
            sys_trmat = res.transformation_matrix;
            last_gs = res.gs_vector;

            if (verbose) out_stream << endl << "Saving to disk..." << endl;

            bloc_disk[{sys_label, sys_bloc.length}] = sys_bloc;
            trmat_disk[{sys_label, sys_bloc.length}] = sys_trmat;

            if (verbose) out_stream << "Saved to disk." << endl;

            if (sys_label == "l" && 2 * sys_bloc.length == num_sites) {
                break;
            }

        }
    }
}

void read_input_file(const string& filename, ostream& out_stream) {
    out_stream << endl << "Attempting to open file: " << filename << endl;
    ifstream infile(filename);
    if (!infile) {
        out_stream << "Error opening file: " << filename << endl;
        exit(1);
    }
    out_stream << "File opened successfully." << endl;

    string param;
    while (infile >> param) {
        if (param == "num_sites")
            infile >> num_sites;
        else if (param == "delta")
            infile >> delta;
        else if (param == "m")
            infile >> m;
        else if (param == "fixed_pos")
            infile >> fixed_pos;
        else if (param == "num_sweeps")
            infile >> num_sweeps;
        else if (param == "num_threads")
            infile >> num_threads;
        else if (param == "verbose")
            infile >> verbose;
        else {
            out_stream << "Unknown parameter in input file: " << param << endl;
            exit(1);
        }
    }

    if (num_sites % 2 != 0) {
        out_stream << "Error: num_sites must be an even number." << endl;
        return;
    }
    if (fixed_pos < 1 || fixed_pos > num_sites / 2) {
        out_stream << "Error: fixed_pos must be between 1 and num_sites/2." << endl;
        return;
    }

    infile.close();
    out_stream << endl << "Read input: num_sites = " << num_sites << ", delta = " << delta << ", m = " << m << ", num_sweeps = " << num_sweeps << ", fixed_pos = " << fixed_pos << ", num_threads = " << num_threads << endl;
}

string results_file_prefix() {
    ostringstream oss;
    oss << "results/"
        << "N" << num_sites
        << "_d" << delta
        << "_m" << m
        << "_fp" << fixed_pos
        << "_sw" << num_sweeps
        << "_nt" << num_threads;
    return oss.str();
}

void write_output_file(ostream& out_stream) {
    string prefix = results_file_prefix();

    system("mkdir -p results");
    ofstream energy_file(prefix + "_energy.txt");
    if (!energy_file.is_open()) {
        out_stream << "Error: Could not open " << prefix << "_energy.txt for writing." << endl;
        return;
    }
    for (int i = 0; i < energies.size(); ++i) {
        energy_file << energies[i] << "\n";
    }
    energy_file.close();

    ofstream trunc_file(prefix + "_truncation_error.txt");
    if (!trunc_file.is_open()) {
        out_stream << "Error: Could not open " << prefix << "_truncation_error.txt for writing." << endl;
        return;
    }
    for (int i = 0; i < truncation_errors.size(); ++i) {
        trunc_file << truncation_errors[i] << "\n";
    }
    trunc_file.close();

    ofstream entropy_file(prefix + "_entropy.txt");
    if (!entropy_file.is_open()) {
        out_stream << "Error: Could not open " << prefix << "_entropy.txt for writing." << endl;
        return;
    }
    for (int i = 0; i < entropies.size(); ++i) {
        entropy_file << entropies[i] << "\n";
    }
    entropy_file.close();

    ofstream correlation_file(prefix + "_correlation.txt");
    if (!correlation_file.is_open()) {
        out_stream << "Error: Could not open " << prefix << "_correlation.txt for writing." << endl;
        return;
    }
    for (int i = 0; i < correlations.size(); ++i) {
        correlation_file << correlations[i] << "\n";
    }
    correlation_file.close();

    out_stream << endl << "Results written to " << prefix << "_energy.txt, "
               << prefix << "_truncation_error.txt, "
               << prefix << "_entropy.txt, "
               << prefix << "_correlation.txt" << endl << endl;
}

int main() {
    ofstream out_stream("output.txt");

    out_stream << endl << "Starting DMRG simulation..." << endl;
    auto start_time = high_resolution_clock::now();

    read_input_file("input.txt", out_stream);

    energies.resize( (num_sites - 4) * num_sweeps );
    truncation_errors.resize( (num_sites - 4) * num_sweeps );
    entropies.resize( (num_sites - 4) * num_sweeps );
    correlations.resize( (num_sites - fixed_pos - 3) * num_sweeps );

    initParallel();

    finite_system_algorithm(out_stream);
    write_output_file(out_stream);

    auto end_time = chrono::high_resolution_clock::now();
    double elapsed_time = duration<double>(end_time - start_time).count();

    double other_time = elapsed_time - total_ham_time - total_rho_time;

    out_stream << "Program execution time: " << elapsed_time << " seconds" << endl;
    out_stream << "Hamiltonian diagonalization time: " << total_ham_time << " seconds" << endl;
    out_stream << "Density matrix diagonalization time: " << total_rho_time << " seconds" << endl;
    out_stream << "Other time: " << other_time << " seconds" << endl << endl;

    out_stream.close();

    return 0;
}
