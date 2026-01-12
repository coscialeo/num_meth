#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>

#include <iostream>
#include <fstream>

#include <vector>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace Spectra;
using namespace std;
using namespace std::chrono;

#define EIGEN_USE_BLAS

int num_sites;
double g, d;

int num_eigpairs, num_realizations;

int num_threads;

vector<double> ratios, divs, divs_sq, entropies, entropies_sq;
vector<double> diag_times, proc_times;

struct realization_info {
    vector<double> mag_z;
    vector< vector<int> > basis;
    vector<int> sz_0_state_at_idx;
    map<int, int> idx_of_sz_0_state;
};

void read_input(const string& filename, ifstream& infile) {
    cout << endl << "Attempting to open file: " << filename << endl;
    if (!infile) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    cout << "File opened successfully." << endl;

    string param;
    while (infile >> param) {
        if (param == "num_sites")
            infile >> num_sites;
        else if (param == "g")
            infile >> g;
        else if (param == "d")
            infile >> d;
        else if (param == "num_eigpairs")
            infile >> num_eigpairs;
        else if (param == "num_realizations")
            infile >> num_realizations;
        else if (param == "num_threads")
            infile >> num_threads;
        else {
            cout << "Unknown parameter in input file: " << param << endl;
            exit(1);
        }
    }

    infile.close();
    cout << endl << "Read input: num_sites = " << num_sites << ", g = " << g << ", d = " << d << ", num_eigpairs = " << num_eigpairs << ", num_realizations = " << num_realizations << ", num_threads = " << num_threads << endl;
}

vector<double> randomize_field() {

    vector<double> gg(num_sites);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> unif(-g, g);

    for (int i = 0; i < num_sites; i++)
        gg[i] = unif(gen);

    return gg;
}

realization_info make_basis(vector<double>& gg) {

    realization_info info;
    info.mag_z.resize(1 << num_sites);
    info.basis.resize(1 << num_sites, vector<int>(num_sites));
    info.sz_0_state_at_idx.resize(0);
    info.idx_of_sz_0_state.clear();

    double min_mag_z = 0.0;
    for (auto x : gg)
        min_mag_z -= x;

    fill(info.mag_z.begin(), info.mag_z.end(), min_mag_z);

    for (int state = 0; state < (1 << num_sites); state++) {

        int temp = state, sz = 0;
        for (int site = 0; site < num_sites; site++) {
            int bit = temp % 2;

            info.mag_z[state] += 2 * gg[site] * bit;
            info.basis[state][site] = bit;
            sz += bit;

            temp /= 2;
        }

        if (sz == num_sites / 2) {
            info.sz_0_state_at_idx.push_back(state);
            info.idx_of_sz_0_state[state] = info.sz_0_state_at_idx.size() - 1;
        }
    }

    return info;
}

SparseMatrix<double> build_ham(realization_info& info) {

    int num_states = info.sz_0_state_at_idx.size();
    SparseMatrix<double> ham(num_states, num_states);
    ham.reserve(VectorXd::Constant(num_states, 2 * num_sites + 1));

    for (int state : info.sz_0_state_at_idx) {

        double temp = -info.mag_z[state];
        for (int site = 0; site < num_sites; site++) {

            int next = (site + 1) % num_sites;
            if (info.basis[state][site] == info.basis[state][next])
                temp -= d;
            else {
                temp += d;

                int coupled;
                if (info.basis[state][site] == 1)
                    coupled = state - (1 << site) + (1 << next);
                else
                    coupled = state + (1 << site) - (1 << next);

                ham.insert(info.idx_of_sz_0_state[state], info.idx_of_sz_0_state[coupled]) = -2.0;
            }
        }
        ham.insert(info.idx_of_sz_0_state[state], info.idx_of_sz_0_state[state]) = temp;
    }
    ham.makeCompressed();

    return ham;
}

double kullback_leibler_divergence(const VectorXd& vec1, const VectorXd& vec2) {

    double kl_div = 0.0;
    for (int i = 0; i < vec1.size(); i++)
        if (vec1(i) > 1e-12 && vec2(i) > 1e-12)
            kl_div += (vec1(i) * vec1(i)) * log( (vec1(i) * vec1(i)) / (vec2(i) * vec2(i)) );

    return kl_div;
}

void get_ratios(const VectorXd& eigvals, int idx) {

    vector<double> ss(num_eigpairs - 1), rr(num_eigpairs - 2);
    ss[0] = eigvals(1) - eigvals(0);

    for (int i = 2; i < eigvals.size(); i++) {
        ss[i - 1] = eigvals(i) - eigvals(i - 1);
        rr[i - 2] = min(ss[i - 1], ss[i - 2]) / max(ss[i - 1], ss[i - 2]);
    }

    ratios[idx] = accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
}

void get_kl_div(const MatrixXd& eigvecs, int idx) {

    vector<double> dd(num_eigpairs - 1), dd_sq(num_eigpairs - 1);

    for (int i = 1; i < eigvecs.cols(); i++) {
        dd[i - 1] = kullback_leibler_divergence(eigvecs.col(i), eigvecs.col(i - 1));
        dd_sq[i - 1] = dd[i - 1] * dd[i - 1];
    }

    divs[idx] = accumulate(dd.begin(), dd.end(), 0.0) / dd.size();
    divs_sq[idx] = accumulate(dd_sq.begin(), dd_sq.end(), 0.0) / dd_sq.size();
}

void get_entropy(const MatrixXd& eigvecs, realization_info& info, int idx) {

    entropies[idx] = 0.0;
    entropies_sq[idx] = 0.0;

    for (int i = 0; i < eigvecs.cols(); i++) {

        VectorXd psi;
        psi.resize(1 << num_sites);
        psi.setZero();
        for (int j = 0; j < eigvecs.rows(); j++)
            psi[info.sz_0_state_at_idx[j]] = eigvecs(j, i);

        Map<MatrixXd> psi_matrix(psi.data(), 1 << (num_sites / 2), 1 << (num_sites - num_sites / 2));
        MatrixXd rho = psi_matrix * psi_matrix.transpose();

        SelfAdjointEigenSolver<MatrixXd> solver(rho);
        double temp = 0.0;
        for (int j = 0; j < solver.eigenvalues().size(); j++)
            if (solver.eigenvalues()(j) > 1e-12)
                temp -= solver.eigenvalues()(j) * log(solver.eigenvalues()(j));
            
        
        entropies[idx] += temp;
        entropies_sq[idx] += temp * temp;
    }

    entropies[idx] /= eigvecs.cols();
    entropies_sq[idx] /= eigvecs.cols();
}

void solve_ham(SparseMatrix<double>& ham, realization_info& info, int idx) {

    auto diag_start = high_resolution_clock::now();

    SparseSymMatProd<double> op(ham);
    SymEigsSolver<SparseSymMatProd<double>> max_eig(op, 1, 6);
    max_eig.init();
    int nconv = max_eig.compute(SortRule::LargestAlge, 10000, 1e-10);
    if (nconv != 1) {
        cerr << "Error: Max eigenvalue computation did not converge!" << endl;
        exit(1);
    }

    SymEigsSolver<SparseSymMatProd<double>> min_eig(op, 1, 6);
    min_eig.init();
    nconv = min_eig.compute(SortRule::SmallestAlge, 10000, 1e-10);
    if (nconv != 1) {
        cerr << "Error: Min eigenvalue computation did not converge!" << endl;
        exit(1);
    }

    SparseSymShiftSolve<double> shift_op(ham);
    SymEigsShiftSolver<SparseSymShiftSolve<double>> mid_eigs(shift_op, num_eigpairs, 6 * num_eigpairs, 0.5 * (max_eig.eigenvalues()[0] + min_eig.eigenvalues()[0]));
    mid_eigs.init();
    nconv = mid_eigs.compute(SortRule::LargestMagn, 10000, 1e-10);
    if (nconv != num_eigpairs) {
        cerr << "Error: Mid eigenvalue computation did not converge!" << endl;
        exit(1);
    }

    auto diag_end = high_resolution_clock::now();
    diag_times[idx] = duration_cast<duration<double>>(diag_end - diag_start).count();

    auto proc_start = high_resolution_clock::now();

    VectorXd eigvals = mid_eigs.eigenvalues();
    MatrixXd eigvecs = mid_eigs.eigenvectors();
    vector<pair<double, VectorXd>> eig_pairs;

    for (int i = 0; i < eigvals.size(); ++i)
        eig_pairs.emplace_back(eigvals(i), eigvecs.col(i));
    sort(eig_pairs.begin(), eig_pairs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    
    for (int i = 0; i < eigvals.size(); ++i) {
        eigvals(i) = eig_pairs[i].first;
        eigvecs.col(i) = eig_pairs[i].second;
    }

    get_ratios(eigvals, idx);
    get_kl_div(eigvecs, idx);
    get_entropy(eigvecs, info, idx);

    auto proc_end = high_resolution_clock::now();
    proc_times[idx] = duration_cast<duration<double>>(proc_end - proc_start).count();
}

int main() {
    cout << endl << "Starting program..." << endl;
    auto start = high_resolution_clock::now();
    
    ifstream infile("input.txt");
    read_input("input.txt", infile);

    ratios.resize(num_realizations);
    divs.resize(num_realizations);
    divs_sq.resize(num_realizations);
    entropies.resize(num_realizations);
    entropies_sq.resize(num_realizations);

    diag_times.resize(num_realizations);
    proc_times.resize(num_realizations);
    double total_diag_time = 0.0, total_proc_time = 0.0;

    initParallel();
    #pragma omp parallel for num_threads(num_threads)
    for (int idx = 0; idx < num_realizations; idx++) {
        diag_times[idx] = 0.0;
        proc_times[idx] = 0.0;

        vector<double> gg = randomize_field();
        realization_info info = make_basis(gg);
        SparseMatrix<double> ham = build_ham(info);
        solve_ham(ham, info, idx);
    }

    double ratio = 0.0, ratio_sq = 0.0, ratio_err = 0.0;
    for (auto r : ratios) {
        ratio += r;
        ratio_sq += r * r;
    }
    ratio /= num_realizations;
    ratio_sq /= num_realizations;
    ratio_err = sqrt((ratio_sq - ratio * ratio) / num_realizations);

    double div = 0.0, div_sq = 0.0, div_err = 0.0;
    for (int i = 0; i < divs.size(); i++) {
        div += divs[i];
        div_sq += divs_sq[i];
    }
    div /= num_realizations;
    div_sq /= num_realizations;
    div_err = sqrt((div_sq - div * div) / num_realizations / num_eigpairs);

    double ent = 0.0, ent_sq = 0.0, ent_err = 0.0;
    for (int i = 0; i < entropies.size(); i++) {
        ent += entropies[i];
        ent_sq += entropies_sq[i];
    }
    ent /= num_realizations;
    ent_sq /= num_realizations;
    ent_err = sqrt((ent_sq - ent * ent) / num_realizations / num_eigpairs);

    string out_file_name = "xxz_" + to_string(num_sites) + "_" + to_string(g) + "_" + to_string(d) + ".txt";

    ofstream out_file(out_file_name);
    out_file << "r " << ratio << endl;
    out_file << "dr " << ratio_err << endl;
    out_file << "d " << div << endl;
    out_file << "dd " << div_err << endl;
    out_file << "s " << ent << endl;
    out_file << "ds " << ent_err << endl;
    out_file.close();

    auto end = high_resolution_clock::now();
    double total_duration = duration_cast<duration<double>>(end - start).count();

    total_diag_time = accumulate(diag_times.begin(), diag_times.end(), 0.0) / num_threads;
    total_proc_time = accumulate(proc_times.begin(), proc_times.end(), 0.0) / num_threads;
    double other_time = total_duration - total_proc_time - total_diag_time;

    cout << endl << "Program completed." << endl;
    cout << "Total time: " << total_duration << " seconds." << endl;
    cout << "Time spent diagonalizing: " << total_diag_time << " seconds." << endl;
    cout << "Time spent processing: " << total_proc_time << " seconds." << endl;
    cout << "Time spent on other tasks: " << other_time << " seconds." << endl;
    cout << endl;

    return 0;
}
