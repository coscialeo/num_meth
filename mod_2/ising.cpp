#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <fstream>

#include <vector>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

#define EIGEN_USE_BLAS

int num_sites;
double g, dg, h;

int num_realizations;
double s_bin_size, r_bin_size;

int num_threads;

vector< map<int, int> > spacing_bins_list, ratio_bins_list;
vector<int> diag_times;

map<int, int> put_in_bins(vector<double>& data, double bin_size) {
    map<int, int> bins;

    for (const auto& value : data) {
        int bin_index = static_cast<int>(value / bin_size);
        bins[bin_index]++;
    }

    return bins;
}

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
        else if (param == "dg")
            infile >> dg;
        else if (param == "h")
            infile >> h;
        else if (param == "num_realizations")
            infile >> num_realizations;
        else if (param == "s_bin_size")
            infile >> s_bin_size;
        else if (param == "r_bin_size")
            infile >> r_bin_size;
        else {
            cout << "Unknown parameter in input file: " << param << endl;
            exit(1);
        }
    }

    infile.close();
    cout << endl << "Read input: num_sites = " << num_sites << ", g = " << g << ", dg = " << dg << ", h = " << h << ", num_realizations = " << num_realizations << endl;
}

vector<double> randomize_field() {

    vector<double> gg(num_sites);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> unif(g - dg, g + dg);

    for (int i = 0; i < num_sites; i++)
        gg[i] = unif(gen);

    return gg;
}

vector<double> compute_magnetization(vector<double>& gg) {
 
    vector<double> mag_z(1 << num_sites);

    double min_mag_z = 0.0;
    for (auto x : gg)
        min_mag_z -= x;

    fill(mag_z.begin(), mag_z.end(), min_mag_z);

    int temp = 0, bit = 0;
    int i, j;

    for (i = 0; i < mag_z.size(); i++) {

        temp = i;
        for (j = 0; j < num_sites; j++) {
            bit = temp % 2;

            mag_z[i] += 2 * gg[j] * bit;

            temp /= 2;
        }
    }

    return mag_z;
}

SparseMatrix<double> build_ham(vector<double>& mag_z) {

    SparseMatrix<double> ham(1 << num_sites, 1 << num_sites);
    ham.reserve(VectorXd::Constant(1 << num_sites, 2 * num_sites + 1));

    int site = 0, next = 1;
    int state = 0, coupled = 0;

    for (state = 0; state < pow(2, num_sites); state++) {

        ham.insert(state, state) = -mag_z[state];

        for (site = 0; site < num_sites; site++) {

            coupled = state ^ (1 << site);
            ham.insert(state, coupled) = - h;

            next = (site + 1) % num_sites;
            coupled = coupled ^ (1 << next);
            ham.insert(state, coupled) = -1.0;
        }
    }
    ham.makeCompressed();

    return ham;
}

void solve_ham(SparseMatrix<double>& ham, int idx) {

    static double diag_time = 0.0;
    auto diag_start = high_resolution_clock::now();

    SelfAdjointEigenSolver<SparseMatrix<double>> solver(ham);

    auto diag_end = high_resolution_clock::now();
    diag_time += duration_cast<duration<double>>(diag_end - diag_start).count();

    vector<double> spacings(pow(2, num_sites) - 1), ratios(pow(2, num_sites) - 2);
    spacings[0] = solver.eigenvalues()(1) - solver.eigenvalues()(0);
    for (int i = 2; i < solver.eigenvalues().size(); i++) {
        spacings[i - 1] = solver.eigenvalues()(i) - solver.eigenvalues()(i - 1);
        ratios[i - 2] = min(spacings[i - 1], spacings[i - 2]) / max(spacings[i - 1], spacings[i - 2]);
    }

    spacing_bins_list[idx] = put_in_bins(spacings, s_bin_size);
    ratio_bins_list[idx] = put_in_bins(ratios, r_bin_size);

    diag_times[idx] = duration_cast<duration<double>>(diag_end - diag_start).count();
}

int main() {
    cout << endl << "Starting program..." << endl;
    auto start = high_resolution_clock::now();
    
    ifstream infile("input.txt");
    read_input("input.txt", infile);

    spacing_bins_list.resize(num_realizations);
    ratio_bins_list.resize(num_realizations);

    diag_times.resize(num_realizations);
    double total_diag_time = 0.0;

    for (int idx = 0; idx < num_realizations; idx++) {
        diag_times[idx] = 0.0;

        spacing_bins_list[idx].clear();
        ratio_bins_list[idx].clear();

        vector<double> gg = randomize_field();
        vector<double> mag_z = compute_magnetization(gg);
        SparseMatrix<double> ham = build_ham(mag_z);
        solve_ham(ham, idx);
    }

    map<int, int> total_spacing_bins, total_ratio_bins;
    for (const auto& spacing_bins : spacing_bins_list) {
        for (const auto& [bin, count] : spacing_bins)
            total_spacing_bins[bin] += count;
    }
    for (const auto& ratio_bins : ratio_bins_list) {
        for (const auto& [bin, count] : ratio_bins)
            total_ratio_bins[bin] += count;
    }

    string spacing_file_name = "spacings_" + to_string(num_sites) + "_" + to_string(g) + "_" + to_string(dg) + "_" + to_string(h) + ".txt";
    string ratio_file_name = "ratios_" + to_string(num_sites) + "_" + to_string(g) + "_" + to_string(dg) + "_" + to_string(h) + ".txt";

    ofstream spacing_file(spacing_file_name);
    for (const auto& [bin, count] : total_spacing_bins)
        spacing_file << bin << " " << count << endl;
    spacing_file.close();

    ofstream ratio_file(ratio_file_name);
    for (const auto& [bin, count] : total_ratio_bins)
        ratio_file << bin << " " << count << endl;
    ratio_file.close();

    auto end = high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    total_diag_time = std::accumulate(diag_times.begin(), diag_times.end(), 0.0);
    double rest_time = duration - total_diag_time;

    cout << endl << "Program completed." << endl;
    cout << "Total time: " << duration << " seconds." << endl;
    cout << "Time spent diagonalizing: " << total_diag_time << " seconds." << endl;
    cout << "Time spent on other tasks: " << rest_time << " seconds." << endl;
    cout << endl;

    return 0;
}
