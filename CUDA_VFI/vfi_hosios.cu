#define nA 11
#define nk 1001
#define nn 1001
#define tol 1e-1
#define maxiter 10000
#define maxouteriter 1
#define kwidth 0.08

/* Includes, system */
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

// Includes, Thrust
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// Includes, cuda 
#include <cublas_v2.h>
#include "cuda_helpers.h"

// Includes model stuff
#include "dmpmodel.h"

// This functor optimal kplus and Vplus
struct update
{
	// Data member
	//====Input pointers
	double *A, *K, *N, *EV;
	para p;
	//==== Ouput pointers
	double *copt, *kopt, *nopt, *vopt;
	int *koptind;
	int *noptind;
    double *Vplus;

	// Construct this object, create util from _util, etc.
	__host__ __device__
    update(double* A_ptr, double* K_ptr, double* N_ptr, double* EV_ptr, double* copt_ptr, double* vopt_ptr, double* kopt_ptr, double* nopt_ptr, int* koptind_ptr, int* noptind_ptr, double* Vplus_ptr, para _p) {
		A = A_ptr; K = K_ptr; N = N_ptr; EV = EV_ptr;
		copt = copt_ptr; kopt = kopt_ptr; nopt = nopt_ptr; vopt = vopt_ptr;
		koptind = koptind_ptr; Vplus = Vplus_ptr;
		noptind = noptind_ptr;
		p = _p;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int subs[3];
		int size_vec[3];
		size_vec[0] = nA;
		size_vec[1] = nk;
		size_vec[2] = nn;
		ind2sub(3,size_vec,index,subs);
		int i_A = subs[0];
		int i_k = subs[1];
		int i_n = subs[2];

		// Find and construct state and control, otherwise they won't update in the for loop
		double a = A[i_A]; 
		double k =K[i_k]; double n=N[i_n];
		int left_n_idx = fit2grid((1.0-p.x)*n,nn,N);
		double asset = a*pow(k/n,p.aalpha)*n + (1.0-p.ddelta)*k; 
		double max_c, max_v;
		double upower = p.xxi*pow((1-n),1.0-p.eeta);

		// Find out values on RHS
		double temp_max = -999999999999999.9;
		int max_kind = 0;
		int max_nind = 0;
		double max_V;
		for (int i_nplus = 0; i_nplus < nn; i_nplus++) {
			// Guess tomorrow employment
			double nplus = N[i_nplus];
			if (nplus < (1-p.x)*n) {
				continue;
			}
			double v = pow((nplus-(1-p.x)*n)/upower,1.0/p.eeta);
			double c;

			// Define temporary lambda function
			auto rhsvalue = [&] (int i_kplus) {
				double kplus = K[i_kplus];
				double net_proceed = asset - kplus; 
				c = net_proceed - p.kkappa*v;
				if (c < 0) {
					return -9e10;
				} else {
					return log(c) -p.ggamma*n + p.bbeta*EV[i_kplus+i_nplus*nk+i_A*nk*nn];
				};
			};

			// Create partial maximum value and indexes
			double partial_max;
			int partial_ind;

			// Binary search over kplus
			int ind1 = 0;
			int ind4 = nk-1;
			int ind2,ind3;
			double value1,value2,value3;
			while (ind4 - ind1 > 2) {
				ind2 = (ind4 + ind1)/2;
				ind3 = ind2 + 1;
				value2 = rhsvalue(ind2);
				value3 = rhsvalue(ind3);
				if (value2 < value3) {
					ind1 = ind2;
				} else {
					ind4 = ind3;
				}
			}
			value1 = rhsvalue(ind1);
			value2 = rhsvalue(ind4-1);
			value3 = rhsvalue(ind4);
			if (value1 < value2) {
				if (value2 < value3) {
					partial_max = value3;
					partial_ind = ind4;
				} else {
					partial_max = value2;
					partial_ind = ind4 - 1;
				}
			} else {
				if (value1 < value3) {
					partial_max = value3;
					partial_ind = ind4;
				} else {
					partial_max = value1;
					partial_ind = ind1;
				}
			}

			// Run rhsvalue again to update c
			rhsvalue(partial_ind);

			// Compare partial max with global max
			if (partial_max > temp_max) {
				temp_max = partial_max;
				max_nind = i_nplus;
				max_kind = partial_ind;
				max_c = c;
				max_v = v;
			}
		}
		max_V = temp_max;
		// =============== old brute force===============//
		// for (int i_kplus = 0; i_kplus < nk; i_kplus++) {
		// 	double kplus = K[i_kplus];
		// 	double net_proceed = asset - kplus; 
		// 	if (net_proceed < 0) {
		// 		continue;
		// 	};
		// 	for (int i_nplus = left_n_idx; i_nplus < nn; i_nplus++) {
		// 		double nplus = N[i_nplus];
		// 		if (nplus < (1-p.x)*n) {
		// 			continue;
		// 		};
		// 		double v = pow((nplus-(1-p.x)*n)/upower,1.0/p.eeta);
		// 		double c = net_proceed - p.kkappa*v;
		// 		if (c < 0) {
		// 			continue;
		// 		};
		// 		double rhs = log(c) -p.ggamma*n + p.bbeta*EV[i_kplus+i_nplus*nk+i_A*nk*nn];
		// 		if (rhs>temp_max) {
		// 			max_V = rhs;
		// 			max_kind = i_kplus;
		// 			max_nind = i_nplus;
		// 			max_c = c;
		// 			max_v = v;
		// 			temp_max = rhs;
		// 		};
		// 	}
		// };
		// =============== old brute force===============//
		Vplus[index] = max_V;
		koptind[index] = max_kind;
		noptind[index] = max_nind;
		copt[index] = max_c; 
		vopt[index] = max_v; 
		kopt[index] = K[max_kind];
		nopt[index] = N[max_nind];
	};
};

// This functor performs the policy update state, from policy to value function
struct policyupdate 
{
	// Data member
	//====Input pointers
	double *A, *K, *N, *EV;
	para p;
	//==== Ouput pointers
	double *copt, *kopt, *nopt, *vopt;
	int *koptind;
	int *noptind;
    double *Vplus;

	// Construct this object, create util from _util, etc.
	__host__ __device__
    policyupdate(double* A_ptr, double* K_ptr, double* N_ptr, double* EV_ptr, double* copt_ptr, double* vopt_ptr, double* kopt_ptr, double* nopt_ptr, int* koptind_ptr, int* noptind_ptr, double* Vplus_ptr, para _p) {
		A = A_ptr; K = K_ptr; N = N_ptr; EV = EV_ptr;
		copt = copt_ptr; kopt = kopt_ptr; nopt = nopt_ptr; vopt = vopt_ptr;
		koptind = koptind_ptr; Vplus = Vplus_ptr;
		noptind = noptind_ptr;
		p = _p;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int subs[3];
		int size_vec[3];
		size_vec[0] = nA;
		size_vec[1] = nk;
		size_vec[2] = nn;
		ind2sub(3,size_vec,index,subs);
		int i_A = subs[0];
		int i_k = subs[1];
		int i_n = subs[2];

		// Find and construct state and control, otherwise they won't update in the for loop
		double a = A[i_A]; 
		double k =K[i_k]; double n=N[i_n];
		int i_kplus = koptind[index];
		int i_nplus = noptind[index];
		double kplus = K[i_kplus];
		double nplus = N[i_nplus];
		double asset = a*pow(k/n,p.aalpha)*n + (1.0-p.ddelta)*k + p.z*(1-n);
		double upower = p.xxi*pow((1-n),1.0-p.eeta);
		double v = pow((nplus-(1-p.x)*n)/upower,1.0/p.eeta);
		double net_proceed = asset - kplus; 
		double c = net_proceed - p.kkappa*v;
		Vplus[index] =  log(c) -p.ggamma*n + p.bbeta*EV[i_kplus+i_nplus*nk+i_A*nk*nn];
	};
};

// This functor calculates the distance 
struct myDist {
	// Tuple is (V1low,Vplus1low,V1high,Vplus1high,...)
	template <typename Tuple>
	__host__ __device__
	double operator()(Tuple t)
	{
		return abs(thrust::get<0>(t)-thrust::get<1>(t));
	}
};

int main(int argc, char ** argv)
{
	// Set Model Parameters
	para p;
	p.bbeta = 0.999;
	p.aalpha = 0.33;
	p.xxi = 1.355;
	p.eeta = 0.4;
	p.ttau = 1.0-p.eeta;
	p.x = 0.0081;
	p.agg_u_target = p.x/(p.x+0.139);
	p.Qss = p.bbeta;
	p.ssigma = 0.03;
	p.rrho = 0.95;
	p.Abar = 1;
	p.kkappa = 0.913920491823929;
	p.z = 2.921277834700383;
	p.ggamma = 0.353783523150495;
	p.ddelta = 0.001540001540002;
	p.rr = 0.4;
	p.z_HM = 0.4;
	p.agg_ttheta = 1;
	p.agg_jfr = 0.139;
	p.i_y_target = 0.2;
	p.complete();	// compute steady state

	std::cout << std::setprecision(16) << "k_ss: " << p.k_ss << std::endl;
	std::cout << std::setprecision(16) << "n_ss: " << p.n_ss << std::endl;
	std::cout << std::setprecision(16) << "Ass: " << p.Abar << std::endl;
	std::cout << std::setprecision(16) << "tol: " << tol << std::endl;

	// Select Device
	// int num_devices;
	// cudaGetDeviceCount(&num_devices);
	if (argc > 1) {
		int gpu = atoi(argv[1]);
		cudaSetDevice(gpu);
	};

	// Only for cuBLAS
	const double alpha = 1.0;
	const double beta = 0.0;

	// Create all STATE, SHOCK grids here
	thrust::host_vector<double> h_A(nA);
	thrust::host_vector<double> h_K(nk); 
	thrust::host_vector<double> h_N(nn); 
	thrust::host_vector<double> h_logA(nA);
	thrust::host_vector<double> h_V(nk*nA*nn,0.0);
	thrust::host_vector<double> h_Vplus(nk*nA*nn,0);
	thrust::host_vector<double> h_EV(nk*nA*nn,0.0);
	thrust::host_vector<double> h_P(nA*nA, 0);
	thrust::host_vector<double> h_copt(nA*nk*nn,0.0);
	thrust::host_vector<double> h_vopt(nA*nk*nn,0.0);
	thrust::host_vector<double> h_kopt(nA*nk*nn,0.0);
	thrust::host_vector<double> h_nopt(nA*nk*nn,0.0);
	thrust::host_vector<int> h_koptind(nk*nA*nn);
	thrust::host_vector<int> h_noptind(nk*nA*nn);

	// Load previous results
    load_vec(h_V,"./results/Vguess.csv"); // in #include "cuda_helpers.h"

	// Create capital grid
	double minK = (1.0-kwidth)*p.k_ss;
	double maxK = (1.0+kwidth)*p.k_ss;
	linspace(minK,maxK,nk,thrust::raw_pointer_cast(h_K.data())); // in #include "cuda_helpers.h"

	// Create capital grid
	double minN = 0.97*p.n_ss;
	double maxN = 1.03*p.n_ss;
	linspace(minN,maxN,nn,thrust::raw_pointer_cast(h_N.data())); // in #include "cuda_helpers.h"

	// Create shocks grids
	thrust::host_vector<double> h_shockgrids(nA);
	double* h_logA_ptr = thrust::raw_pointer_cast(h_logA.data());
	double* h_P_ptr = thrust::raw_pointer_cast(h_P.data());
    tauchen(p.rrho, p.ssigma, nA, h_logA_ptr, h_P_ptr); // in #include "cuda_helpers.h"
	for (int i_shock = 0; i_shock < nA; i_shock++) {
		h_A[i_shock] = p.Abar*exp(h_logA[i_shock]);
	};
	save_vec(h_A,"./results/Agrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_P,"./results/Pgrid.csv"); // in #include "cuda_helpers.h"
    save_vec(h_K,"./results/Kgrid.csv"); // in #include "cuda_helpers.h"
    save_vec(h_N,"./results/Ngrid.csv"); // in #include "cuda_helpers.h"

	// Copy to the device
	thrust::device_vector<double> d_A = h_A;
	thrust::device_vector<double> d_K = h_K;
	thrust::device_vector<double> d_N = h_N;
	thrust::device_vector<double> d_V = h_V;
	thrust::device_vector<double> d_Vplus = h_Vplus;
	thrust::device_vector<double> d_copt = h_copt;
	thrust::device_vector<double> d_vopt = h_vopt;
	thrust::device_vector<double> d_kopt = h_kopt;
	thrust::device_vector<double> d_nopt = h_nopt;
	thrust::device_vector<int> d_koptind = h_koptind;
	thrust::device_vector<int> d_noptind = h_noptind;
	thrust::device_vector<double> d_EV = h_EV;
	thrust::device_vector<double> d_P = h_P;

	// Obtain device pointers to be used by cuBLAS
	double* d_A_ptr = raw_pointer_cast(d_A.data());
	double* d_K_ptr = raw_pointer_cast(d_K.data());
	double* d_N_ptr = raw_pointer_cast(d_N.data());
	double* d_V_ptr = raw_pointer_cast(d_V.data());
	double* d_Vplus_ptr = raw_pointer_cast(d_Vplus.data());
	double* d_copt_ptr = raw_pointer_cast(d_copt.data());
	double* d_vopt_ptr = raw_pointer_cast(d_vopt.data());
	double* d_kopt_ptr = raw_pointer_cast(d_kopt.data());
	double* d_nopt_ptr = raw_pointer_cast(d_nopt.data());
	int* d_koptind_ptr = raw_pointer_cast(d_koptind.data());
	int* d_noptind_ptr = raw_pointer_cast(d_noptind.data());
	double* d_EV_ptr = raw_pointer_cast(d_EV.data());
	double* d_P_ptr = raw_pointer_cast(d_P.data());

	// Firstly a virtual index array from 0 to nk*nk*nA
	thrust::counting_iterator<int> begin(0);
	thrust::counting_iterator<int> end(nk*nA*nn);

    // Create Timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // Start Timer
	cudaEventRecord(start,NULL);
	
	// Step.1 Has to start with this command to create a handle
	cublasHandle_t handle;

	// Step.2 Initialize a cuBLAS context using Create function,
	// and has to be destroyed later
	cublasCreate(&handle);

	// HH's problem
	double diff = 10;  int iter = 0;
	while ((diff>(1-p.bbeta)*tol)&&(iter<maxiter)){
		// Find EMs for low and high 
		// Compute EV = V'*P', where V is nA-by-(nk*nn)
		// i,j of EV is i_kplus,i_nplus,i_A
		cublasDgemm(handle,
				CUBLAS_OP_T,  
				CUBLAS_OP_T,
				nk*nn, nA, nA,
				&alpha,
				d_V_ptr, 
				nA, 
				d_P_ptr,
				nA,
				&beta,
				d_EV_ptr,
				nk*nn);

		// Solve the household's problem
		thrust::for_each(
				begin,
				end,
				update(d_A_ptr, d_K_ptr, d_N_ptr, d_EV_ptr, d_copt_ptr, d_vopt_ptr, d_kopt_ptr, d_nopt_ptr, d_koptind_ptr, d_noptind_ptr, d_Vplus_ptr, p)
				);

		// Find diff 
		diff = thrust::transform_reduce(
				thrust::make_zip_iterator(make_tuple(d_V.begin(),d_Vplus.begin())),
				thrust::make_zip_iterator(make_tuple(d_V.end()  ,d_Vplus.end())),
				myDist(),
				0.0,
				thrust::maximum<double>()
				);

		// update correspondence
		d_V = d_Vplus;
		++iter;
		if (iter % 1 == 0) {
			std::cout << "HH diff is: "<< diff << std::endl;
			std::cout << iter << std::endl;
			std::cout << "=====================" << std::endl;
		}

		// Policy update for k = 30
		// for (int p_step = 0; p_step < 21; p_step++) {
		// 	cublasDgemm(handle,
		// 			CUBLAS_OP_T,  
		// 			CUBLAS_OP_T,
		// 			nk*nn, nA, nA,
		// 			&alpha,
		// 			d_V_ptr, 
		// 			nA, 
		// 			d_P_ptr,
		// 			nA,
		// 			&beta,
		// 			d_EV_ptr,
		// 			nk*nn);

		// 	thrust::for_each(
		// 			begin,
		// 			end,
		// 			policyupdate(d_A_ptr, d_K_ptr, d_N_ptr, d_EV_ptr, d_copt_ptr, d_vopt_ptr, d_kopt_ptr, d_nopt_ptr, d_koptind_ptr, d_noptind_ptr, d_Vplus_ptr, p)
		// 			);

		// 	d_V = d_Vplus;
		// };
	};

	//==========cuBLAS stuff ends=======================
	// Step.3 Destroy the handle.
	cublasDestroy(handle);

	// Stop Timer
	cudaEventRecord(stop,NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal;
	std::cout << "Time= " << msecPerMatrixMul/1000.0 << " sec, iter= " << iter << std::endl;

	// Copy back to host and print to file
	h_V = d_V;
	h_EV = d_EV;
	h_koptind = d_koptind;
	h_noptind = d_noptind;
	h_copt = d_copt;
	h_vopt = d_vopt;
	h_nopt = d_nopt;

    save_vec(h_koptind,"./results/koptind.csv"); // in #include "cuda_helpers.h"
	save_vec(h_copt,"./results/copt.csv");
	save_vec(h_nopt,"./results/nopt.csv");
	save_vec(h_vopt,"./results/vopt.csv");
	save_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_EV,"./results/EVgrid.csv"); // in #include "cuda_helpers.h"
	std::cout << "Policy functions output completed." << std::endl;

	return 0;
}
