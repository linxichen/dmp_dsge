#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

// Tauchen's method
void tauchen(double rrho, double ssigma, int nz, double * Z, double * P) {
	double ssigma_z = sqrt( pow(ssigma,2)/(1-pow(rrho,2)) );
	int nzgrid = nz;
	Z[nzgrid-1] = 2.5*ssigma_z; Z[0] = -2.5*ssigma_z;
	double step = (Z[nzgrid-1] - Z[0])/ double(nzgrid-1);
	for (int i = 2; i <= nzgrid-1; i++) {
		Z[i-1] = Z[i-2] + step;
	};
    
	for (int i_z = 0; i_z <= nzgrid-1; ++i_z) {
	    P[i_z] = normcdf( (Z[0]-rrho*Z[i_z]+step/2)/ssigma  );
	    P[i_z + nzgrid*(nzgrid-1)] = 1 - normcdf( (Z[nzgrid-1]-rrho*Z[i_z]-step/2)/ssigma  );
	};
    
	for (int i_z = 0; i_z <= nzgrid-1; ++i_z) {
	    for (int i_zplus = 1; i_zplus <= nzgrid-2; ++i_zplus) {
            P[i_z+nzgrid*i_zplus] = normcdf( (Z[i_zplus]-rrho*Z[i_z]+step/2)/ssigma  )-normcdf( (Z[i_zplus]-rrho*Z[i_z]-step/2)/ssigma  );
	    };
	};
};

////////////////////////////////////////
//
/// Interpolation Stuff
//
////////////////////////////////////////
// Linear interpolation
template<typename T>
__host__ __device__
T linear_interp(T x, T x_left, T x_right, T f_left, T f_right) {
	if (abs(x_left-x_right)<1e-7) {
		return f_left;
	} else if ( (x_left > x_right) || (x < x_left) || (x > x_right) ) {
		return -9999999999.9999;
	} else {
		return f_left + (f_right-f_left)/(x_right-x_left)*(x-x_left);
	};
};

// Bilinear interpolation
template<typename T>
__host__ __device__
T bilinear_interp(T x, T y, T x_left, T x_right, T y_low, T y_high, T f_leftlow, T f_lefthigh, T f_rightlow, T f_righthigh) {
	T f_low = linear_interp<T>(x,x_left,x_right,f_leftlow,f_rightlow);
	T f_high = linear_interp<T>(x,x_left,x_right,f_lefthigh,f_righthigh);
	return linear_interp<T>(y,y_low,y_high,f_low,f_high);
};

// This function converts index to subscripts like ind2sub in MATLAB
__host__ __device__
void ind2sub(int length_size, int* siz_vec, int index, int* subs) {
// Purpose:		Converts index to subscripts. i -> [i_1, i_2, ..., i_n]
//
// Input:		length_size = # of coordinates, i.e. how many subscripts you are getting
// 				siz_vec = vector that stores the largest coordinate value for each subscripts. Or the dimensions of matrices
// 				index = the scalar index
//
// Ouput:		subs = the vector stores subscripts
	int done = 0;
	for (int i=length_size-1; i>=0; i--) {
		// Computer the cumulative dimension
		int cumdim = 1;
		for (int j=0; j<=i-1; j++) {
			cumdim *= siz_vec[j];
		};
		int temp_sub = (index - done)/cumdim;
		subs[i] = temp_sub; 
		done += temp_sub*cumdim;
	};
};

// This function fit a valuex x to a grid X of size n.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
__host__ __device__
int fit2grid(const double x, const int n, const double* X) {
	if (x < X[0]) {
		return 0;
	} else if (x > X[n-1]) {
		return n-1;
	} else {
		int left=0; int right=n-1; int mid=(n-1)/2;
		while(right-left>1) {
			mid = (left + right)/2;
			if (X[mid]==x) {
				return mid;
			} else if (X[mid]<x) {
				left = mid;
			} else {
				right = mid;
			};

		};
		return left;
	}
};

// This function fit a valuex x to a "even" grid X of size n. Even means equi-distance among grid points.
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
__host__ __device__
int fit2evengrid(const double x, const int n, const double min, const double max) {
	if (x <= min) return 0;
	if (x >= max) return n-1;
	double step = (max-min)/(n-1);
	return floor((x-min)/step);
};
// This function fit a valuex x to a grid X of size n. For std::vector like stuff
// The largest value on grid X that is smaller than x is returned ("left grid point" is returned).
template <class T>
int fit2grid(const double x, const T X) {
	int n = X.size();
	if (x < X[0]) {
		return 0;
	} else if (x > X[n-1]) {
		return n-1;
	} else {
		int left=0; int right=n-1; int mid=(n-1)/2;
		while(right-left>1) {
			mid = (left + right)/2;
			if (X[mid]==x) {
				return mid;
			} else if (X[mid]<x) {
				left = mid;
			} else {
				right = mid;
			};

		};
		return left;
	}
};

////////////////////////////////////////
//
/// Utilities for Vector 
//
////////////////////////////////////////
__host__ __device__
void linspace(double min, double max, int N, double* grid) {
	double step  = (max-min)/(N-1);
	for (int i = 0; i < N; i++) {
		grid[i] = min + step*i;
	};
};

// A function template to display vectors, C array style
template <class T>
void display_vec(T vec, int size) {
	for (int i = 0; i < size; i++) {
		std::printf("The %ith element, @[%i] = %f\n", i+1, i, vec[i]);
	};
};

// A function template to display vectors, std::vector style
template <class T>
void display_vec(T vec) {
	int size = vec.size();
	for (int i = 0; i < size; i++) {
		std::printf("The %ith element, @[%i] = %f\n", i+1, i, vec[i]);
	};
};

// A function template to save vectors to file, C array style
template <class T>
void save_vec(T vec, int size, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Saving to " << filename << std::endl;
	std::ofstream fileout(filename.c_str(), std::ofstream::trunc);
	for (int i = 0; i < size; i++) {
		fileout << std::setprecision(16) << vec[i] << '\n';
	};
	fileout.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;

};

// A function template to save vectors to file, std::vector style
template <class T>
void save_vec(T vec, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Saving to " << filename << std::endl;
	int size = vec.size();
	std::ofstream fileout(filename.c_str(), std::ofstream::trunc);
	for (int i = 0; i < size; i++) {
		fileout << std::setprecision(16) << vec[i] << '\n';
	};
	fileout.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;
};

// A function template to save vectors to file, C array style
template <class T>
void load_vec(T& vec, int size, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Loading from " << filename << std::endl;
	std::ifstream filein(filename.c_str());
	for (int i = 0; i < size; i++) {
		filein >> vec[i]; 
	};
	filein.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;
};

// A function template to save vectors to file, vector style
template <class T>
void load_vec(T& vec, std::string filename ) {
	std::cout << "================================================================================" << std::endl;
	std::cout << "Loading from " << filename << std::endl;
	std::ifstream filein(filename.c_str());
	double temp;
	int N = vec.size();
	for (int i = 0; i < N; i++) {
		filein >> temp;
		vec[i] = temp;
	};
	filein.close();
	std::cout << "Done!" << std::endl;
	std::cout << "================================================================================" << std::endl;
};

////////////////////////////////////////
// 
// Nonlinear Equation Solver
//
////////////////////////////////////////
// Find max over discrete arugments of a concave function 
// Input: concave function func returns value given integer index, usually a lambda function
// Output: maximizer max_ind and maxed value max_value
template<typename F>
__host__ __device__
void concavemax(int left_ind, int right_ind, F func, int& max_ind, double& max_value) {

	if (right_ind-left_ind==1) {
		double left_value, right_value;
		left_value = func(left_ind);
		right_value = func(right_ind);
		if (left_value>right_value) {
			max_value = left_value;
			max_ind = left_ind;
		} else {
			max_value = right_value;
			max_ind = right_ind;
		};
	} else if (right_ind-left_ind==2) {
		double value1 = func(left_ind);
		double value2 = func(left_ind+1);
		double value3 = func(right_ind);
		if (value1 < value2) {
			if (value2 < value3) {
				max_value = value3;
				max_ind= right_ind;
			} else {
				max_value= value2;
				max_ind = left_ind+1;
			}
		} else {
			if (value1 < value3) {
				max_value = value3;
				max_ind = right_ind;
			} else { 
				max_value = value1;
				max_ind = left_ind;
			}
		}
	} else {
		int ind1 = left_ind; int ind4 = right_ind;
		int ind2, ind3;
		double value1, value2, value3;
		while (ind4 - ind1 > 2) {
			ind2 = (ind1+ind4)/2;
			ind3 = ind2 + 1;
			value2 = func(ind2);
			value3 = func(ind3);
			if (value2 < value3) {
				ind1 = ind2;
			} else {
				ind4 = ind3;
			};
		};

		// Now the number of candidates is reduced to three
		value1 = func(ind1);
		value2 = func(ind4-1);
		value3 = func(ind4);

		if (value1 < value2) {
			if (value2 < value3) {
				max_value= value3;
				max_ind= ind4;
			} else {
				max_value= value2;
				max_ind= ind4-1;
			}
		} else {
			if (value1 < value3) {
				max_value= value3;
				max_ind= ind4;
			} else { 
				max_value= value1;
				max_ind= ind1;
			}
		}
	}
};



// Newton's Method with bracketing, i.e. we know on two points the function differs in sign.
// Codes from Numerical Recipes 3rd. Ed.
// BEWARE: The stopping criteria is not right yet.
template <class T>
__host__ __device__
double newton_bracket(T func, const double x1, const double x2, double x0) {
// Purpose: Tries to find a root for function named func. Its first derivative is given by func.prime().
//			It is assumed that func(x1) and func(x2) are different in sign so a root exists within. x0 is the guess.
	const int newton_maxiter = 100;
	const double newton_tol = 1e-3;
	// Checking the bounds: they need to make sense. Or sometimes the bounds are solutions.
	double f1 = func(x1);
	double f2 = func(x2);
	if (f1*f2>0) return -5179394.1; // The different sign assumption violated!
	if (f1 == 0) return x1;
	if (f2 == 0) return x2;

	// Orient the search so that f(xl) < 0
	double xl, xh;
	if (f1 < 0.0) {
		xl = x1;
		xh = x2; 
	} else {
		xh = x1;
		xl = x2;
	};

	// Initialize guess and other things
	double rts = x0;
	double dxold = abs(x2-x1);
	double dx = dxold;
	double f = func(rts);
	double df = func.prime(rts);

	for (int iter = 0; iter < newton_maxiter; iter++) { 
		if ( 
			( ((rts-xh)*df-f)*((rts-xl)*df-f) > 0.0 )   ||	// Bisect if Newton step out of range
			( abs(2.0*f) > abs(dxold*df)  ) // ... or step not decreasing fast enough
		   )
		{
			dxold = dx;
			dx = 0.5*(xh-xl);
			rts += dxold; // undo the newton step
			rts = xl + dx;
			if (xl == rts) return rts;
		} else {
			// If newton step is okay 
			dxold = dx;
			dx = f/df;
			double temp = rts;
			rts -= dx;
			if (temp==rts) return rts;
		};

		// Check for convergence
		if ( abs(dx)/(1+abs(rts+dx)) < newton_tol ) return rts;

		// Compute new f and df for next iteration
		f = func(rts);
		df = func.prime(rts);

		// Maintain the bracket
		if (f < 0.0) {
			xl = rts;
		} else {
			xh = rts;
		};
	};

	return -51709394.2;
};

// "Raw" Newton's Method
// Codes from Numerical Recipes 3rd. Ed.
template <class T>
__host__ __device__
double newton(T func, const double x1, const double x2, double x0) {
// Purpose: Tries to find a root for function named func. Its first derivative is given by func.prime().
//			func is only defined on [x1,x2] We "pull back" when outside. x0 is the guess.
	const int newton_maxiter = 50;
	const double newton_tol = 1e-4;
	// Initialize guess and other things
	double x_old = x0;
	double x = x0;
	double f1 = func(x1);
	double f2 = func(x2);
	if (f1==0) return x1;
	if (f2==0) return x2;
	for (int iter = 0; iter < newton_maxiter; iter++) { 
		x = x_old - func(x)/func.prime(x);

		// Pull back if outside of support
		if (x<=x1) {
			return -51709394.2;
		};
		if (x>=x2) {
			return -51709394.2;
		};

		// Check for convergence
		if ( (abs(x-x_old)/(1+abs(x_old))<newton_tol) && (abs(func(x)) < newton_tol) ) {
		   	return x;
		} else {
			x_old = x;
		};
	};
	return -51709394.2;
};

////////////////////////////////////////
// 
// Chebyshev Toolset
//
////////////////////////////////////////
// Evaluate Chebychev polynomial of any degree
__host__ __device__
double chebypoly(const int p, const double x) {
	switch (p) {
		case 0: // 0-th order Chebyshev Polynomial
			return 1;
		case 1:
			return x;
		case 2:
			return 2*x*x - 1;
		case 3:
			return 4*x*x*x - 3*x;
	}
	
	// When p>=4, apply the recurrence relation
	double lag1 = 4*x*x*x -3*x;
	double lag2 = 2*x*x - 1;
	double lag0;
	int distance = p - 3;
	while (distance >= 1) {
		lag0 = 2*x*lag1 - lag2;
		lag2 = lag1;
		lag1 = lag0;
		distance--;
	};
	return lag0;
};

// Evaluate Chebychev polynomial of any degree
__host__ __device__
int chebyroots(const int p, double* roots) {
	for (int i=0; i<p; i++) {
		double stuff = p - 0.5 - 1*i;
		roots[i] = cos(M_PI*(stuff)/(p));
	};

	// Account for the fact that cos(pi/2) is not exactly zeros
	if (p%2) {
		roots[(p-1)/2] = 0;
	};
	return 0;
};

// Evaluate Chebychev approximation of any degree
__host__ __device__
double chebyeval(int p, double x, double* coeff) {
	// Note that coefficient vector has p+1 values
	double sum = 0;
	for (int i=0; i<=p; i++) {
		sum += coeff[i]*chebypoly(i,x);	
	};
	return sum;
};

// Eval multi-dimensional Chebyshev tensor basis 
// y = sum T_pi(x_i), i = 1,2,...p
__host__ __device__
double chebyeval_multi (const int n_var, double* x, int* size_vec,int* temp_subs, double* coeff) {
	// Note size_vec's elements are p+1 for each var
	int tot_deg = 1;
	for (int i = 0; i < n_var; i++) {
		tot_deg *= (size_vec[i]); // Note there's p+1 coeffs
	};

	double eval = 0;
	for (int index = 0; index < tot_deg; index++) {
		// Perform ind2sub to get current degrees for each var
		ind2sub(n_var, size_vec, index, temp_subs);

		// Find the values at current degrees
		double temp = 1;
		for (int i = 0; i < n_var; i++) {
			// printf("%i th subscript is %i\n",i,temp_subs[i]);
			temp *= chebypoly(temp_subs[i],x[i]);
		};

		// Add to the eval
		eval += (coeff[index]*temp);
	};
	return eval;
};
