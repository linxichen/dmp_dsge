// Define an class that contains parameters and steady states
struct para {
	// Model parameters
	double aalpha;			// share of capital
	double bbeta ;			// discount rate
	double ddelta;			// depreciation rate of capital
	double xxi;				// matching efficiency
	double eeta;			// share of vancany in matching function
	double ttau;			// worker's bargainning power
	double x;				// exo separation rate
	double Abar  ;
	double rrho;
	double ssigma;
	/* The following parameters are fsolve in MATLAB in Nir's code */
	double kkappa;			// unit vacancy cost
	double z;				// unemployment benefit
	double ggamma;			// disutility coefficient

	// Steady States and Target moments
	double agg_u_target;	// ??? aggregate unemployment rate
	double Qss;				// SDF in SS
	double rr;				// replacement ratio
	double z_HM;			// to match the leisure value as a fraction of wage Hall Milgrom
	double agg_ttheta;		// average tightness ratio (normalization)
	double agg_jfr;			// average job finding rate
	double i_y_target;		// target investment over output ratio
	double n_ss;			// employment
	double ttheta_ss;		// tightness
	double k_ss;			// capital
	double r_ss;			// return rate
	double oomega_ss;		// wage

	// Find steady state and update parameters based on steady state target
	__host__ __device__
	void complete() {
		r_ss = 1/bbeta - 1 + ddelta;
		double k_n_ss = pow(r_ss/(Abar*aalpha),1/(aalpha-1));
		double v_ss = agg_ttheta*agg_u_target;
		n_ss = 1 - agg_u_target;
		ttheta_ss = agg_ttheta;
		double mmu_ss = agg_jfr;
		xxi = mmu_ss/pow(agg_ttheta,eeta);
		double q_ss = xxi*pow(agg_ttheta,eeta-1);
		k_ss = k_n_ss*n_ss;
		// double inv_ss = ddelta*k_ss;
		double y_ss = Abar*pow(k_ss,aalpha)*pow(n_ss,1-aalpha);
		double c_ss = y_ss - ddelta*k_ss - kkappa*v_ss;
		oomega_ss = ttau*Abar*(1-aalpha)*pow(k_n_ss,aalpha) + (1-ttau)*(z + ggamma*c_ss) + ttau*kkappa*ttheta_ss;
		double const1 = 1-(1-x)*bbeta;
		double Jn_ss = ((Abar*(1-aalpha))*pow(k_n_ss,aalpha) - oomega_ss)/const1;
		// double rep_ratio = z/oomega_ss;
		// double leisure_value = (ggamma*c_ss)/oomega_ss;
		// double i_y_ss = inv_ss/(y_ss);
		// double lp_agg_ss = y_ss/n_ss;
	};

	// Export parameters to a .m file in MATLAB syntax
	__host__
	void exportmatlab(std::string filename) {
		std::ofstream fileout(filename.c_str(), std::ofstream::trunc);

		// Model Parameters
		fileout << std::setprecision(16) << "bbeta=" << bbeta << ";"<< std::endl;
		fileout << std::setprecision(16) << "ddelta=" << ddelta << ";"<< std::endl;
		// fileout << std::setprecision(16) << "ttheta=" << ttheta << ";"<< std::endl;
		// fileout << std::setprecision(16) << "zbar=" << zbar << ";"<< std::endl;
		// fileout << std::setprecision(16) << "rrhozz=" << rrhozz << ";"<< std::endl;
		// fileout << std::setprecision(16) << "ssigmaepsz=" << std_epsz << ";"<< std::endl;

		// Steady States
		fileout << std::setprecision(16) << "kss=" << k_ss << ";"<< std::endl;
		// fileout << std::setprecision(16) << "css=" << c_ss << ";"<< std::endl;
		fileout.close();
	};
};

// Define state struct that contains "natural" state 
struct state {
	// Data member
	double A, k, n;
	double MPK, MPN, Y;

	// Constructor
	__host__ __device__
	state(double _A, double _k, double _n, para p) {
		A = _A;
		k = _k;
		n = _n;
		Y = _A*pow(_k/_n,p.aalpha)*_n;
		MPK = p.aalpha*Y/_k;
		MPN = (1.0-p.aalpha)*Y/_n;
	};
};

// Define state struct that contains extra state variables, shadow values 
struct shadow {
	// Data member
	double mf, mh;

	// Constructor
	__host__ __device__
	shadow(double _mh, double _mf) {
		mf = _mf;
		mh = _mh;
	};
};

// The function and its derivative to be solved. To find ttheta
struct findttheta {
	// Data members, constants and coefficients
	double czero, coneminuseeta, cone;
	double eeta;

	// Constructor
	__host__ __device__
	findttheta(state s, shadow m, double c, para p) {
		coneminuseeta = (1.0-p.x)*p.kkappa/p.xxi;
		cone = -p.ttau*p.kkappa;
		czero = (1-p.ttau)*(s.MPN - p.z - p.ggamma*c) - m.mf*c;
		eeta = p.eeta;
	};

	// Value operator
	__host__ __device__
	double operator()(double ttheta) {
		return czero + coneminuseeta*pow(ttheta,1.0-eeta) + cone*ttheta;
	};

	// Derivative
	__host__ __device__
	double prime(double ttheta) {
		return (1.0-eeta)*coneminuseeta*pow(ttheta,-eeta) + cone;
	};
};	

// Define control struct that contains jump variables, shadow values 
struct control {
	// Data member
	double c, kplus, nplus, ttheta, v, q, mmu;

	// Constructor
	__host__ __device__
	void compute(state s, shadow m, para p) {
		c = (1.0-p.ddelta+s.MPK)/m.mh;
		ttheta = newton(findttheta(s,m,c,p),0.0,200.0,1.0);
		v = ttheta*(1.0-s.n);
		mmu = p.xxi*pow(ttheta,p.eeta);
		q = mmu/ttheta;
		nplus = (1-p.x)*s.n + mmu*(1-s.n);
		kplus = s.Y - p.kkappa*v + (1-p.ddelta)*s.k - c;
	};
};

