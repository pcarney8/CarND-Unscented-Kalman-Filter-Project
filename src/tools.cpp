#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	int estimationsSize = estimations.size();

	// check that the estimation vector size should not be zero and they are equal
	if(estimationsSize == 0 || estimationsSize != ground_truth.size()){
	    cout << "AH, THEY WEREN'T THE RIGHT SIZE" << endl;
	    return rmse;
	}

	//accumulate squared residuals
	for(int i=0; i < estimationsSize; ++i){
        cout << "estimations[" << i << "]: " << endl << estimations[i] << endl;
        cout << "ground_truth[" << i << "]: " << endl << ground_truth[i] << endl;

        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array().square();

        rmse += residual;
        cout << "step: " << i << endl;
        cout << "rmse: " << rmse << endl;
	}

	// calculate the mean
    rmse = rmse.array()/estimationsSize;

	// calculate the squared root
	rmse = rmse.array().sqrt();

    // return the result
	return rmse;

}
