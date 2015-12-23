#ifndef UTILS_H
#define UTILS_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <Eigen>
//#include "Eigen"
#include <ctime>
#include <random>
#include <cstdlib>
#include <omp.h>
#include <string>
#include <Windows.h>
#include "liblinear-2.1\linear.h"
const int Nfp = 74;
const int Nimg =10858;
const int Nadd = 7258;
const int Ntest = 9888;
//const int Nimg = 100;
const int N_aug = 10;
const int observation_bin = 3;
const int stage = 1;
const double radius[5] = { 0.4, 0.25, 0.2, 0.16, 0.1 };
typedef Eigen::Matrix<double, Nfp, 2> landmark;
typedef Eigen::Matrix<double, 2, 2> transformer;
typedef Eigen::Matrix<double, Nimg*N_aug, 2> regression_err;
//typedef double *regression_errx;
//typedef double *regression_erry;
typedef Eigen::Matrix<double, Ntest, 2> delta_y;
typedef struct trans_matrix{
	double sint;
	double cost;
	Eigen::Vector2d translation;
	double scale;
	//landmark normal;
};
typedef struct refershape{
	landmark shape;
	Eigen::Vector2d translation;
	double scale;
};
typedef struct treenode{
	int split_point;
	std::vector<int> left_child;
	std::vector<int> right_child;
	double threshold;
	Eigen::Vector2d output;
} rfs;
typedef struct boundingbox{
	int centerx;
	int centery;
	int width;
	int height;
};
typedef std::vector<rfs> randomtree;
typedef std::vector<randomtree> randomforest;

rfs split_node(const Eigen::MatrixXd& feat, const regression_err& Y, const rfs* parent, const std::vector<int>& selected_feature, int min_num);
randomforest train_rfs(const Eigen::MatrixXd& feat, const regression_err&Y, const int num_trees, const int max_depth, const int num_feat, const int num_selected_sample, const int num_selected_feature);
void test_rfs(const int Nsample,const randomforest & r, const Eigen::MatrixXd& feat, struct feature_node **global_binary_features, const int num_landmark, const int num_trees, const int max_depth, const int offset=0);
#endif // !UTILS_H
