#include "utils.h"

using namespace cv;
using namespace std;

vector<landmark> S_t;
vector<Mat> full_color_test_img;
//delta_y* ytest[Nfp];
//Eigen::MatrixXd feat;
//rfs root;
//return shape 0


inline double interp1(const Mat_<uchar>&img,double x, double y){
	//srand(unsigned(time(NULL)));

	int x0 = floor(x);
	int x1 = ceil(x);
	int y0 = floor(y);
	int y1 = ceil(y);
	//cout << x << " " << y << endl;
	//cout << img[img_idx].cols << " " << img[img_idx].rows << endl;
	x = x - x0;
	y = y - y0;
	if (x0 >= img.cols)
		x0 = img.cols - 1;
	else if (x0 < 0)
		x0 = 0;
	if (y0 >= img.rows)
		y0 = img.rows - 1;
	else if (y0 < 0)
		y0 = 0;

	if (x1 >= img.cols)
		x1 = img.cols - 1;
	else if (x1 < 0)
		x1 = 0;
	if (y1 >= img.rows)
		y1 = img.rows - 1;
	else if (y1 < 0)
		y1 = 0;

	double b1, b2, b3, b4;
	b1 = img(y0, x0);
	b2 = img(y0, x1) - b1;
	b3 = img(y1, x0) - b1;
	b4 = img(y1, x1) - img(y0, x1) - img(y1, x0) + b1;
	return b1 + b2*x + b3*y + b4*x*y + x*y;
}
inline double interp2(const vector<Mat_<uchar>>&img,int img_idx, double x, double y){
	//srand(unsigned(time(NULL)));
	
	int x0 = floor(x);
	int x1 = ceil(x);
	int y0 = floor(y);
	int y1 = ceil(y);
	//cout << x << " " << y << endl;
	//cout << img[img_idx].cols << " " << img[img_idx].rows << endl;
	x = x - x0;
	y = y - y0;
	if (x0 >= img[img_idx].cols)
		x0 = img[img_idx].cols - 1;
	else if (x0 < 0)
		x0 = 0;
	if (y0 >= img[img_idx].rows)
		y0 = img[img_idx].rows - 1;
	else if (y0 < 0)
		y0 = 0;

	if (x1 >= img[img_idx].cols)
		x1 = img[img_idx].cols - 1;
	else if (x1 < 0)
		x1 = 0;
	if (y1 >= img[img_idx].rows)
		y1 = img[img_idx].rows - 1;
	else if (y1 < 0)
		y1 = 0;
	
	double b1, b2, b3, b4;
	b1 = img[img_idx](y0, x0);
	b2 = img[img_idx](y0, x1)-b1;
	b3 = img[img_idx](y1, x0)-b1;
	b4 = img[img_idx](y1, x1) - img[img_idx](y0, x1) - img[img_idx](y1, x0) + b1;
	return b1 + b2*x + b3*y + b4*x*y + x*y;
}
trans_matrix procrustes(const refershape& la,const landmark& lb){
	Eigen::Vector2d mean_x;
	double tanx, tany, sint, cost;
	landmark tmp, norm_tmp=la.shape;
	trans_matrix trans_tmp;
	double stdx;
	norm_tmp.rowwise() -= la.translation.transpose();
	norm_tmp /= la.scale;
	mean_x = lb.colwise().mean();
	tmp = lb.rowwise() - mean_x.transpose();
	stdx = sqrt(tmp.squaredNorm() / Nfp);
	tmp /= stdx;
	tany = tmp.col(0).dot(norm_tmp.col(1)) - tmp.col(1).dot(norm_tmp.col(0));
	tanx = tmp.col(0).dot(norm_tmp.col(0)) + tmp.col(1).dot(norm_tmp.col(1));
	sint = tany / sqrt(pow(tany, 2) + pow(tanx, 2));
	cost = tanx / sqrt(pow(tany, 2) + pow(tanx, 2));
	//norm_tmp.col(0) = la.scale*(tmp.col(0)*cost - tmp.col(1)*sint);
	//norm_tmp.col(1) = la.scale*(tmp.col(0)*sint + tmp.col(1)*cost);
	//norm_tmp.rowwise() += la.translation.transpose();
	//cout << tmp << norm_tmp << endl;
	trans_tmp.sint = sint;
	trans_tmp.cost = cost;
	trans_tmp.translation = la.translation - mean_x;
	trans_tmp.scale = la.scale / stdx;
	//trans_tmp.normal = norm_tmp;
	return trans_tmp;
}
void imageRead(vector<Mat_<uchar>>& img,vector<landmark>& shape, const string address, const int N_img){
	//string file_names = "train_file.txt";
	//string img_names = "train_img.txt";
	string file_names = "train_file.txt";
	string img_names = "train_imgc.txt";
	//string file_names = "test_file.txt";
	//string img_names = "test_img.txt";

	ifstream fin_img, fin_pts, fin_points;
	
	fin_img.open(img_names);
	fin_pts.open(file_names);
	string name,nothing;
	double x, y;
	landmark temp;
	
	if (!fin_img.is_open())
		cout << "Error!" << endl;
	for (int i = 0; i < N_img; ++i){
		fin_img >> name;
		cout << name << endl;
		
		img.push_back(imread(address + name, CV_LOAD_IMAGE_GRAYSCALE));
		//if (N_img==Ntest)
		//	full_color_test_img.push_back(imread(address + name, CV_LOAD_IMAGE_COLOR));
		//imshow("test", img[i]);
		//waitKey(0);
		fin_pts >> name;
		
		//imshow("test", img[i]);
		//getchar();
		//cout << img[i].size() << endl;
		
		//cout << name << endl;
		//cout << address + name << endl;
		fin_points.open(address + name);
		fin_points >> nothing >> nothing >> nothing>>nothing>>nothing;
		if (!fin_points.is_open())
			cout << "No points file" << endl;
		//cout << "hehe"<<nothing << endl;
		for (int j = 0; j < temp.rows(); ++j){
			fin_points >> x >> y;
			/*if (i >= Nadd){
				x *= 0.520833;
				y *= 0.520833;
			}*/
			//cout << x << " " << y << endl;
			//cout << j << endl;
			temp(j, 0) = x;
			temp(j, 1) = y;
			//getchar();
			//cout << temp(j, 0) << " " << temp(j, 1) << endl;
		}
		shape.push_back(temp);
		fin_points.close();
	}
	fin_img.close();
	fin_pts.close();
}
refershape generateMeanshape(const vector<Mat_<uchar>>&img,const vector<landmark>& S){
	int iter_times = 5;
	int init_face;
	double stdx,scale_norm = 0;
	double tanx, tany,sint,cost;
	refershape meanface;
	Eigen::Vector2d mean_x,mean_norm;
	vector<landmark> Stmp=S;
	landmark meanshape, norm_tmp, tmp, sumshape;
	trans_matrix trans_tmp;
	//Scalar mean_x, std_x;
	int n = img.size();
	//cout << n << endl;
	//srand(time(NULL));
	init_face = rand() % n;
	cout << init_face << endl;
	meanshape = S[init_face];
	for (int iter = 1; iter <= iter_times; ++iter){
		tmp = meanshape;
		mean_norm = tmp.colwise().mean();
		tmp = tmp.rowwise() - mean_norm.transpose();
		sumshape.setZero();
		/*
		imshow("test", img[0]);
		waitKey(0);*/
		scale_norm = sqrt(tmp.squaredNorm() / Nfp);
		tmp /= scale_norm;
		if (iter == iter_times)
			break;
		//Stmp[init_face] /= scale_norm;
		for (int i = 0; i < Nimg; ++i){
			if (iter == 1 && i == init_face){
				sumshape += meanshape;
				continue;
			}
			//mean_x=S[i].colwise().mean();
			//mean_y = S[i].col(1).mean();
			
			mean_x = S[i].colwise().mean();
			Stmp[i] = S[i].rowwise() - mean_x.transpose();
			stdx = sqrt(Stmp[i].squaredNorm() / Nfp);
			Stmp[i] /= stdx;
			tany = Stmp[i].col(0).dot(tmp.col(1)) - Stmp[i].col(1).dot(tmp.col(0));
			tanx = Stmp[i].col(0).dot(tmp.col(0)) + Stmp[i].col(1).dot(tmp.col(1));
			sint = tany / sqrt(pow(tany, 2) + pow(tanx, 2));
			cost=tanx/ sqrt(pow(tany, 2) + pow(tanx, 2));
			norm_tmp.col(0) = scale_norm*(Stmp[i].col(0)*cost - Stmp[i].col(1)*sint);
			norm_tmp.col(1) = scale_norm*(Stmp[i].col(0)*sint + Stmp[i].col(1)*cost);
			norm_tmp.rowwise() += mean_norm.transpose();
			sumshape += norm_tmp;
			/*if (iter == iter_times){
				trans_tmp.sint = sint;
				trans_tmp.cost = cost;
				trans_tmp.translation = mean_norm - mean_x;
				trans_tmp.scale = scale_norm / stdx;
				trans_tmp.normal = tmp;
				trans.push_back(trans_tmp);
			}*/
			/*for (int k = 0; k < Nfp; k++){
				Point p;
				p.x = tmp(k, 0);
				p.y = tmp(k, 1);
				circle(img[i], p, 3, Scalar(0, 0, 0));
			}
			for (int k = 0; k < Nfp; k++){
				Point p;
				p.x = meanshape(k, 0);
				p.y = meanshape(k, 1);
				circle(img[i], p, 3, Scalar(1, 1, 0));
			}
			imshow("test", img[i]);*/

			//cout << Stmp[i] << endl;
			//cout << S[i].rowwise().mean() << endl;
			//cout << mean_x<<std_x << endl;
			//cout << mean(Stmp[i].col(1)) << endl;
			//waitKey(0);
			//getchar();
			//return;
		}
		meanshape = sumshape / Nimg;
	}
	meanface.shape = meanshape;
	meanface.translation = mean_norm;
	meanface.scale = scale_norm;
	return meanface;
}
refershape readMeanshape(ifstream& f_shape){
	refershape t;
	string Meanshape, Translation, Scale;
	f_shape >> Meanshape;
	f_shape >> Translation >> t.translation(0) >> t.translation(1);
	f_shape >> Scale >> t.scale;
	for (int i = 0; i < Nfp; ++i)
		f_shape >> t.shape(i, 0) >> t.shape(i, 1);
	return t;
}
vector<landmark> initial_face(int N, int N_aug, const vector<Mat_<uchar>>& img,const vector<landmark>& init_set){
	int NMAX = init_set.size();
	vector<landmark> init;
	CascadeClassifier face_cascade;
	string face_cascade_name = "D:\\haomaiyi\\Facial_landmark\\src\\Landmarkdetector\\Project1\\haarcascade_frontalface.xml";
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading face cascade\n");
	}
	Mat face_equal;
	vector<transformer> faces;
	for (int i = 0; i < N; ++i){
		transformer tmp;
		tmp.row(0) = init_set[i].colwise().mean();
		tmp.row(1) = init_set[i].colwise().maxCoeff() - init_set[i].colwise().minCoeff();
		faces.push_back(tmp);
	}
	int idx;
	
	//srand(time(NULL));
	for (int i = 0; i < N; ++i){
		vector<Rect> temp;
		cv::equalizeHist(img[i], face_equal);
		face_cascade.detectMultiScale(face_equal, temp, 1.1, 2, 0, Size(30, 30));
		for (int j = 0; j < N_aug; ++j){
			idx = rand() % NMAX;
			landmark t_landmark = init_set[idx];
			int max_i = 0;
			if (temp.size()>1){
				double max_size = 0;
				for (int s = 0; s < temp.size();++s)
				if (temp[s].width>max_size){
					max_size = temp[s].width;
					max_i = s;
				}
			}
			if (temp.size()){
				t_landmark.rowwise() -= faces[idx].row(0);
				t_landmark.col(0) /= faces[idx](1, 0);
				t_landmark.col(1) /= faces[idx](1, 1);

				t_landmark.col(0) *= temp[max_i].width;
				t_landmark.col(1) *= temp[max_i].height;
				//cout << temp[0].x << temp[0].y << endl;
				t_landmark.rowwise() += Eigen::Vector2d(temp[max_i].x + temp[max_i].width / 2, temp[max_i].y + temp[max_i].height / 2).transpose();
				//}
			}
			/*
			for (int k = 0; k < Nfp; k++){
			Point p;
			//p.x = init_set[idx](k, 0);
			//p.y = init_set[idx](k, 1);
			p.x = t_landmark(k, 0);
			p.y = t_landmark(k, 1);
			circle(img[i], p, 3, Scalar(255, 0, 0));
			}*/
			init.push_back(t_landmark);
		}
		//faces.push_back(temp[0]);
		/*for (size_t j = 0; j < faces.size(); j++)
		{
		rectangle(img[i], faces[j], Scalar(255, 0, 0), 2, 8);
		}*/
		//cout << faces.size() << endl;
		//cout << init_set[idx].colwise().mean() << endl;
		//cout << init_set[idx].colwise().maxCoeff() << endl;
		//cout << init_set[idx].colwise().minCoeff() << endl;
		//init_set[idx].colwise().maxCoeff()-init_set[idx].colwise().minCoeff()
		//faces[0].x + faces[0].width/2
		//faces[0].y + faces[0].height/2
		
	}

	return init;
}
vector<landmark> test_face(int Ntest, int Nimg, const vector<Mat_<uchar>>& img, refershape& meanshape){
	//int NMAX = init_set.size();
	vector<landmark> init;
	CascadeClassifier face_cascade;
	String face_cascade_name = "haarcascade_frontalface.xml";
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading face cascade\n");
	}
	Mat face_equal;
	//ifstream init_face("test_init.txt");
	//srand(time(NULL));
	for (int i = 0; i < Ntest; ++i){
		vector<Rect> temp;
		equalizeHist(img[i], face_equal);
		face_cascade.detectMultiScale(face_equal, temp, 1.1, 2, 0, Size(20, 20));
		landmark t_landmark = meanshape.shape;
		double width = 0.8*t_landmark.colwise().mean()(0);
		double height = 0.8*t_landmark.colwise().mean()(1);
		if (temp.size()){
			t_landmark.rowwise() -= meanshape.translation.transpose();
			t_landmark.col(0) /= width;
			t_landmark.col(1) /= height;

			
			t_landmark.col(1) *= temp[0].height;
			t_landmark.col(0) *= temp[0].width;
			//cout << temp[0].x << temp[0].y << endl;
			t_landmark.rowwise() += Eigen::Vector2d(temp[0].x + temp[0].width / 2, temp[0].y +temp[0].height/2).transpose();
			
		}
		init.push_back(t_landmark);
	}
	return init;
}
Eigen::MatrixXd GenerateShapeIndexFeature(Eigen::MatrixXd& feat, const vector<Mat_<uchar>>&img,const vector<landmark>& S_t, const trans_matrix* trans, int num_landmark, int N, int N_aug, int P, double radius){
	Eigen::MatrixXd pts;
	vector<double> pixels;
	//srand(time(NULL));
	pts.resize(P, 2);
	double x, y,w,z;
	int count=0,Ns;
	Ns = round(1.28*P + 2.5*sqrt(P) + 100);
	//cout << Ns << endl;
	//int max = rt.max;
	//cout << rt.max << endl;
	for (int i = 0; i < Ns; ++i){
		x= (double)rand()/RAND_MAX *2 * radius - radius;
		y = (double)rand() / RAND_MAX * 2 * radius - radius;
		//if (i==100)
		//	cout << x << y << endl;
		if (x*x + y*y <= radius*radius){
			pts(count, 0) = x;
			pts(count++, 1) = y;
		}
		if (count>=P)
			break;
	}
	if (count < P-1){
		cout << "Error,insufficient samples!" << endl;
	}
	
	double sinx, cosx;
	//vector<double> ori_x, ori_y;
	double ori_x, ori_y,scale;
	//double scale;
	double t;
	//cout << "Here count:" <<count<< endl;
	for (int i = 0; i < N*N_aug; ++i){;
		sinx = trans[i].sint;
		cosx = trans[i].cost;
		scale = 1/trans[i].scale;
		if (pixels.size())
			cout << "Error, pixels should be zero!" << endl;
		//cout << tanx << " " << cotx << " " << sinx << " " << cosx << " " << scale << endl;
		for (int j = 0; j < P; ++j){
			ori_x = S_t[i](num_landmark, 0) + scale * (pts(j, 0) * cosx + pts(j, 1) * sinx);
			ori_y = S_t[i](num_landmark, 1) + scale * (pts(j, 1) * cosx - pts(j, 0) * sinx);
			//if (ori_x>=250||ori_y>=250)
			//	cout << i << "¡¡" << ori_x << " " << ori_y << endl;
			pixels.push_back(interp2(img,int(i/N_aug), ori_x, ori_y));
			//ori_pts(j, 0) = ori_x ;
			//ori_pts(j, 1) = ori_y ;
			//feat(i, j) = img[i]()
		}
		count = 0;
		for (int px = 0; px < P - 1; ++px)
			for (int py = px + 1; py < P; ++py){
				/*if (pixels[px] - pixels[py] == 0){
					cout << pixels[px] << " " << pixels[py] << endl;
					cout << "px" << px << "::" <<ori_x[px] << " " << ori_y[px] << "py" << py<< "::"<<ori_x[py] << " " << ori_y[py] << endl;
					getchar();
				}*/
				feat(i, count++) = pixels[px] - pixels[py];
			}
		pixels.clear();
		//ori_x.clear();
		//ori_y.clear();
		//cout << "count:"<<count << endl;
		//cout << pixels.size() << endl;
	}
	return pts;
}
void updateError(regression_err* Y,int N, int Naug, int num_landmark, const vector<landmark>&S0, const vector<landmark>&St, const trans_matrix* trans){
	landmark delta; // = S0 - St;
	landmark norm_delta;
	for (int i = 0; i < N; ++i){
		for (int j = 0; j < Naug; ++j){
			int idx = i*Naug + j;
			delta = S0[i] - St[idx];
			norm_delta.col(0) = trans[idx].scale*(delta.col(0)*(trans[idx].cost) - delta.col(1)*(trans[idx].sint));
			norm_delta.col(1) = trans[idx].scale*(delta.col(0)*(trans[idx].sint) + delta.col(1)*(trans[idx].cost));
			//cout << Y[j]->rows << endl;
			for (int t = 0; t < num_landmark; ++t)
				//Y[j]->setZero();
				Y[t].row(idx) = norm_delta.row(t);
		}
	}
}
void updateShape(int N, vector<landmark>& S_t, const Eigen::MatrixXd& y, const trans_matrix* trans, const int num_landmark){
	double tanx, cotx, sinx, cosx, scale;
	for (int i = 0; i < N; ++i){
		sinx = trans[i].sint;
		cosx = trans[i].cost;
		scale = 1 / trans[i].scale;
		S_t[i](num_landmark, 0) += scale * (y(i, 0) * cosx + y(i, 1) * sinx);
		S_t[i](num_landmark, 1) += scale * (y(i, 1) * cosx - y(i, 0) * sinx);
	}
}
void writeTree(ofstream& f_tree,const randomforest& treenode, const int num_trees,const int max_depth){
	int non_leaves = pow(2, max_depth)-1;
	int leaves = pow(2, max_depth);
	int depth;
	for (int i = 0; i < num_trees; ++i){
		f_tree << treenode[i][0].split_point << "\t" << treenode[i][0].threshold << endl;
		depth = 1;
		for (int j = 1; j < non_leaves; ++j){
			if (j >= pow(2, depth + 1) - 1)
				++depth;
			for (int t = 0; t < depth; ++t)
				f_tree << "\t";
			f_tree << treenode[i][j].split_point << "\t" << treenode[i][j].threshold << endl;
			
		}
		for (int j = non_leaves; j < non_leaves+leaves; ++j)
			f_tree << "\t\t\t\t"<<treenode[i][j].output(0) << "\t" << treenode[i][j].output(1) << endl;
		f_tree << ">>>>>>>>>>" << endl;
	}
}
randomforest readTree(ifstream& f_tree, const int num_trees, const int max_depth){
	int non_leaves = pow(2, max_depth) - 1;
	int leaves = pow(2, max_depth);
	int depth;
	randomforest forest;
	randomtree rf;
	rfs tmp;
	string nothing;
	for (int i = 0; i < num_trees; ++i){
		f_tree >> tmp.split_point >> tmp.threshold;
		rf.push_back(tmp);
		//f_tree << treenode[i][0].split_point << "\t" << treenode[i][0].threshold << endl;
		depth = 1;
		for (int j = 1; j < non_leaves; ++j){
			if (j >= pow(2, depth + 1) - 1)
				++depth;
			//for (int t = 0; t < depth; ++t)
			//	f_tree >> "\t";
			f_tree >> tmp.split_point >> tmp.threshold;
			rf.push_back(tmp);

		}
		for (int j = non_leaves; j < non_leaves + leaves; ++j){
			f_tree >> tmp.output(0) >> tmp.output(1);
			rf.push_back(tmp);
		}
		f_tree >> nothing;
		forest.push_back(rf);
		rf.clear();
	}
	return forest;
}
Eigen::MatrixXd readPoints(ifstream& f_point,const int P){
	Eigen::MatrixXd pts;
	pts.resize(P, 2);
	for (int i = 0; i < P; ++i){
		f_point >> pts(i, 0) >> pts(i, 1);
	}
	return pts;
}
Eigen::MatrixXd GenerateTestFeature(const Eigen::MatrixXd& pts, const vector<Mat_<uchar>>&img,const vector<landmark>&St, const trans_matrix* trans, int num_landmark, int N, int N_aug, int P, int num_feat){
	Eigen::MatrixXd feat;
	int count;
	vector<double> pixels;
	//srand(time(NULL));
	double sinx, cosx;
	//vector<double> ori_x, ori_y;
	double ori_x, ori_y, scale;
	//double scale;
	double t;

	feat.resize(N*N_aug, num_feat);
	//cout << "Here count:" <<count<< endl;
	for (int i = 0; i < N; ++i){
		sinx = trans[i].sint;
		cosx = trans[i].cost;
		scale = 1 / trans[i].scale;
		if (pixels.size())
			cout << "Error, pixels should be zero!" << endl;
		//cout << tanx << " " << cotx << " " << sinx << " " << cosx << " " << scale << endl;
		for (int j = 0; j < P; ++j){
			ori_x = S_t[i](num_landmark, 0) + scale * (pts(j, 0) * cosx + pts(j, 1) * sinx);
			ori_y = S_t[i](num_landmark, 1) + scale * (pts(j, 1) * cosx - pts(j, 0) * sinx);
			//if (ori_x>=250||ori_y>=250)
			//cout << "coordiate" << pts(j, 0) << " " << pts(j, 1) << endl;
			pixels.push_back(interp2(img,i, ori_x, ori_y));
			//ori_pts(j, 0) = ori_x ;
			//ori_pts(j, 1) = ori_y ;
			//feat(i, j) = img[i]()
		}
		count = 0;
		for (int px = 0; px < P - 1; ++px)
		for (int py = px + 1; py < P; ++py){
			/*if (pixels[px] - pixels[py] == 0){
			cout << pixels[px] << " " << pixels[py] << endl;
			cout << "px" << px << "::" <<ori_x[px] << " " << ori_y[px] << "py" << py<< "::"<<ori_x[py] << " " << ori_y[py] << endl;
			getchar();
			}*/
			feat(i, count++) = pixels[px] - pixels[py];
		}
		pixels.clear();
		//ori_x.clear();
		//ori_y.clear();
		//cout << "count:"<<count << endl;
		//cout << pixels.size() << endl;
	}
	return feat;
}

inline Eigen::MatrixXd onlineTest(const Eigen::MatrixXd& pts, const Mat_<uchar>&img, const landmark &St, const trans_matrix& trans, int num_landmark,int P, int num_feat){
	Eigen::MatrixXd feat;
	int count;
	vector<double> pixels;
	//srand(time(NULL));
	double sinx, cosx;
	//vector<double> ori_x, ori_y;
	double ori_x, ori_y, scale;
	//double scale;
	double t;
	feat.resize(1, num_feat);
	//cout << "Here count:" <<count<< endl;
		sinx = trans.sint;
		cosx = trans.cost;
		scale = 1 / trans.scale;
		if (pixels.size())
			cout << "Error, pixels should be zero!" << endl;
		//cout << tanx << " " << cotx << " " << sinx << " " << cosx << " " << scale << endl;
		for (int j = 0; j < P; ++j){
			ori_x = St(num_landmark, 0) + scale * (pts(j, 0) * cosx + pts(j, 1) * sinx);
			ori_y = St(num_landmark, 1) + scale * (pts(j, 1) * cosx - pts(j, 0) * sinx);
			pixels.push_back(interp1(img,ori_x, ori_y));
		}
		count = 0;
		for (int px = 0; px < P - 1; ++px)
		for (int py = px + 1; py < P; ++py){
			feat(0, count++) = pixels[px] - pixels[py];
		}
		pixels.clear();
	return feat;
}

int train_main(int argc, char** argv)
{
	regression_err*Y;
	vector<Mat_<uchar>> train_img;
	vector<landmark> train_shape;
	int num_trees = 10;
	int max_depth = 4;
	int Pt = 100;
	trans_matrix trans[Nimg * N_aug];
	randomforest m[Nfp];
	//regression_err y_b;
	Eigen::MatrixXd points;
	ofstream f_points;
	int num_selected_feature = 500;
	int num_selected_sample = 60;

	string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	
	imageRead(train_img,train_shape,address,Nimg);
	//getchar();

	int num_feat = Pt*(Pt - 1) / 2;

	/*Liblinear definition*/
	/////////////////////////
	struct feature_node **global_binary_features;
	global_binary_features = new struct feature_node*[Nimg*N_aug];

	for (int i = 0; i < Nimg*N_aug; ++i){
		global_binary_features[i] = new feature_node[num_trees*Nfp + 1];
	}


	Y = new regression_err[Nfp];
	
	//for (int i = 0; i < Nfp; ++i){
	//	Y[i] = new regression_err;
	//}

	///////////////////////////////////////////////////////////
	/*Liblinear*/
	for (int i = 0; i < Nimg*N_aug; ++i){
		global_binary_features[i][num_trees*Nfp].index = -1;
		global_binary_features[i][num_trees*Nfp].value = -1.0;
	}


	struct parameter* regression_params = new struct parameter;
	regression_params->solver_type = L2R_L2LOSS_SVR;
	regression_params->C = 1.0 / Nimg*N_aug;
	regression_params->p = 0;
	struct model* model_x[Nfp];
	struct model* model_y[Nfp];
	double** targets = new double*[Nfp];
	for (int i = 0; i < Nfp; ++i){
		targets[i] = new double[Nimg*N_aug];
	}
	cout << "Liblinear initialize done!" << endl;

	f_points.open("D:\\haomaiyi\\Facial_landmark\\model\\meanshape.txt");
	//if (!f_tree.is_open() || !f_points.is_open())
	//	cout << "File not exist!" << endl;
	refershape t = generateMeanshape(train_img,train_shape);
	f_points << "Meanshape" << endl;
	f_points << "Translation:" << "\t" << t.translation(0) << "\t" << t.translation(1) << endl;
	f_points << "Scale:" << "\t" << t.scale << endl;
	for (int i = 0; i < Nfp; ++i)
		f_points << t.shape(i, 0) << "\t" << t.shape(i, 1) << endl;
	S_t = initial_face(Nimg, N_aug, train_img, train_shape);
	//feat.resize(Nimg+Ntest, num_feat);
	//for (int i = 0;)
	//change in every landmark, but save points for each landmark
	for (int iter = 0; iter < 5; ++iter){
		string stage = "stage" + to_string(iter) + "\\";
#pragma omp parallel for
		for (int i = 0; i < Nimg*N_aug; ++i){
			trans[i] = procrustes(t, S_t[i]);
		}

		updateError(Y,Nimg, N_aug, Nfp, train_shape, S_t, trans);

		double err2 = 0;

		cout << "begin training" << endl;
		//omp_set_num_threads(8);
//#pragma omp parallel for
		
		for (int i = 0; i < Nfp; ++i){
			srand(int(time(NULL)) ^ omp_get_thread_num());
			//srand(time(NULL)^i);

			//cout << "Currentid" << int(GetCurrentProcessId()) << endl;
			//srand((int)());
			ofstream f_tree, f_points;
			string filename = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "tree" + to_string(i);
			string pointname = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "point" + to_string(i);
			//cout << filename;
			//cout << "threads num" << omp_get_num_threads() << endl;
			f_tree.open(filename);
			f_points.open(pointname);
			Eigen::MatrixXd feat;
			feat.resize(Nimg*N_aug, num_feat);
			points = GenerateShapeIndexFeature(feat, train_img, S_t, trans, i, Nimg, N_aug, Pt, radius[iter]*t.scale); //corresponding landmark
			for (int j = 0; j < Pt; ++j)
				f_points << points(j, 0) << '\t' << points(j, 1) << endl;
			clock_t t1 = clock();
			m[i] = train_rfs(feat, Y[i], num_trees, max_depth, num_feat, num_selected_sample, num_selected_feature);
			clock_t t2 = clock();
			cout << "Landmark #" << i << "Timing:" << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;
			test_rfs(Nimg*N_aug, m[i], feat, global_binary_features, i, num_trees, max_depth);
			writeTree(f_tree, m[i], num_trees, max_depth);
			f_tree.close();

		}

		//getchar();
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			struct problem* prob = new struct problem;
			prob->bias = -1;
			prob->l = Nimg*N_aug;
			prob->n = num_trees*Nfp * 16;
			prob->x = global_binary_features;
			cout << "Target column " << i << endl;
			for (int j = 0; j < Nimg*N_aug; ++j){
				targets[i][j] = Y[i](j,0);
			}
			prob->y = targets[i];
			check_parameter(prob, regression_params);
			struct model* regression_model = train(prob, regression_params);
			if (!save_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblinearx" + to_string(i)).c_str(), regression_model))
				cout << "err" << endl;
			//return -1;
			model_x[i] = regression_model;
			for (int j = 0; j < Nimg*N_aug; ++j){
				targets[i][j] = Y[i](j,1);
			}
			prob->y = targets[i];
			regression_model = train(prob, regression_params);
			model_y[i] = regression_model;
			if (!save_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblineary" + to_string(i)).c_str(), regression_model))
				cout << "err" << endl;
		}

		double erra = 0;
		for (int i = 0; i < Nfp; i++){
			regression_err prediction;
			for (int j = 0; j < Nimg*N_aug; j++){
				prediction(j, 0) = predict(model_x[i], global_binary_features[j]);
				prediction(j, 1) = predict(model_y[i], global_binary_features[j]);
			}
			updateShape(Nimg*N_aug, S_t, prediction, trans, i);
			erra += Y[i].norm();
			err2 += (Y[i] - prediction).norm();
		}
		f_points << "Epoch:" << iter << "Total Residue" << erra << "Global learning err:" << err2 <<endl;

	}
	for (int i = 0; i < Nfp; ++i){
		delete[] targets[i];
	}
	delete[] targets;
	//for (int i = 0; i < Nfp; ++i){
	//	delete Y[i];
	//}
	delete[]Y;
	f_points.close();
	//f_tree.close();
	///getchar();
	//}
	return 0;
}
int randomtest_main(int argc, char** argv){
	vector<Mat_<uchar>> test_img;
	vector<landmark> test_shape;
	regression_err* Y[Nfp];

	trans_matrix trans[Ntest];
	//string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	string address = "D:\\haomaiyi\\Facial_landmark\\testInternet\\img\\";
	randomforest m[Nfp];
	int num_trees = 10;
	int max_depth = 4;
	int Pt = 100;
	int num_feat = Pt*(Pt - 1) / 2;

	//imageRead(train_img, train_shape, address,Nimg);
	imageRead(test_img, test_shape, address,Ntest);
	ifstream f_shape;
	f_shape.open("D:\\haomaiyi\\Facial_landmark\\model\\meanshape.txt");
	
	refershape t = readMeanshape(f_shape);
	f_shape.close();


	struct feature_node **test_binary_features;
	test_binary_features = new struct feature_node*[Ntest];
	for (int i = 0; i < Ntest; ++i){
		test_binary_features[i] = new feature_node[num_trees*Nfp + 1];
	}
	for (int i = 0; i < Ntest; ++i){
		test_binary_features[i][num_trees*Nfp].index = -1;
		test_binary_features[i][num_trees*Nfp].value = -1.0;
	}
	struct model* model_x[Nfp];
	struct model* model_y[Nfp];
	/*struct parameter* regression_params = new struct parameter;
	regression_params->solver_type = L2R_L2LOSS_SVR;
	regression_params->C = 1.0 / Nimg;
	regression_params->p = 0;
	double** targets = new double*[Nfp];
	for (int i = 0; i < Nfp; ++i){
		targets[i] = new double[Nimg];
	}*/
	cout << "Liblinear initialize done!" << endl;


	cout << "test" << test_img.size() << endl;
	S_t = test_face(Ntest, Nimg, test_img, t);
	//landmark tmp;
	
	for (int iter = 0; iter < 5; ++iter){
		
		string stage = "stage" + to_string(iter) + "\\";


#pragma omp parallel for
		for (int i = 0; i < Ntest; ++i){
			trans[i] = procrustes(t, S_t[i]);
			//cout << trans[i].scale << endl;
		}
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			//cout << "Epoch" << iter << "Fp" << i << endl;
			srand(int(time(NULL)) ^ omp_get_thread_num());
			Eigen::MatrixXd points;
			
			//srand(time(NULL));

			//cout << "Currentid" << int(GetCurrentProcessId()) << endl;
			//srand((int)());
			ifstream f_tree, f_points;
			//ofstream f_check;
			string filename = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "tree" + to_string(i);
			string pointname = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "point" + to_string(i);
			//cout << filename;
			//cout << "threads num" << omp_get_num_threads() << endl;
			f_tree.open(filename);
			f_points.open(pointname);
			points = readPoints(f_points,Pt);
			
			Eigen::MatrixXd feat;
			feat = GenerateTestFeature(points,test_img, S_t, trans, i, Ntest, 1, Pt, num_feat);
			//cout << feat << endl;
			//getchar();
			m[i] = readTree(f_tree, num_trees, max_depth);
			test_rfs(Ntest, m[i], feat, test_binary_features, i, num_trees, max_depth);
			f_points.close();
			f_tree.close();
		}


		/*for (int i = 0; i < Ntest; i+=10){
			for (int j = 0; j < 5; ++j)
				cout << "index" << test_binary_features[i][j].index << "value" << test_binary_features[i][j].value << " ";
			cout << endl;
		}*/
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			//_itoa(i, stringx, 10);
			
			model_x[i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblinearx" + to_string(i)).c_str());
			model_y[i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblineary" + to_string(i)).c_str());
			
		}


		double err3 = 0;
		for (int i = 0; i < Ntest; ++i){
			err3 += (test_shape[i] - S_t[i]).norm();
		}

		for (int i = 0; i < Nfp; i++){
			delta_y prediction;
			for (int j = 0; j < Ntest; j++){
				prediction(j, 0) = predict(model_x[i], test_binary_features[j]);
				prediction(j, 1) = predict(model_y[i], test_binary_features[j]);
			}
			updateShape(Ntest, S_t, prediction, trans, i);
			//cout << "y2 norm" << (*Y[i] - prediction).norm() << endl;
			//cv::Mat_<double> rot;
			//cv::transpose(rotations_[i], rot);
			//predict_regression_targets[i] = scales_[i] * a * rot;
			//if (i % 500 == 0 && i > 0){
			//	std::cout << "predict " << i << " images" << std::endl;
			//}
		}
		
		
		/*double err4 = 0;
		//f_points << "Iter:" << iter << endl;
		for (int i = 0; i < Ntest; ++i){
			//f_points << (shape[i + Nimg] - S_t[i + Nimg]).norm() << endl;

			err4 += (test_shape[i] - S_t[i]).norm();
		}
		
		cout<< "Shape error on testing:" << err3 << "updating" << err4 << endl;*/
	}
	//getchar();
	
	for (int i = 0; i < Ntest; ++i){
		landmark tmp = S_t[i];
		Mat tempimg = full_color_test_img[i].clone();
		for (int k = 0; k < Nfp; k++){
			Point p;

			p.x = tmp(k, 0);
			p.y = tmp(k, 1);
			circle(tempimg, p, 3, Scalar(0, 0, 0));
		}
		imshow("test", tempimg);
		waitKey(0);
	}
	 
	return 0;
}
int videotest_main(int argc, char** argv){
	regression_err* Y;

	trans_matrix trans;
	//vector<Mat_<uchar>> testimg;
	//vector<landmark> testshape;
	//string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	//string address = "D:\\haomaiyi\\Facial_landmark\\testInternet\\img\\";
	randomforest m[5][Nfp];
	Eigen::MatrixXd points[5][Nfp];
	int num_trees = 10;
	int max_depth = 4;
	int Pt = 100;
	int num_feat = Pt*(Pt - 1) / 2;

	//imageRead(train_img, train_shape, address,Nimg);
	//imageRead(test_img, test_shape, address, Ntest);
	ifstream f_shape;
	f_shape.open("D:\\haomaiyi\\Facial_landmark\\model\\meanshape.txt");

	refershape t = readMeanshape(f_shape);
	f_shape.close();


	struct feature_node *test_binary_features = new feature_node[num_trees*Nfp + 1];
	test_binary_features[num_trees*Nfp].index = -1;
	test_binary_features[num_trees*Nfp].value = -1.0;
	struct model* model_x[5][Nfp];
	struct model* model_y[5][Nfp];

	cout << "Liblinear initialize done!" << endl;
	for (int iter = 0; iter < 5; ++iter){
		string stage = "stage" + to_string(iter) + "\\";
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			ifstream f_tree, f_points;
			string filename = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "tree" + to_string(i);
			string pointname = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "point" + to_string(i);
			f_tree.open(filename);
			f_points.open(pointname);
			points[iter][i] = readPoints(f_points, Pt);
			m[iter][i] = readTree(f_tree, num_trees, max_depth);
			model_x[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblinearx" + to_string(i)).c_str());
			model_y[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblineary" + to_string(i)).c_str());
			f_points.close();
			f_tree.close();
		}
	}
	cout << "Randomforest load done!" << endl;
	VideoCapture cap("C:\\Users\\Administrator\\Desktop\\debate.mp4"); // open the default camera
	//VideoCapture cap(0);
	//VideoWriter outputVideo;
	//Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
	//	(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	//int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
	//outputVideo.open("bryan.avi", ex, cap.get(CV_CAP_PROP_FPS), S, true);
	//if (!outputVideo.isOpened())
	//	cout << "error" << endl;


	if (!cap.isOpened())  // check if we succeeded
		return -1;
	cout << "Camera installed!" << endl;
	CascadeClassifier face_cascade;
	String face_cascade_name = "haarcascade_frontalface.xml";
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading face cascade\n");
	}
	Mat_<uchar> face_equal;
	vector<Rect> temp;
	//landmark t_landmark;
	Mat test_img;
	landmark test_shape;
	//imageRead(testimg, testshape, address, Ntest);
	//for (int tt = 0; tt < Ntest;++tt){
	//	test_img = testimg[tt];
	//	test_shape = testshape[tt];
	int count = 0;
	//string address = "\\\\share.local.shiyijian.cc\\share\\face_annotation\\Collection\\FaceDataBaseRename\\";
	//for(;;){
	//	string name = to_string(count) + ".jpg";
	while (cap.isOpened()){
		cap >> test_img;
		//cout << test_img.size() << endl;
		//test_img = imread(address + name, CV_LOAD_IMAGE_COLOR);
		//Size ss = test_img.size();
		//if (ss.width > 1000 || ss.height > 1000){
		//	resize(test_img, test_img, Size(0, 0), 0.3, 0.3);
		//}
		//resize(test_img, test_img, Size(0, 0), 0.5, 0.5);
		cvtColor(test_img, face_equal, CV_RGB2GRAY);
		equalizeHist(face_equal, face_equal);
		face_cascade.detectMultiScale(face_equal, temp, 1.1, 2, 0, Size(40,40));
		//cout << "Face detected!" <<temp.size()<< endl;
		test_shape = t.shape;
		double width = 0.8*test_shape.colwise().mean()(0);
		double height = 0.8*test_shape.colwise().mean()(1);
		//clock_t t1 = clock();
		for (int tt = 0; tt < temp.size(); ++tt){
			//temp[tt].height;
			test_shape.rowwise() -= t.translation.transpose();
			test_shape.col(0) /= width;
			test_shape.col(1) /= height;
			test_shape.col(1) *= temp[tt].height;
			test_shape.col(0) *= temp[tt].width;
			//cout << temp[0].x << temp[0].y << endl;
			test_shape.rowwise() += Eigen::Vector2d(temp[tt].x + temp[tt].width / 2, temp[tt].y + temp[tt].height / 2).transpose();
			//test_shape = t_landmark;
			//landmark tmp;
			//Eigen::MatrixXd feat;
				for (int iter = 0; iter < 5; ++iter){
					trans = procrustes(t, test_shape);
					//cout << trans[i].scale << endl;
#pragma omp parallel for
					for (int i = 0; i < Nfp; ++i){
						Eigen::MatrixXd feat;
						feat = onlineTest(points[iter][i], face_equal, test_shape, trans, i, Pt, num_feat);
						for (int k = 0; k < num_trees; ++k){
							int index = 0;
							for (int d = 0; d < max_depth; ++d){
								int _feat = m[iter][i][k][index].split_point;
								if (_feat < 0)
									index = 2 * index + 1;
								else if (feat(0, _feat) >= m[iter][i][k][index].threshold)
									index = 2 * index + 2;
								else
									index = 2 * index + 1;
							}
							test_binary_features[i*num_trees + k].index = 16 * (i*num_trees + k) + index - 14;
							test_binary_features[i*num_trees + k].value = 1.0;
						}
					}
#pragma omp parallel for
					for (int i = 0; i < Nfp; ++i){
						double delta_x, delta_y;
						delta_x = predict(model_x[iter][i], test_binary_features);
						delta_y = predict(model_y[iter][i], test_binary_features);
						test_shape(i, 0) += (delta_x * trans.cost + delta_y * trans.sint) / trans.scale;
						test_shape(i, 1) += (delta_y * trans.cost - delta_x * trans.sint) / trans.scale;
					}
					/*Mat temgimg = test_img.clone();
					rectangle(temgimg, temp[0], Scalar(255, 255, 0), 3);
					for (int k = 0; k < Nfp; k++){
					Point p;
					p.x = test_shape(k, 0);
					p.y = test_shape(k, 1);
					cout << p << endl;
					circle(temgimg, p, 1, Scalar(0, 255, 0));
					}
					imshow("test", temgimg);
					waitKey(0);*/
				}
			
			rectangle(test_img, temp[tt], Scalar(255, 255, 0), 3);
//#pragma omp parallel for
			for (int k = 0; k < Nfp; k++){
				Point p;
				p.x = test_shape(k, 0);
				p.y = test_shape(k, 1);
				//cout << p << endl;
				circle(test_img, p, 1, Scalar(0, 255, 0));
			}
			test_shape = t.shape;
			//getchar();
		}
		//clock_t t2 = clock();
		//cout << "Timing:" << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;
		//getchar();
		temp.clear();
		imshow("test", test_img);
		//waitKey(0);
		//getchar();
		//++count;
		//waitKey(5);
		if (!waitKey(1)) break;
		//outputVideo << test_img;
	}
	cap.release();
	destroyAllWindows();
	return 0;
}
int test_main(int argc, char** argv){
	regression_err* Y;

	trans_matrix trans;
	//vector<Mat_<uchar>> testimg;
	//vector<landmark> testshape;
	//string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	//string address = "D:\\haomaiyi\\Facial_landmark\\testInternet\\img\\";
	randomforest m[5][Nfp];
	Eigen::MatrixXd points[5][Nfp];
	int num_trees = 10;
	int max_depth = 4;
	int Pt = 100;
	int num_feat = Pt*(Pt - 1) / 2;

	//imageRead(train_img, train_shape, address,Nimg);
	//imageRead(test_img, test_shape, address, Ntest);
	ifstream f_shape;
	f_shape.open("D:\\haomaiyi\\Facial_landmark\\model\\meanshape.txt");
	

	refershape t = readMeanshape(f_shape);
	f_shape.close();


	struct feature_node *test_binary_features = new feature_node[num_trees*Nfp + 1];
	test_binary_features[num_trees*Nfp].index = -1;
	test_binary_features[num_trees*Nfp].value = -1.0;
	struct model* model_x[5][Nfp];
	struct model* model_y[5][Nfp];

	cout << "Liblinear initialize done!" << endl;
	for (int iter = 0; iter < 5; ++iter){
		string stage = "stage" + to_string(iter) + "\\";
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			ifstream f_tree, f_points;
			string filename = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "tree" + to_string(i);
			string pointname = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "point" + to_string(i);
			f_tree.open(filename);
			f_points.open(pointname);
			points[iter][i] = readPoints(f_points, Pt);
			m[iter][i] = readTree(f_tree, num_trees, max_depth);
			model_x[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblinearx" + to_string(i)).c_str());
			model_y[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblineary" + to_string(i)).c_str());
			f_points.close();
			f_tree.close();
		}
	}
	cout << "Randomforest load done!" << endl;
	CascadeClassifier face_cascade;
	String face_cascade_name = "haarcascade_frontalface.xml";
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading face cascade\n");
	}

	//imageRead(testimg, testshape, address, Ntest);
	//for (int tt = 0; tt < Ntest;++tt){
	//	test_img = testimg[tt];
	//	test_shape = testshape[tt];
	int count = 0;
	string address = "\\\\share.local.shiyijian.cc\\share\\face_annotation\\Collection\\FaceDataBaseRename\\";
	//for(;;){
	//	string name = to_string(count) + ".jpg";
	string fileaddress = "\\\\share.local.shiyijian.cc\\share\\face_annotation\\Collection\\combine\\";
	string name;
	string nothing;
	ifstream fin_img, fin_pts, fin_points;
	ofstream f_err;
	f_err.open("err_out.txt");
	fin_img.open("73p_img.txt");
	fin_pts.open("73p_file.txt");
	double mean = 0;
	//vector<Mat_<uchar>> img;
	//vector<landmark> groundtruth;
	landmark temp;
	double x, y;
	Mat_<uchar> face_equal;
	vector<Rect> detected_face;
	//landmark t_landmark;
	Mat test_img;
	landmark test_shape;

	for (int i = 0; i < Ntest; ++i){
		double sum_err = 0;
		fin_img >> name;
		test_img = imread(address + name, CV_LOAD_IMAGE_GRAYSCALE);
		cout << "image"<<i << endl;
		fin_pts >> name;

		//imshow("test", img[i]);
		//getchar();
		//cout << img[i].size() << endl;

		//cout << name << endl;
		//cout << address + name << endl;
		fin_points.open(fileaddress + name);
		fin_points >> nothing >> nothing >> nothing >> nothing >> nothing;
		if (!fin_points.is_open())
			cout << "No points file" << endl;
		//cout << "hehe"<<nothing << endl;
		for (int j = 0; j < temp.rows(); ++j){
			if (j == 64)
				continue;
			fin_points >> x >> y;
			temp(j, 0) = x;
			temp(j, 1) = y;
		}

		/*for (int j = 0; j < temp.rows(); ++j){
			if (j == 64)
				continue;
			cout << temp(j, 0) << " " << temp(j, 1) << endl;
		}*/

		//groundtruth.push_back(temp);
		fin_points.close();
		//Size ss = test_img.size();
		//if (ss.width > 1000 || ss.height > 1000){
		//	resize(test_img, test_img, Size(0, 0), 0.3, 0.3);
		//}
		//resize(test_img, test_img, Size(0, 0), 0.5, 0.5);
		//cvtColor(test_img, face_equal, CV_RGB2GRAY);
		equalizeHist(test_img, face_equal);
		face_cascade.detectMultiScale(face_equal, detected_face, 1.1, 2, 0, Size(30, 30));
		//cout << "Face detected!" <<temp.size()<< endl;
		test_shape = t.shape;
		int max_i = 0;
		if (detected_face.size()){
			if (detected_face.size()>1){
				double max_size = 0;
				for (int s = 0; s < detected_face.size(); ++s)
				if (detected_face[s].width>max_size){
					max_size = detected_face[s].width;
					max_i = s;
				}
			}
			double width = 0.7*test_shape.colwise().mean()(0);
			double height = 0.8*test_shape.colwise().mean()(1);
			//temp[tt].height;
			test_shape.rowwise() -= t.translation.transpose();
			test_shape.col(0) /= width;
			test_shape.col(1) /= height;
			test_shape.col(1) *= detected_face[max_i].height;
			test_shape.col(0) *= detected_face[max_i].width;
			//cout << temp[0].x << temp[0].y << endl;
			test_shape.rowwise() += Eigen::Vector2d(detected_face[max_i].x + detected_face[max_i].width / 2, detected_face[max_i].y + detected_face[max_i].height / 2).transpose();
		}
		//test_shape = t_landmark;
		//landmark tmp;
		//imshow("test", face_equal);
		//waitKey(0);
		for (int iter = 0; iter < 5; ++iter){
			trans = procrustes(t, test_shape);
			//cout << trans[i].scale << endl;
#pragma omp parallel for
			for (int i = 0; i < Nfp; ++i){
				Eigen::MatrixXd feat;
				feat = onlineTest(points[iter][i], face_equal, test_shape, trans, i, Pt, num_feat);
				for (int k = 0; k < num_trees; ++k){
					int index = 0;
					for (int d = 0; d < max_depth; ++d){
						int _feat = m[iter][i][k][index].split_point;
						if (_feat < 0)
							index = 2 * index + 1;
						else if (feat(0, _feat) >= m[iter][i][k][index].threshold)
							index = 2 * index + 2;
						else
							index = 2 * index + 1;
					}
					test_binary_features[i*num_trees + k].index = 16 * (i*num_trees + k) + index - 14;
					test_binary_features[i*num_trees + k].value = 1.0;
				}
			}
#pragma omp parallel for
			for (int i = 0; i < Nfp; ++i){
				double delta_x, delta_y;
				delta_x = predict(model_x[iter][i], test_binary_features);
				delta_y = predict(model_y[iter][i], test_binary_features);
				test_shape(i, 0) += (delta_x * trans.cost + delta_y * trans.sint) / trans.scale;
				test_shape(i, 1) += (delta_y * trans.cost - delta_x * trans.sint) / trans.scale;
			}
			/*Mat temgimg = test_img.clone();
			rectangle(temgimg, temp[0], Scalar(255, 255, 0), 3);
			for (int k = 0; k < Nfp; k++){
			Point p;
			p.x = test_shape(k, 0);
			p.y = test_shape(k, 1);
			cout << p << endl;
			circle(temgimg, p, 1, Scalar(0, 255, 0));
			}
			imshow("test", temgimg);
			waitKey(0);*/
		}

		//if (detected_face.size())
		//	cv::rectangle(test_img, detected_face[max_i], Scalar(255, 255, 0), 3);

		//#pragma omp parallel for
		for (int k = 0; k < Nfp; k++){
			if (k == 64)
				continue;
			sum_err += (temp.row(k) - test_shape.row(k)).norm();
			f_err << (temp.row(k) - test_shape.row(k)).norm()<<" ";
			//Point p;
			//p.x = test_shape(k, 0);
			//p.y = test_shape(k, 1);
			//cout << p << endl;
			//circle(test_img, p, 1, Scalar(0, 255, 0));
		}
		//f_err << endl;
		cout << "image" << i << "error" << sum_err / (temp.row(27) - temp.row(31)).norm() << endl;
		f_err << (temp.row(27) - temp.row(31)).norm() << endl;
		mean += sum_err / (temp.row(27) - temp.row(31)).norm();
		f_err.flush();
		//getchar();
		detected_face.clear();
		//imshow("test", test_img);
		//waitKey(0);

	}
	f_err << "Mean Error" << mean/Ntest << endl;
	f_err.flush();
	f_err.close();
	destroyAllWindows();
	return 0;
}
int test2_main(int argc, char** argv){
	regression_err* Y;

	trans_matrix trans;
	//vector<Mat_<uchar>> testimg;
	//vector<landmark> testshape;
	//string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	//string address = "D:\\haomaiyi\\Facial_landmark\\testInternet\\img\\";
	randomforest m[5][Nfp];
	Eigen::MatrixXd points[5][Nfp];
	int num_trees = 10;
	int max_depth = 4;
	int Pt = 100;
	int num_feat = Pt*(Pt - 1) / 2;

	//imageRead(train_img, train_shape, address,Nimg);
	//imageRead(test_img, test_shape, address, Ntest);
	ifstream f_shape;
	f_shape.open("D:\\haomaiyi\\Facial_landmark\\1000model\\meanshape.txt");


	refershape t = readMeanshape(f_shape);
	f_shape.close();


	struct feature_node *test_binary_features = new feature_node[num_trees*Nfp + 1];
	test_binary_features[num_trees*Nfp].index = -1;
	test_binary_features[num_trees*Nfp].value = -1.0;
	struct model* model_x[5][Nfp];
	struct model* model_y[5][Nfp];

	cout << "Liblinear initialize done!" << endl;
	for (int iter = 0; iter < 5; ++iter){
		string stage = "stage" + to_string(iter) + "\\";
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			ifstream f_tree, f_points;
			string filename = "D:\\haomaiyi\\Facial_landmark\\7000model\\" + stage + "tree" + to_string(i);
			string pointname = "D:\\haomaiyi\\Facial_landmark\\7000model\\" + stage + "point" + to_string(i);
			f_tree.open(filename);
			f_points.open(pointname);
			points[iter][i] = readPoints(f_points, Pt);
			m[iter][i] = readTree(f_tree, num_trees, max_depth);
			model_x[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\1000model\\" + stage + "liblinearx" + to_string(i)).c_str());
			model_y[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\1000model\\" + stage + "liblineary" + to_string(i)).c_str());
			f_points.close();
			f_tree.close();
		}
	}
	cout << "Randomforest load done!" << endl;
	CascadeClassifier face_cascade;
	String face_cascade_name = "haarcascade_frontalface.xml";
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading face cascade\n");
	}

	//imageRead(testimg, testshape, address, Ntest);
	//for (int tt = 0; tt < Ntest;++tt){
	//	test_img = testimg[tt];
	//	test_shape = testshape[tt];
	int count = 0;
	string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	//for(;;){
	//	string name = to_string(count) + ".jpg";
	string name;
	string nothing;
	ifstream fin_img, fin_pts, fin_points;
	ofstream f_err;
	f_err.open("err_out.txt");
	fin_img.open("test_img.txt");
	fin_pts.open("test_file.txt");
	double mean = 0;
	//vector<Mat_<uchar>> img;
	//vector<landmark> groundtruth;
	landmark temp;
	double x, y;
	Mat_<uchar> face_equal;
	vector<Rect> detected_face;
	//landmark t_landmark;
	Mat test_img;
	landmark test_shape;
	for (int i = 0; i < Ntest; ++i){
		double sum_err = 0;
		fin_img >> name;
		test_img = imread(address + name, CV_LOAD_IMAGE_GRAYSCALE);
		cout << "image" << i << endl;
		fin_pts >> name;

		//imshow("test", img[i]);
		//getchar();
		//cout << img[i].size() << endl;

		//cout << name << endl;
		//cout << address + name << endl;
		fin_points.open(address + name);
		fin_points >> nothing >> nothing >> nothing >> nothing >> nothing;
		if (!fin_points.is_open())
			cout << "No points file" << endl;
		//cout << "hehe"<<nothing << endl;
		for (int j = 0; j < temp.rows(); ++j){
			fin_points >> x >> y;
			temp(j, 0) = x;
			temp(j, 1) = y;
		}

		/*for (int j = 0; j < temp.rows(); ++j){
		if (j == 64)
		continue;
		cout << temp(j, 0) << " " << temp(j, 1) << endl;
		}*/

		//groundtruth.push_back(temp);
		fin_points.close();
		//Size ss = test_img.size();
		//if (ss.width > 1000 || ss.height > 1000){
		//	resize(test_img, test_img, Size(0, 0), 0.3, 0.3);
		//}
		//resize(test_img, test_img, Size(0, 0), 0.5, 0.5);
		//cvtColor(test_img, face_equal, CV_RGB2GRAY);
		equalizeHist(test_img, face_equal);
		face_cascade.detectMultiScale(face_equal, detected_face, 1.1, 2, 0, Size(30, 30));
		if (!detected_face.size())
			continue;
		//cout << "Face detected!" <<temp.size()<< endl;
		test_shape = t.shape;
		int max_i = 0;
		if (detected_face.size()){
			if (detected_face.size()>1){
				double max_size = 0;
				for (int s = 0; s < detected_face.size(); ++s)
				if (detected_face[s].width>max_size){
					max_size = detected_face[s].width;
					max_i = s;
				}
			}
			double width = 0.7*test_shape.colwise().mean()(0);
			double height = 0.8*test_shape.colwise().mean()(1);
			//temp[tt].height;
			test_shape.rowwise() -= t.translation.transpose();
			test_shape.col(0) /= width;
			test_shape.col(1) /= height;
			test_shape.col(1) *= detected_face[max_i].height;
			test_shape.col(0) *= detected_face[max_i].width;
			//cout << temp[0].x << temp[0].y << endl;
			test_shape.rowwise() += Eigen::Vector2d(detected_face[max_i].x + detected_face[max_i].width / 2, detected_face[max_i].y + detected_face[max_i].height / 2).transpose();
		}
		//test_shape = t_landmark;
		//landmark tmp;
		//imshow("test", face_equal);
		//waitKey(0);
		for (int iter = 0; iter < 5; ++iter){
			trans = procrustes(t, test_shape);
			//cout << trans[i].scale << endl;
#pragma omp parallel for
			for (int i = 0; i < Nfp; ++i){
				Eigen::MatrixXd feat;
				feat = onlineTest(points[iter][i], face_equal, test_shape, trans, i, Pt, num_feat);
				for (int k = 0; k < num_trees; ++k){
					int index = 0;
					for (int d = 0; d < max_depth; ++d){
						int _feat = m[iter][i][k][index].split_point;
						if (_feat < 0)
							index = 2 * index + 1;
						else if (feat(0, _feat) >= m[iter][i][k][index].threshold)
							index = 2 * index + 2;
						else
							index = 2 * index + 1;
					}
					test_binary_features[i*num_trees + k].index = 16 * (i*num_trees + k) + index - 14;
					test_binary_features[i*num_trees + k].value = 1.0;
				}
			}
#pragma omp parallel for
			for (int i = 0; i < Nfp; ++i){
				double delta_x, delta_y;
				delta_x = predict(model_x[iter][i], test_binary_features);
				delta_y = predict(model_y[iter][i], test_binary_features);
				test_shape(i, 0) += (delta_x * trans.cost + delta_y * trans.sint) / trans.scale;
				test_shape(i, 1) += (delta_y * trans.cost - delta_x * trans.sint) / trans.scale;
			}
			/*Mat temgimg = test_img.clone();
			rectangle(temgimg, temp[0], Scalar(255, 255, 0), 3);
			for (int k = 0; k < Nfp; k++){
			Point p;
			p.x = test_shape(k, 0);
			p.y = test_shape(k, 1);
			cout << p << endl;
			circle(temgimg, p, 1, Scalar(0, 255, 0));
			}
			imshow("test", temgimg);
			waitKey(0);*/
		}

		//if (detected_face.size())
		//	cv::rectangle(test_img, detected_face[max_i], Scalar(255, 255, 0), 3);

		//#pragma omp parallel for
		for (int k = 0; k < Nfp; k++){
			sum_err += (temp.row(k) - test_shape.row(k)).norm();
			f_err << (temp.row(k) - test_shape.row(k)).norm() << " ";
			//Point p;
			//p.x = test_shape(k, 0);
			//p.y = test_shape(k, 1);
			//cout << p << endl;
			//circle(test_img, p, 1, Scalar(0, 255, 0));
		}
		//f_err << endl;
		cout << "image" << i << "error" << sum_err / (temp.row(27) - temp.row(31)).norm() << endl;
		f_err << (temp.row(27) - temp.row(31)).norm() << endl;
		f_err.flush();
		mean += sum_err / (temp.row(27) - temp.row(31)).norm();
		//getchar();
		detected_face.clear();
		//imshow("test", test_img);
		//waitKey(0);

	}
	f_err << "Mean Error" << mean / Ntest << endl;
	f_err.flush();
	f_err.close();
	destroyAllWindows();
	return 0;
}
int main(int argc, char** argv){
	regression_err* Y;

	trans_matrix trans;
	//vector<Mat_<uchar>> testimg;
	//vector<landmark> testshape;
	//string address = "D:\\haomaiyi\\Facial_landmark\\ZhouKunDataBase FH GroundTruth revised1\\";
	//string address = "D:\\haomaiyi\\Facial_landmark\\testInternet\\img\\";
	randomforest m[5][Nfp];
	Eigen::MatrixXd points[5][Nfp];
	int num_trees = 10;
	int max_depth = 4;
	int Pt = 100;
	int num_feat = Pt*(Pt - 1) / 2;

	//imageRead(train_img, train_shape, address,Nimg);
	//imageRead(test_img, test_shape, address, Ntest);
	ifstream f_shape;
	f_shape.open("D:\\haomaiyi\\Facial_landmark\\model\\meanshape.txt");


	refershape t = readMeanshape(f_shape);
	f_shape.close();


	struct feature_node *test_binary_features = new feature_node[num_trees*Nfp + 1];
	test_binary_features[num_trees*Nfp].index = -1;
	test_binary_features[num_trees*Nfp].value = -1.0;
	struct model* model_x[5][Nfp];
	struct model* model_y[5][Nfp];

	cout << "Liblinear initialize done!" << endl;
	for (int iter = 0; iter < 5; ++iter){
		string stage = "stage" + to_string(iter) + "\\";
#pragma omp parallel for
		for (int i = 0; i < Nfp; ++i){
			ifstream f_tree, f_points;
			string filename = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "tree" + to_string(i);
			string pointname = "D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "point" + to_string(i);
			f_tree.open(filename);
			f_points.open(pointname);
			points[iter][i] = readPoints(f_points, Pt);
			m[iter][i] = readTree(f_tree, num_trees, max_depth);
			model_x[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblinearx" + to_string(i)).c_str());
			model_y[iter][i] = load_model(("D:\\haomaiyi\\Facial_landmark\\model\\" + stage + "liblineary" + to_string(i)).c_str());
			f_points.close();
			f_tree.close();
		}
	}
	cout << "Randomforest load done!" << endl;
	CascadeClassifier face_cascade;
	String face_cascade_name = "haarcascade_frontalface.xml";
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading face cascade\n");
	}

	//imageRead(testimg, testshape, address, Ntest);
	//for (int tt = 0; tt < Ntest;++tt){
	//	test_img = testimg[tt];
	//	test_shape = testshape[tt];
	int count = 0;
	string address = "\\\\share.local.shiyijian.cc\\share\\face_annotation\\Collection\\FaceDataBaseRename\\";
	//for(;;){
	//	string name = to_string(count) + ".jpg";
	string name;
	string nothing;
	ifstream fin_img;
	fin_img.open("73p_img.txt");
	double mean = 0;
	//vector<Mat_<uchar>> img;
	//vector<landmark> groundtruth;
	landmark temp;
	double x, y;
	Mat_<uchar> face_equal;
	vector<Rect> detected_face;
	//landmark t_landmark;
	Mat test_img;
	landmark test_shape;

	for (int i = 0; i < Ntest; ++i){
		double sum_err = 0;
		//fin_img >> name;
		name = to_string(i) + ".jpg";
		test_img = imread(address + name, CV_LOAD_IMAGE_COLOR);
		cout << "image" << i << endl;
		/*for (int j = 0; j < temp.rows(); ++j){
		if (j == 64)
		continue;
		cout << temp(j, 0) << " " << temp(j, 1) << endl;
		}*/

		//groundtruth.push_back(temp);
		Size ss = test_img.size();
		if (ss.width > 1000 || ss.height > 1000){
			resize(test_img, test_img, Size(0, 0), 0.5, 0.5);
		}
		//resize(test_img, test_img, Size(0, 0), 0.5, 0.5);
		cvtColor(test_img, face_equal, CV_RGB2GRAY);
		equalizeHist(face_equal, face_equal);
		face_cascade.detectMultiScale(face_equal, detected_face, 1.1, 2, 0, Size(30, 30));
		//cout << "Face detected!" <<temp.size()<< endl;
		test_shape = t.shape;
		int max_i = 0;
		if (detected_face.size()){
			if (detected_face.size()>1){
				double max_size = 0;
				for (int s = 0; s < detected_face.size(); ++s)
				if (detected_face[s].width>max_size){
					max_size = detected_face[s].width;
					max_i = s;
				}
			}
			double width = 0.7*test_shape.colwise().mean()(0);
			double height = 0.8*test_shape.colwise().mean()(1);
			//temp[tt].height;
			test_shape.rowwise() -= t.translation.transpose();
			test_shape.col(0) /= width;
			test_shape.col(1) /= height;
			test_shape.col(1) *= detected_face[max_i].height;
			test_shape.col(0) *= detected_face[max_i].width;
			//cout << temp[0].x << temp[0].y << endl;
			test_shape.rowwise() += Eigen::Vector2d(detected_face[max_i].x + detected_face[max_i].width / 2, detected_face[max_i].y + detected_face[max_i].height / 2).transpose();
		}
		//test_shape = t_landmark;
		//landmark tmp;
		//imshow("test", face_equal);
		//waitKey(0);
		for (int iter = 0; iter < 5; ++iter){
			trans = procrustes(t, test_shape);
			//cout << trans[i].scale << endl;
#pragma omp parallel for
			for (int i = 0; i < Nfp; ++i){
				Eigen::MatrixXd feat;
				feat = onlineTest(points[iter][i], face_equal, test_shape, trans, i, Pt, num_feat);
				for (int k = 0; k < num_trees; ++k){
					int index = 0;
					for (int d = 0; d < max_depth; ++d){
						int _feat = m[iter][i][k][index].split_point;
						if (_feat < 0)
							index = 2 * index + 1;
						else if (feat(0, _feat) >= m[iter][i][k][index].threshold)
							index = 2 * index + 2;
						else
							index = 2 * index + 1;
					}
					test_binary_features[i*num_trees + k].index = 16 * (i*num_trees + k) + index - 14;
					test_binary_features[i*num_trees + k].value = 1.0;
				}
			}
#pragma omp parallel for
			for (int i = 0; i < Nfp; ++i){
				double delta_x, delta_y;
				delta_x = predict(model_x[iter][i], test_binary_features);
				delta_y = predict(model_y[iter][i], test_binary_features);
				test_shape(i, 0) += (delta_x * trans.cost + delta_y * trans.sint) / trans.scale;
				test_shape(i, 1) += (delta_y * trans.cost - delta_x * trans.sint) / trans.scale;
			}
			/*Mat temgimg = test_img.clone();
			rectangle(temgimg, temp[0], Scalar(255, 255, 0), 3);
			for (int k = 0; k < Nfp; k++){
			Point p;
			p.x = test_shape(k, 0);
			p.y = test_shape(k, 1);
			cout << p << endl;
			circle(temgimg, p, 1, Scalar(0, 255, 0));
			}
			imshow("test", temgimg);
			waitKey(0);*/
		}

		//if (detected_face.size())
		//	cv::rectangle(test_img, detected_face[max_i], Scalar(255, 255, 0), 3);

		//#pragma omp parallel for
		for (int k = 0; k < Nfp; k++){
			Point p;
			p.x = test_shape(k, 0);
			p.y = test_shape(k, 1);
			//cout << p << endl;
			circle(test_img, p, 2, Scalar(0, 255, 0),-1);
		}
		//f_err << endl;
		//getchar();
		detected_face.clear();
		imshow("test", test_img);
		waitKey(0);

	}
	destroyAllWindows();
	return 0;
}
