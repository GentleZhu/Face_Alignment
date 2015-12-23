#include "utils.h"

using namespace std;
using namespace cv;
int zero_count = 0;

//std::random_device rd;
inline double calculateVariance(const Eigen::MatrixXd& feat, const regression_err& Y, const vector<int>& sample, double threshold, int num_feat, int ind){
	deque<double> temp;
	int cnt = 0;
	Eigen::Vector2d mean_l(0, 0), mean_r(0,0);
	double var1 = 0, var2 = 0, tt = 0;
	//cout << "num_feat"<<num_feat << endl;
	for (int i = 0; i < sample.size(); ++i){
		if (feat(sample[i], num_feat) < threshold){ //leaf node
			//cout << "left" << endl;
			temp.push_front(sample[i]);
			mean_l += Y.row(sample[i]);
			++cnt;
			
		}
		else{
			//cout << "right" << endl;
			temp.push_back(sample[i]);
			mean_r += Y.row(sample[i]);
		}
	}
	//if (cnt != ind){
	//	cout << "Sample size"<<sample.size() <<":threshold"<<threshold<< "cnt" << cnt << "ind" << ind << endl;
		/*for (int i = 0; i < sample.size(); ++i)
			cout << sample[i] << "::";
		cout << endl;*/
	//}

	mean_l /= cnt;
	for (int i = 0; i < ind; ++i){
		var1 += Y.row(temp[i]).squaredNorm();
	}
	var1 -= cnt*mean_l.squaredNorm();
	//var1 /= cnt;
	//if (num_feat>0){
		mean_r /= temp.size() - cnt;
		for (int i = ind; i < temp.size(); ++i){
			var2 += Y.row(temp[i]).squaredNorm();
		}
		var2 -= mean_r.squaredNorm()*(temp.size() - cnt);
		return var1 + var2;
	//}
	//else
	//	return var1;
}
void permulation(int num_feat, int k, vector<int>&selected_feature){
	int index;
	//short * bin = new short[num_feat];
	selected_feature.clear();
	//cout <<k<<" "<< num_feat << endl;
	/*memset(bin, 0, sizeof(short)*num_feat);
	for (int i = 0; i < k; ++i){
		//cout << "here" << endl;
		while (bin[index = rand() % num_feat]);
			bin[index] = 1;
		selected_feature.push_back(index);
	}
	delete[]bin;*/
	short* a=new short [num_feat];
	for (int i = 0; i < num_feat; ++i) a[i] = i;
	for (int i = num_feat; i > num_feat - k; --i) {
		index = rand() % i;
		//t = a[index];
		a[index] = a[i];
		selected_feature.push_back(index);
		//a[i] = t;
	}
	delete[]a;
}
inline void boostrap_sample(int num_feat,vector<int>&selected_feature){
	selected_feature.clear();
	for (int i = 0; i < int(0.8*num_feat); ++i)
		selected_feature.push_back(rand() % num_feat);
}
rfs split_node(const Eigen::MatrixXd& feat, const regression_err& Y, const vector<int>&sample, const vector<int>& selected_feature, int height){
	rfs new_node;
	int num_sample, ind;
	double candidate[2], can_th[2];
	double varLR;
	num_sample = sample.size();
	int max_i = -1;
	double min_loss = 0;
	double max_threshold = 0, threshold;
	//cout << "split_node" << endl;
	if (num_sample>observation_bin&&height>0){
		for (vector<int>::const_iterator fiter = selected_feature.begin(); fiter != selected_feature.end(); fiter++){
			ind = rand() % num_sample;
			threshold = feat(sample[ind], *fiter);
			varLR = calculateVariance(feat, Y, sample, threshold, *fiter, ind);
			if (varLR < min_loss||min_loss==0){
				min_loss = varLR;
				max_threshold = threshold;
				max_i = *fiter;
				//cout << "max_i" << max_i;
			}
		}
		for (vector<int>::const_iterator siter = sample.begin(); siter != sample.end(); siter++){
			if (feat(*siter, max_i) < max_threshold)
				new_node.left_child.push_back(*siter);
			else
				new_node.right_child.push_back(*siter);
		}
		new_node.threshold = max_threshold;
		new_node.split_point = max_i;
	
	}
	else if (height>0){
		new_node.split_point = -1;
		Eigen::Vector2d output(0, 0);
		for (int i = 0; i < num_sample; ++i)
			output += Y.row(sample[i]);
		new_node.output = output / num_sample;
	}
	else if (num_sample){
		Eigen::Vector2d output(0, 0);
		for (int i = 0; i < num_sample; ++i)
			output += Y.row(sample[i]);
		new_node.output = output /num_sample;
		new_node.split_point = num_sample;
	}
	return new_node;
}
randomforest train_rfs(const Eigen::MatrixXd& feat, const regression_err&Y, const int num_trees, const int max_depth, const int num_feat, const int num_selected_sample, const int num_selected_feature)
{
	
	randomforest r;
	randomtree rf;
	rfs tmp;
	int parent;
	vector<int> selected_feature;
	vector<int> selected_sample;
	for (int i = 0; i < num_trees; ++i){
		int depth = 0;
		boostrap_sample(Nimg*N_aug, selected_sample);
		permulation(num_feat, num_selected_feature, selected_feature);
		tmp=split_node(feat, Y, selected_sample, selected_feature, max_depth - depth);
		rf.push_back(tmp); 
		for (int j = 1; j < pow(2, max_depth + 1) - 1; ++j){
			if (j >= pow(2, depth + 1) - 1)
				++depth;

			permulation(num_feat, num_selected_feature, selected_feature);
			parent = floor((double)(j + 1) / 2) - 1;
			if (!rf[parent].split_point){
				rfs node;
				node.split_point = -1;
				rf.push_back(node);
				continue;
			}
			//cout << "parent" << parent << endl;
			if (j == parent * 2 + 1)
				rf.push_back(split_node(feat, Y, rf[parent].left_child, selected_feature, max_depth - depth));
			else
				rf.push_back(split_node(feat, Y, rf[parent].right_child, selected_feature, max_depth - depth));
		}
		r.push_back(rf);
		rf.clear();
	}
	return r;
}
void test_rfs(const int Nsample, const randomforest & r, const Eigen::MatrixXd& feat, struct feature_node **global_binary_features, const int num_landmark, const int num_trees, const int max_depth, const int offset){
	//Eigen::MatrixXd y_temp;
	int num_feat;
	//y_temp.resize(num_trees, 2);
	int index;
	//cout << "Node number:"<<r.size() << endl;
	for (int i = 0; i < Nsample; ++i){
		for (int k = 0; k < num_trees; ++k){
			index = 0;
			for (int d = 0; d < max_depth; ++d){
				num_feat = r[k][index].split_point;
				if (num_feat < 0)
					index = 2 * index + 1;
				else if (feat(i + offset, num_feat) >= r[k][index].threshold)
					index = 2 * index + 2;
				else
					index = 2 * index + 1;
			}

			global_binary_features[i][num_landmark*num_trees + k].index = 16 * (num_landmark*num_trees + k) + index - 14;
			global_binary_features[i][num_landmark*num_trees + k].value = 1.0;
			//cout << "index"<<index << endl;
			//y_temp.row(k) = r[k][index].output;
		}
		//y_b.row(i) = y_temp.colwise().mean();
	}
	//return y_temp;
}
/*
int main(){
	rfs a;
	a.split_point = 3;
	a.threshold = 1.11;
	cout << "here" << a.split_point<<" "<<a.threshold<<endl;
	getchar();
	return 0;
}
*/