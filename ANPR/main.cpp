#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <ml.hpp>

using namespace std;
using namespace cv;
using namespace ml;

bool verify(RotatedRect rect) {
	float error = 0.4;
	const float aspect = 4.7272;
	int min = 15 * aspect * 15; // 面积下限
	int max = 125 * aspect * 125; // 面积上限

	float rmin = aspect - aspect * error; // 宽高比下限
	float rmax = aspect + aspect * error; // 宽高比上限

	int area = rect.size.width * rect.size.height; // 计算面积
	float r = rect.size.width / rect.size.height;  // 计算宽高比
	r = r < 1 ? 1 / r : r;

	return area >= min && area <= max && r >= rmin && r <= rmax;
}


/* 基于SVM的图像分类 */
bool classification(Mat image_crop) {
	
	// 设置训练数据
	FileStorage fs;
	fs.open("SVM.xml", FileStorage::READ);

	Mat trainingDataMat;
	Mat classesMat;
	fs["TrainingData"] >> trainingDataMat;
	fs["classes"] >> classesMat;

	Ptr<TrainData> trainingData = TrainData::create(trainingDataMat, ROW_SAMPLE, classesMat);

    // 创建分类器，并设置参数
	SVM::ParamTypes params;
	SVM::KernelTypes kernel_type = SVM::LINEAR;
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(kernel_type);

	// 训练分类器
	svm->trainAuto(trainingData);
	
	// 预测
	Mat src;
	src.create(33, 144, CV_32FC1);
	resize(image_crop, src, src.size(), 0, 0, INTER_CUBIC);


	imshow("zvhb", src);




	src.convertTo(src, CV_32FC1);
	src = src.reshape(1, 1);
	/*cout << src.rows << " " << src.cols << endl;
	cout << src.channels() << endl;
	cout << src.type() << endl;
	cout << CV_32F << endl;*/

	int response = svm->predict(src);
	// cout << "response = " << response << endl;
	
	return response;
}



int main()
{
	string in = "images/2715DTZ.jpg";
	// string in = "images/3028BYS.JPG"; 

	Mat image = imread(in, IMREAD_GRAYSCALE);
	Mat image2 = imread(in);
	Mat image3 = imread(in, IMREAD_GRAYSCALE);

	if (image.empty()){
		return -1;
	}
	//imshow("【原始图】", image);

	blur(image, image, Size(5, 5));
	//imshow("【去噪后】", image);


	Sobel(image, image, CV_8U, 1, 0, 3, 1, 0);
	//imshow("【sobel滤波】", image);

	threshold(image, image, 0, 255, CV_THRESH_OTSU);
	//imshow("【otsu阈值化】", image);

	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(image, image, CV_MOP_CLOSE, element);
	//imshow("【闭运算】", image);

	vector<vector<Point>> contours;
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	map<int, RotatedRect> _map;

	for (int i = 0; i < contours.size(); i++) {
		drawContours(image, contours, i, Scalar(255), 1); // 绘制轮廓

		// 绘制矩形
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f vertices[4];
		rect.points(vertices);
		for (int i = 0; i < 4; i++) {
			line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255), 2);
		}

		// 验证
		if (verify(rect)) {
			_map[i] = rect;
		}
	}
	imshow("【轮廓提取】", image);



	// 绘制通过验证的矩形
	int min_diff = 100000;
	int index = 0;
	const float square = 27.75;

	map<int, RotatedRect>::iterator iter;
	iter = _map.begin();
	while (iter != _map.end()) {

		RotatedRect rect = iter->second;
		Point2f vertices[4];
		rect.points(vertices);
		for (int j = 0; j < 4; j++) {
		    line(image, vertices[j], vertices[(j + 1) % 4], Scalar(255), 10);
		}

		// 选择最接近的矩形
		int perimeter = arcLength(contours[iter->first], true);
		int area = contourArea(contours[iter->first]);
		if (area != 0) {
		int squareness = perimeter * perimeter / area;
		
		float diff = abs(squareness - square);
		    if (diff < min_diff) {
		        min_diff = diff;
		        index = iter->first;
		    }
		}
		iter++;
	}

	// imshow("【通过验证】", image);

	

	// 绘制最接近的矩形
	RotatedRect rect = _map[index];
	Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; i++) {
		line(image2, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 10);
	}
	// imshow("【最接近的矩形】", image2);


	// 图像切割
	Mat image_crop;
	Size rect_size = rect.size;
	if (rect_size.width < rect_size.height) {
		swap(rect_size.width, rect_size.height);
	}
	//rect_size.width += 10;
	//rect_size.height += 10;
	getRectSubPix(image3, rect_size, rect.center, image_crop);
	imshow("【切割后的车牌】", image_crop);

	// Mat src = imread("2715DTZ.JPG", 0);
	// imshow("src", src);

	// bool flag = classification(src);
	// bool flag = classification(image_crop);
	// cout << "flag = " << flag << endl;

	waitKey();
	return 0;
}