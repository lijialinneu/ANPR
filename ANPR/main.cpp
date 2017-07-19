#include <iostream>
#include <map>
#include <set>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <ml.hpp>
#include <string.h>

using namespace std;
using namespace cv;
using namespace ml;

const int HORIZONTAL = 1;
const int VERTICAL = 0;
const char strCharacters[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z' };
const int numCharacters = 30;
const int charSize = 20;

bool verify(RotatedRect rect) {
	float error = 0.4;
	const float aspect = 4.7272;
	int min = 15 * aspect * 15; // �������
	int max = 125 * aspect * 125; // �������

	float rmin = aspect - aspect * error; // ��߱�����
	float rmax = aspect + aspect * error; // ��߱�����

	int area = rect.size.width * rect.size.height; // �������
	float r = rect.size.width / rect.size.height;  // �����߱�
	r = r < 1 ? 1 / r : r;

	return area >= min && area <= max && r >= rmin && r <= rmax;
}


/* ����SVM��ͼ����� */
bool classification(Mat image_crop) {
	
	// ����ѵ������
	FileStorage fs;
	fs.open("SVM.xml", FileStorage::READ);

	Mat trainingDataMat;
	Mat classesMat;
	fs["TrainingData"] >> trainingDataMat;
	fs["classes"] >> classesMat;

	Ptr<TrainData> trainingData = TrainData::create(trainingDataMat, ROW_SAMPLE, classesMat);

    // �����������������ò���
	SVM::ParamTypes params;
	SVM::KernelTypes kernel_type = SVM::LINEAR;
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(kernel_type);

	// ѵ��������
	svm->trainAuto(trainingData);
	
	// Ԥ��
	Mat src;
	src.create(33, 144, CV_32FC1);
	resize(image_crop, src, src.size(), 0, 0, INTER_CUBIC);

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

/* ��֤��ĸ */
bool verifyLetter(Mat r) {
	const float aspect = 45.0f / 77.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.35;
	float minHeight = 15;
	float maxHeight = 28;
	float minAspect = 0.2;
	float maxAspect = aspect + aspect * error;
	float area = countNonZero(r);
	float bbArea = r.cols * r.rows;
	float percPixels = area / bbArea;
	return percPixels < 0.8 &&
		charAspect > minAspect &&
		charAspect < maxAspect &&
		r.rows >= minHeight && r.rows <= maxHeight;
	
}

/* �ۻ�ֱ��ͼ */
Mat ProjectHistogram(Mat img, int t) {
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = countNonZero(data);
	}

	double min, max;
	minMaxLoc(mhist, &min, &max); // �ҵ������е����ֵ

	if (max > 0) {
		// �����ÿһ��Ԫ�ض��������ֵ
		mhist.convertTo(mhist, -1, 1.0f/max, 0);
		return mhist;
	}
}


/* ������������ */
Mat features(Mat in, int sizeData) {

	// �ۻ�ֱ��ͼ
	Mat vhist = ProjectHistogram(in, VERTICAL);
	Mat hhist = ProjectHistogram(in, HORIZONTAL);

	// �ͷֱ�������
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));

	int numCols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;
	Mat out = Mat::zeros(1, numCols, CV_32F); // ������������

	// ����������ֵ
	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x < lowData.cols; x++) {
		for (int y = 0; y < lowData.rows; y++){
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}

	return out;
}

int classificationANN(Mat TrainingData, Mat classes, int nlayers, Mat f){

	// ����ѵ������
	Mat trainClasses;
	trainClasses.create(TrainingData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++) {
		for (int j = 0; j < trainClasses.cols; j++) {
			if (j == classes.at<int>(i))
				trainClasses.at<float>(i, j) = 1;
			else
				trainClasses.at<float>(i, j) = 0;
		}
	}
	// Mat weights(1, TrainingData.rows, CV_32FC1, Scalar::all(1));
	Ptr<TrainData> trainingData = TrainData::create(TrainingData, ROW_SAMPLE, trainClasses);
	
	// ����
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = TrainingData.cols;
	layers.at<int>(1) = nlayers;
	layers.at<int>(2) = numCharacters;


	// ����������
	Ptr<ANN_MLP> ann = ANN_MLP::create();
	ann->setLayerSizes(layers);
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
	// ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001);

	// ѵ��
	ann->train(trainingData);

	Mat output(1, numCharacters, CV_32FC1);

	// �������������Mat f
	Mat src;
	src.create(45, 77, CV_32FC1);
	resize(f, src, src.size(), 0, 0, INTER_CUBIC);
	src.convertTo(src, CV_32FC1);
	src = src.reshape(1, 1);

	ann->predict(f, output);
	
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc);

	return maxLoc.x;

}


int main()
{
	/* ͼ��Ԥ����

	string in = "images/2715DTZ.jpg";
	// string in = "images/3028BYS.JPG"; 

	Mat image = imread(in, IMREAD_GRAYSCALE);
	Mat image2 = imread(in);
	Mat image3 = imread(in, IMREAD_GRAYSCALE);

	if (image.empty()){
		return -1;
	}
	//imshow("��ԭʼͼ��", image);

	blur(image, image, Size(5, 5));
	//imshow("��ȥ���", image);


	Sobel(image, image, CV_8U, 1, 0, 3, 1, 0);
	//imshow("��sobel�˲���", image);

	threshold(image, image, 0, 255, CV_THRESH_OTSU);
	//imshow("��otsu��ֵ����", image);

	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	morphologyEx(image, image, CV_MOP_CLOSE, element);
	//imshow("�������㡿", image);

	vector<vector<Point>> contours;
	findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	map<int, RotatedRect> _map;

	for (int i = 0; i < contours.size(); i++) {
		drawContours(image, contours, i, Scalar(255), 1); // ��������

		// ���ƾ���
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f vertices[4];
		rect.points(vertices);
		for (int i = 0; i < 4; i++) {
			line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255), 2);
		}

		// ��֤
		if (verify(rect)) {
			_map[i] = rect;
		}
	}
	imshow("��������ȡ��", image);



	// ����ͨ����֤�ľ���
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

		// ѡ����ӽ��ľ���
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

	// imshow("��ͨ����֤��", image);

	

	// ������ӽ��ľ���
	RotatedRect rect = _map[index];
	Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; i++) {
		line(image2, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 10);
	}
	// imshow("����ӽ��ľ��Ρ�", image2);


	// ͼ���и�
	Mat image_crop;
	Size rect_size = rect.size;
	if (rect_size.width < rect_size.height) {
		swap(rect_size.width, rect_size.height);
	}
	//rect_size.width += 10;
	//rect_size.height += 10;
	getRectSubPix(image3, rect_size, rect.center, image_crop);
	imshow("���и��ĳ��ơ�", image_crop);

	// Mat src = imread("2715DTZ.JPG", 0);
	// imshow("src", src);

	// bool flag = classification(src);
	// bool flag = classification(image_crop);
	// cout << "flag = " << flag << endl;
	*/

    /* ���ƺ���ȡ
    Mat src = imread("2715DTZ.jpg", 0);
	Mat copy = imread("2715DTZ.jpg");

	equalizeHist(src, src);
	// imshow("�����⻯��ĻҶ�ͼ��", src);


	threshold(src, src, 60, 255, CV_THRESH_BINARY_INV);
	imshow("����ֵ�����ͼ��", src);

	Mat element = getStructuringElement(0, Size(3, 3));
	//���Ͳ���
	Mat dst;
	dilate(src, dst, element);
	imshow("�����ͺ��ͼ��", dst);

	vector<vector<Point>> contours;
	findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++) {
		//drawContours(copy, contours, i, Scalar(0, 255, 0), 1); // ��������
		Rect rect = boundingRect(contours[i]);
		rect.height += 1;
		rect.width += 1;
		//rectangle(copy, rect, Scalar(0,0,255), 1);
		

		Mat roi(src, rect);
		if (verifyLetter(roi)) {

			// ����ͨ����֤�ľ���
			rectangle(copy, rect, Scalar(255, 0, 0), 1);

			// ͼ���и�		
			imwrite( to_string(i) + ".jpg", roi);
		}
	}

	imshow("������������", copy);
	*/

    // ʶ��
    Mat src = imread("4.jpg", 0);
	


	FileStorage fs;
	fs.open("OCR.xml", FileStorage::READ);
	Mat TrainingData;
	Mat Classes;
	fs["TrainingDataF5"] >> TrainingData;
	fs["classes"] >> Classes;


	int h = src.rows;
	int w = src.cols;
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = m / 2 - w / 2;
	transformMat.at<float>(1, 2) = m / 2 - h / 2;

	Mat warpImage(m, m, src.type());
	warpAffine(src, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	Mat out;
	resize(warpImage, out, Size(charSize, charSize));
	imshow("out", out);


	Mat f = features(out, 5);

	int index = classificationANN(TrainingData, Classes, 10, f);
	
	cout << strCharacters[index] << endl;



	waitKey();
	return 0;
}