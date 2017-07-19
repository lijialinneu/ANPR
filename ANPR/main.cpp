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

	waitKey();
	return 0;
}