#include <iostream>
#include <hash_map>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;


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


int main()
{
	string in = "images/2715DTZ.jpg";
	Mat image = imread(in, IMREAD_GRAYSCALE);
	Mat image2 = imread(in);

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

	hash_map<int, RotatedRect> map;
	// vector<RotatedRect> rects;

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
			cout << "ͨ����֤" << endl;
			//rects.push_back(rect);
			map[i] = rect;
		}
	}
	//imshow("��������ȡ��", image);


	cout << "ͨ����֤�ľ��θ���" << map.size() << endl;





	// ����ͨ����֤�ľ���
	int min_diff = 100000;
	int index = 0;
	const float square = 27.75;
	int size = map.size();
	for (int i = 0; i < size; i++) {
		RotatedRect rect = map[i];
		Point2f vertices[4];
		rect.points(vertices);
		for (int j = 0; j < 4; j++) {
			line(image, vertices[j], vertices[(j + 1) % 4], Scalar(255), 10);
		}

		// ѡ����ӽ��ľ���
		int perimeter = arcLength(contours[i], true);
		int area = contourArea(contours[i]);
		if (area != 0) {
			int squareness = perimeter * perimeter / area;
			cout << squareness << endl;
			float diff = abs(squareness - square);
			if (diff < min_diff) {

				min_diff = diff;
				index = i;

			}
		}

	}
	imshow("��ͨ����֤��", image);

	cout << index << endl;
	// ������ӽ��ľ���
	RotatedRect rect = map[index];
	Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; i++) {
		cout << " asdf" << endl;
		line(image2, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 10);
	}
	imshow("����ӽ��ľ��Ρ�", image2);


	//   Canny(image, image, 50, 200, 3); // Apply canny edge
	//   // Create and LSD detector with standard or no refinement.
	//   Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);

	//   double start = double(getTickCount());
	//   vector<Vec4f> lines_std;
	//   // Detect the lines
	//   ls->detect(image, lines_std);
	//   double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	//   std::cout << "It took " << duration_ms << " ms." << std::endl;
	//   // Show found lines
	//   Mat drawnLines(image);
	//Mat dst;
	//dst.create(image.size(), image.type());

	//   ls->drawSegments(drawnLines, lines_std);
	//ls->drawSegments(dst, lines_std);

	//   imshow("Standard refinement", drawnLines);
	//imshow("LSDֱ�߼��", dst);

	waitKey();
	return 0;
}