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
	int min = 15 * aspect * 15; // 面积下限
	int max = 125 * aspect * 125; // 面积上限

	float rmin = aspect - aspect * error; // 宽高比下限
	float rmax = aspect + aspect * error; // 宽高比上限

	int area = rect.size.width * rect.size.height; // 计算面积
	float r = rect.size.width / rect.size.height;  // 计算宽高比
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

	hash_map<int, RotatedRect> map;
	// vector<RotatedRect> rects;

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
			cout << "通过验证" << endl;
			//rects.push_back(rect);
			map[i] = rect;
		}
	}
	//imshow("【轮廓提取】", image);


	cout << "通过验证的矩形个数" << map.size() << endl;





	// 绘制通过验证的矩形
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

		// 选择最接近的矩形
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
	imshow("【通过验证】", image);

	cout << index << endl;
	// 绘制最接近的矩形
	RotatedRect rect = map[index];
	Point2f vertices[4];
	rect.points(vertices);
	for (int i = 0; i < 4; i++) {
		cout << " asdf" << endl;
		line(image2, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 10);
	}
	imshow("【最接近的矩形】", image2);


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
	//imshow("LSD直线检测", dst);

	waitKey();
	return 0;
}