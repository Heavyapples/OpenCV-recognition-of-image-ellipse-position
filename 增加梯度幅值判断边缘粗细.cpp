#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

// 函数声明
RotatedRect detectLargestEllipse(Mat& src);

int main() {
    // 导入图片
    Mat src = imread("C:/Users/13729/Documents/WeChat Files/wxid_a6l9v8idcwc822/FileStorage/File/2023-04/test/test/test5.jpg", IMREAD_COLOR);
    if (src.empty()) {
        cerr << "无法读取图片" << endl;
        return -1;
    }

    // 识别图像上的最大圆或椭圆
    RotatedRect largestEllipse = detectLargestEllipse(src);

    // 输出圆或椭圆的圆心和长短轴
    Point2f center = largestEllipse.center;
    Size2f axes = largestEllipse.size;

    double a, b;
    cout << "识别到的印章:" << endl;
    cout << "圆心: (" << center.x << ", " << center.y << ")" << endl;
    if (axes.width >= axes.height) {
        cout << "长轴: " << axes.width << ", 短轴: " << axes.height << endl;
        b = axes.height / 2;
        a = axes.width / 2;
    }
    else {
        cout << "长轴: " << axes.height << ", 短轴: " << axes.width << endl;
        a = axes.height / 2;
        b = axes.width / 2;
    }
    ellipse(src, Point(center.x, center.y), Size(a, b), 0, 0, 360, 1, LINE_8);
    imwrite(string("C:/Users/13729/Documents/WeChat Files/wxid_a6l9v8idcwc822/FileStorage/File/2023-04/test/test/result.jpg"), src);
    return 0;
}

RotatedRect detectLargestEllipse(Mat& src) {
    Mat hsv, redMask1, redMask2, redMask, gray, blurred, binary, grad_mag;

    double edge_thickness_lower_bound = 50;
    double edge_thickness_upper_bound = 100;

    // 将图像从 BGR 转换为 HSV
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // 定义红色的 HSV 范围，分为两个区间
    inRange(hsv, Scalar(0, 70, 50), Scalar(10, 255, 255), redMask1);
    inRange(hsv, Scalar(170, 70, 50), Scalar(180, 255, 255), redMask2);

    // 合并两个红色区间的掩膜
    redMask = redMask1 | redMask2;

    // 将原始图像与红色掩膜进行按位与操作，只保留红色区域
    Mat redRegion;
    bitwise_and(src, src, redRegion, redMask);

    // 将红色区域图像转换为灰度图像
    cvtColor(redRegion, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, blurred, Size(5, 5), 1);
    Canny(blurred, binary, 50, 150);

    Mat grad_x, grad_y;
    Sobel(blurred, grad_x, CV_16S, 1, 0, 3);
    Sobel(blurred, grad_y, CV_16S, 0, 1, 3);
    convertScaleAbs(grad_x, grad_x);
    convertScaleAbs(grad_y, grad_y);
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad_mag);


    // 形态学运算，填充椭圆
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    dilate(binary, binary, kernel, Point(-1, -1), 2);

    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    RotatedRect largestEllipse;
    double maxArea = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() > 5) {
            RotatedRect ellipse = fitEllipse(contours[i]);
            double area = ellipse.size.width * ellipse.size.height;
            double axis_ratio = ellipse.size.width / ellipse.size.height;

            // 检查长轴和短轴的比例
            if (axis_ratio > 5 || 1.0 / axis_ratio > 5) {
                continue;
            }

            // 检查椭圆尺寸是否在合理范围内
            if (ellipse.size.width > src.cols || ellipse.size.height > src.rows) {
                continue;
            }

            // 检查椭圆面积占整个图像面积的比例
            double area_ratio = area / (src.cols * src.rows);
            if (area_ratio < 0.01 || area_ratio > 0.95) {
                continue;
            }

            double avg_grad_mag = mean(grad_mag, binary)[0];
            if (avg_grad_mag < edge_thickness_lower_bound || avg_grad_mag > edge_thickness_upper_bound) {
                continue;
            }

            if (area > maxArea) {
                maxArea = area;
                largestEllipse = ellipse;
            }
        }
    }

    return largestEllipse;
}