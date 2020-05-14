#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp" //resize
#include <opencv2/tracking.hpp> // ROI
//#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <chrono> 
using namespace std::chrono;

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])

{

    //********************Input**************//    

    const Mat src = imread("C:/Users/akeee/OneDrive/Desktop/test/SSS FIlter/samples/1.jpeg");
    //********************Input**************//   

    //********************Resize**************//   
    Mat resizedImg;
    cv::resize(src, resizedImg, cv::Size(src.cols / 2, src.rows / 2));
    //cv::imshow("Source", src);
    //cv::imshow("Resized Image", resizedImg);
    //********************Resize**************//   

    //********************Region on interest**************//   

    // Select ROI
    bool fromCenter = false;
    Rect2d r = selectROI(resizedImg, fromCenter);    //only box. no quadrant
    //Rect2d r = selectROI(resizedImg);     //4 quadrant

    // Crop image
    Mat imgCropRoi = resizedImg(r);

    // Display Cropped Image
    //imshow("Select Region of Interest", imgCropRoi);



    //********************Region on interest**************//   

    auto start = high_resolution_clock::now();   // Process time start 

    //********************Convert to binary**************//
    //Mat imgBin;
    //threshold(imgCropRoi, imgBin, 50, 255, THRESH_BINARY);
    //imshow("Binary", imgBin);
    //********************Convert to binary**************//






    //********************Color Scale Conversion**************//

    Mat imgGrey, imgHSV;
    //cvtColor(imgCropRoi, imgGrey, CV_BGR2GRAY);
    //cvtColor(imgMedian, imgHSV, cv::COLOR_BGR2HSV);

    //********************Color Scale Conversion**************//



    //********************Canny Edge**************//
    //Mat imgCanny;
    //Canny(imgGrey, imgCanny, 0, 0 * 3, 5);
    //dst = Scalar::all(0);
    //src.copyTo(dst, detected_edges);
    //imshow("Canny Edge", imgCanny);

    //********************Canny Edge**************//


     //********************Soble Filter**************//


    Mat sobel_X, sobel_Y, laplacian;
    Mat abs_soble_X, abs_soble_Y, abs_laplac, imgSobel;
    int ksize = 3, scale = 3, delta = 1;

    Sobel(imgCropRoi, sobel_X, CV_16S, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(imgCropRoi, sobel_Y, CV_16S, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    Laplacian(imgCropRoi, laplacian, CV_16S);
    convertScaleAbs(laplacian, abs_laplac);
    // converting back to CV_8U
    convertScaleAbs(sobel_X, abs_soble_X);
    convertScaleAbs(sobel_Y, abs_soble_Y);
    addWeighted(abs_soble_X, 0.5, abs_soble_Y, 0.5, 0, imgSobel);
    imshow("Sobel Laplacian", abs_laplac);
    //imshow("Sobel X", abs_soble_X);
    //imshow("Sobel Y", abs_soble_Y);
    imshow("Sobel X+Y", imgSobel);
    //********************Soble Filter**************//
    //********************Median Filter**************// 
    Mat imgMedian;
    //medianBlur(imgCropRoi, imgMedian, 15);
    medianBlur(imgSobel, imgMedian, 3);

    //imshow("Median Blured image", imgMedian);

    //********************Median Filter**************// 


    //********************SURF**************// 

    Ptr< xfeatures2d::SURF> SURFdetector = xfeatures2d::SURF::create(800);
    vector<KeyPoint> keypoints;
    Mat tro;
    SURFdetector->detect(imgMedian, keypoints);

    // Add results to image and save.
    Mat output;
    drawKeypoints(imgCropRoi, keypoints, output);
    //imwrite("surf_result.jpg", output);
    imshow("SURF Keypoints", output);
    //********************SURF**************// 

    //********************ImShow**************//
    //imshow("Source", src);
    //imshow("Resized Image", resizedImg);
    //imshow("Region of Interest", imCropRoi);
    //imshow("Median Blured image", imgMedian);
    //imshow("Grey Scale", imgGrey);
    //imshow("HSV Scale", imgHSV);
    //imshow("Canny Edge", imgCanny);
    //imshow("SOBEL Laplacian", abs_laplac);
    //********************ImShow**************//   




     //********************Circle Detection**************// 

    vector<Vec3f> circles;
    cvtColor(imgMedian, imgGrey, CV_BGR2GRAY);

    HoughCircles(imgGrey, circles, CV_HOUGH_GRADIENT, 1, 60, 200, 20, 0, 0);
    //HoughCircles(imgMedian, circles, CV_HOUGH_GRADIENT, 1, imgMedian.rows / 8, 200, 100, 0, 0);

    for (size_t i = 0; i < 1; i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(imgCropRoi, center, 3, Scalar(0, 255, 0), -1);
        circle(imgCropRoi, center, radius, Scalar(0, 0, 255), 2);
    }

    imshow("Hough Circle", imgCropRoi);

    //********************Circle Detection**************// 

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    // To get the value of duration use the count() 
    // member function on the duration object 
    cout << duration.count() << endl;


    //imwrite("C:/Users/akeee/OneDrive/Desktop/Output/Detected1.jpg", src);

    cv::waitKey(0);

    return 0;
}