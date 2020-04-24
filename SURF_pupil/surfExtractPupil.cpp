#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp" //resize
#include <opencv2/tracking.hpp> // ROI
//#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
//int main()
{
    
    //********************Input**************//    
    //Mat src = imread("333.png");
    const Mat src = imread("9.jpeg"); //Load as grayscale
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
    Mat imCropRoi = resizedImg(r);

    // Display Cropped Image
    //imshow("Select Region of Interest", imCropRoi);

    //********************Region on interest**************//   

    //********************Convert to binary**************//
    Mat imgBin;
    threshold(imCropRoi, imgBin, 50, 255, THRESH_BINARY);
    imshow("Binary", imgBin);
    //********************Convert to binary**************//



    //********************Median Filter**************// 
    Mat imgMedian;
    medianBlur(imgBin, imgMedian, 15);

    //imshow("Median Blured image", imgMedian);

    //********************Median Filter**************//   


    //********************Color Scale Conversion**************//

    Mat imgGrey, imgHSV;
    cvtColor(imgMedian, imgGrey, cv::COLOR_BGR2GRAY);
    cvtColor(imgMedian, imgHSV, cv::COLOR_BGR2HSV);

    //********************Color Scale Conversion**************//
   


    //********************Canny Edge**************//
    Mat imgCanny;
    Canny(imgGrey, imgCanny, 0, 0 * 3, 5);
    //dst = Scalar::all(0);
    //src.copyTo(dst, detected_edges);
    imshow("Canny Edge", imgCanny);

    //********************Canny Edge**************//


     //********************Soble Filter**************//
    

    Mat soble_X, sobel_Y, laplacian;
    Mat abs_soble_X, abs_soble_Y, abs_laplac;
    int ksize=3, scale=1, delta=0;

    //Sobel(imgCanny, grad_x, CV_16S, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    //Sobel(imgCanny, grad_y, CV_16S, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    Laplacian(imgCanny, laplacian, CV_16S);
    convertScaleAbs(laplacian, abs_laplac);
    // converting back to CV_8U
    //convertScaleAbs(soble_X, abs_soble_X);
    //convertScaleAbs(soble_Y, abs_soble_Y);
    //addWeighted(abs_soble_X, 0.5, abs_soble_Y, 0.5, 0, imgCanny);
    imshow("Sobel Laplacian", abs_laplac);
    //********************Soble Filter**************//

    


    Ptr< xfeatures2d::SURF> sift = xfeatures2d::SURF::create();
    vector<KeyPoint> keypoints;
    Mat tro;
    sift->detect(abs_laplac, keypoints);

    // Add results to image and save.
    Mat output;
    drawKeypoints(imCropRoi, keypoints, output);
    //imwrite("surf_result.jpg", output);
    
    
    //********************ImShow**************//
    //cv::imshow("Source", src);
    //imshow("Resized Image", resizedImg);
    //imshow("Region of Interest", imCropRoi);
    //imshow("Median Blured image", imgMedian);
    //imshow("Grey Scale", imgGrey);
    //imshow("HSV Scale", imgHSV);
    //imshow("Canny Edge", imgCanny);
    //imshow("SOBEL Laplacian", abs_laplac);
    //********************ImShow**************//   
    
    
    imshow("SURF Keypoints", output);
    cv::waitKey(0);

    return 0;
}