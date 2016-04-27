/*
 * Author: Henrique Grandinetti Barbosa Amaral
 * Email: henriquegrandinetti@gmail.com
 * Computer Science, University of Bristol
 * Image Processing and Computer Vision
 *
 * Optical Flow - Lucas and Kanade Algorithm	
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <string>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

void Ixyt (Mat &first, Mat next, Mat Ix, Mat Iy, Mat It, Size region, int shift);
double IxRegion (Mat &first, Mat &next, Size regionSize, Point start);
double IyRegion (Mat &first, Mat &next, Size regionSize, Point start);
double ItRegion (Mat &first, Mat &next, Size regionSize, Point start);
void Normalize (Mat input);

int main( int argc, const char** argv )
{
    cv::VideoCapture cap;

    if(argc > 1){
        cap.open(string(argv[1]));
    }
    else{
        cap.open(CV_CAP_ANY);
    }

    if(!cap.isOpened()){
        printf("Error: could not load a camera or video.\n");
    }

    Mat firstbig;
    Mat nextbig;
    Mat first;
    Mat next;
    Mat firstgray;
    Mat nextgray;

    cap >> firstbig;
    cap >> nextbig;

    if (firstbig.empty() || nextbig.empty()) {
        return 0;
    }

    resize(firstbig, first, Size(firstbig.cols/2.5, firstbig.rows/2.5));
    resize(nextbig, next, Size(nextbig.cols/2.5, nextbig.rows/2.5));

    Size region(5, 5);
    int shift = 2;

    int width  = (first.cols - region.width)/shift + 1;
    int height = (first.rows - region.height)/shift + 1;

    Mat Ix(height, width, CV_8UC1);
    Mat Iy(height, width, CV_8UC1);
    Mat It(height, width, CV_8UC1);

    namedWindow("original", 1);
    namedWindow("Ix", 1);
    namedWindow("Iy", 1);
    namedWindow("It", 1);

    while(1){

        cvWaitKey(40);

        cvtColor(first, firstgray, CV_BGR2GRAY);
        cvtColor(next, nextgray, CV_BGR2GRAY);


        Ixyt(firstgray, nextgray, Ix, Iy, It, region, shift);

        Normalize(Ix);
        Normalize(Iy);
        Normalize(It);

        imshow("original", first);
        imshow("Ix", Ix);
        imshow("Iy", Iy);
        imshow("It", It);

        first = next.clone();

        cap >> nextbig;

        resize(nextbig, next, Size(nextbig.cols/2.5, nextbig.rows/2.5));
    }
}

void Ixyt (Mat &first, Mat next, Mat Ix, Mat Iy, Mat It, Size region, int shift){
    int x = 0, y = 0;

    for (int j = 0; j <= first.rows - region.height; j += shift){
        for (int i = 0; i <= next.cols - region.width; i += shift) {
            Ix.at<uchar>(y,x) = (255 + IxRegion(first, next, region, Point(i,j)))/2;
            Iy.at<uchar>(y,x) = (255 + IyRegion(first, next, region, Point(i,j)))/2;
            It.at<uchar>(y,x) = (255 + ItRegion(first, next, region, Point(i,j)))/2;
            x++;
        }
        x = 0;
        y++;
    }
}

double IxRegion (Mat &first, Mat &next, Size regionSize, Point start){
    double Ix = 0;

    for (int j = 0; j < regionSize.height; j++) {
        for (int i = 0; i < regionSize.width - 1; i++) {
            Ix += first.at<uchar>(Point(start.x + i + 1,start.y + j)) - first.at<uchar>(Point(start.x + i,start.y + j));
            Ix += next.at<uchar>(Point(start.x + i + 1,start.y + j)) - next.at<uchar>(Point(start.x + i,start.y + j));
        }
    }

    return Ix/(2 * regionSize.height * (regionSize.width - 1));
}

double IyRegion (Mat &first, Mat &next, Size regionSize, Point start){
    double Iy = 0;

    for (int i = 0; i < regionSize.width; i++) {
        for (int j = 0; j < regionSize.height - 1; j++) {
            Iy += first.at<uchar>(Point(start.x + i,start.y + j + 1)) - first.at<uchar>(Point(start.x + i,start.y + j));
            Iy += next.at<uchar>(Point(start.x + i,start.y + j + 1)) - next.at<uchar>(Point(start.x + i,start.y + j));
        }
    }

    return Iy/(2 * regionSize.width * (regionSize.height - 1));
}

double ItRegion (Mat &first, Mat &next, Size regionSize, Point start){
    double It = 0;

    for (int j = 0; j < regionSize.height; j++) {
        for (int i = 0; i < regionSize.width; i++) {
            It += next.at<uchar>(Point(start.x + i, start.y + j)) - first.at<uchar>(Point(start.x + i,start.y + j));
        }
    }

    return It/(regionSize.height * regionSize.width);
}

void Normalize (Mat input){
    int min = 256, max = -1;
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            if (input.at<uchar>(j,i) < min)
                min = input.at<uchar>(j,i);
            if (input.at<uchar>(j,i) > max)
                max = input.at<uchar>(j,i);
        }
    }

    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            input.at<uchar>(j,i) = 255 * ((float)(input.at<uchar>(j,i) - min) / (max - min));
        }
    }
}
