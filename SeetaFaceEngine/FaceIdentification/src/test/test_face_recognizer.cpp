/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Identification module, containing codes implementing the
 * face identification method described in the following paper:
 *
 *   
 *   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
 *   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
 *   In Frontiers of Computer Science.
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developped by Wanglong Wu(a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems. 
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include<iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix


#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "face_identification.h"
#include "common.h"

#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <stdlib.h>
#include <stdio.h>

#include "ctime"

using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "
std::string choose ;

std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
void TEST(FaceRecognizerTest, CropFace) {
  FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
  std::string test_dir = DATA_DIR + "test_face_recognizer/";
  /* data initialize */
  std::ifstream ifs;
  std::string img_name;
  FacialLandmark pt5[5];
  ifs.open(test_dir + "test_file_list.txt", std::ifstream::in);
  clock_t start, count = 0;
  int img_num = 0;
  while (ifs >> img_name) {
    img_num ++ ;
    // read image
    Mat src_img = imread(test_dir + img_name, 1);
    EXPECT_NE(src_img.data, nullptr) << "Load image error!";

    // ImageData store data of an image without memory alignment.
    ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
    src_img_data.data = src_img.data;

    // 5 located landmark points (left eye, right eye, nose, left and right
    // corner of mouse).
    for (int i = 0; i < 5; ++ i) {
      ifs >> pt5[i].x >> pt5[i].y;
    }
/*
    // Create a image to store crop face.
    Mat dst_img(face_recognizer.crop_height(),
      face_recognizer.crop_width(),
      CV_8UC(face_recognizer.crop_channels()));
    ImageData dst_img_data(dst_img.cols, dst_img.rows, dst_img.channels());
    dst_img_data.data = dst_img.data;
    // Crop Face
    start = clock();
    face_recognizer.CropFace(src_img_data, pt5, dst_img_data);
    count += clock() - start;
    // Show crop face
        int i = 0;
        std::string num = std::to_string(i);
        imshow("Crop Face", dst_img);
        imwrite(num + ".jpg",dst_img);
        waitKey(0);
        destroyWindow("Crop Face");
        i ++;
  */}
  ifs.close();
  std::cout << "Test successful! \nAverage crop face time: "
    << 1000.0 * count / CLOCKS_PER_SEC / img_num << "ms" << std::endl;
}



void TEST(FaceRecognizerTest, ExtractFeatureWithCrop) {
//int databasechoose = 0;//0代表女网红脸，1代表男网红脸，2代表猛男脸，3代表萌妹脸，4代表呆萌脸
//std::cout << "choose database: " << endl;
//std::cin >> databasechoose;
int databasechoose = atoi(choose.c_str());
  FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
  std::string test_dir = DATA_DIR + "test_face_recognizer/";

  int feat_size = face_recognizer.feature_size();
  EXPECT_EQ(feat_size, 2048);

  FILE* feat_file = NULL;

  // Load features extract from caffe
  fopen_s(&feat_file, (test_dir + "feats.dat").c_str(), "rb");
  int n, c, h, w;
  EXPECT_EQ(fread(&n, sizeof(int), 1, feat_file), (unsigned int)1);
  EXPECT_EQ(fread(&c, sizeof(int), 1, feat_file), (unsigned int)1);
  EXPECT_EQ(fread(&h, sizeof(int), 1, feat_file), (unsigned int)1);
  EXPECT_EQ(fread(&w, sizeof(int), 1, feat_file), (unsigned int)1);
  float* feat_sdk_test = new float[n * c * h * w];
  float* feat_sdk = new float[n * c * h * w];
  //EXPECT_EQ(fread(feat_sdk_test, sizeof(float), n * c * h * w, feat_file),
  //  n * c * h * w);
  EXPECT_EQ(feat_size, c * h * w);

  int cnt = 0;

  /* Data initialize */
std::ifstream ifs;
if(databasechoose == 0)	  ifs.open(test_dir + "test_file_list_girl.txt", std::ifstream::in); 
else if(databasechoose == 1)   ifs.open(test_dir + "test_file_list_boy.txt", std::ifstream::in); 
else if(databasechoose == 2)	ifs.open(test_dir + "test_file_list_man.txt",std::ifstream::in);
else if(databasechoose == 3)	ifs.open(test_dir + "test_file_list_sister.txt",std::ifstream::in);
else if(databasechoose == 4)	ifs.open(test_dir + "test_file_list_cute.txt",std::ifstream::in);
  std::ifstream ifs_test("/home/dh/program/SeetaFaceEngine/FaceIdentification/data.txt");

  std::string img_name;
  FacialLandmark pt5[5];
  FacialLandmark pt5_test[5];

      for (int i = 0; i < 5; ++ i) {
        ifs_test >> pt5_test[i].x >> pt5_test[i].y;
      }
  clock_t start, count = 0;
  int img_num = 0;
  Mat test_img = imread("input.jpg");
  EXPECT_NE(test_img.data, nullptr) << "Load image error1!";
  double average_sim = 0.0;

  // ImageData store data of an image without memory alignment.
  ImageData test_img_data(test_img.cols, test_img.rows, test_img.channels());
  test_img_data.data = test_img.data;
  /* Extract feature: ExtractFeatureWithCrop */
  start = clock();
  face_recognizer.ExtractFeatureWithCrop(test_img_data, pt5_test,
    feat_sdk_test);
  count += clock() - start;

  while (ifs >> img_name) {
    // read image
    Mat src_img = imread(test_dir + img_name, 1);
    EXPECT_NE(src_img.data, nullptr) << "Load image error2!";

    // ImageData store data of an image without memory alignment.
    ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
    src_img_data.data = src_img.data;

    // 5 located landmark points (left eye, right eye, nose, left and right 
    // corner of mouse).
    for (int i = 0; i < 5; ++ i) {
      ifs >> pt5[i].x >> pt5[i].y;
    }

    /* Extract feature: ExtractFeatureWithCrop */
    start = clock();
    face_recognizer.ExtractFeatureWithCrop(src_img_data, pt5,
      feat_sdk + img_num * feat_size);
    count += clock() - start;

    /* Caculate similarity*/
    float* feat1 = feat_sdk_test;
    float* feat2 = feat_sdk + img_num * feat_size;

    float sim = face_recognizer.CalcSimilarity(feat1, feat2);
    //cout << sim <<endl;
    average_sim += sim;
    img_num ++ ;
  }
  ifs.close();
  average_sim /= img_num;
  average_sim += 0.5;
  //if (databasechoose == 0) average_sim += 0.05;
 // if (average_sim >= 0.9) average_sim = 100;
  //else if(average_sim >= 0.70)
  average_sim = average_sim * 100;
 // else average_sim = 60;
if (average_sim > 100) average_sim = 100;
    std::cout<< "average similarity: " << (int)average_sim << "%" << std::endl;
 
  delete []feat_sdk_test;
  delete []feat_sdk;
}

int main(int argc, char* argv[]) {
  choose = argv[1];
  TEST(FaceRecognizerTest, CropFace);
  //TEST(FaceRecognizerTest, ExtractFeature);
  TEST(FaceRecognizerTest, ExtractFeatureWithCrop);
  return 0;
}
