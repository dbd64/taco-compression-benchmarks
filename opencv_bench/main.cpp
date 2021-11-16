#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

#include "timers.h"

using namespace cv;

#define TACO_TIME_REPEAT(CODE, TEARDOWN, REPEAT, RES, COLD) {  \
    Timer timer;                                     \
    for(int i=0; i<REPEAT; i++) {                    \
      if(COLD)                                       \
        timer.clear_cache();                         \
      timer.start();                                 \
      CODE;                                          \
      timer.stop();                                  \
      TEARDOWN;                                      \
    }                                                \
    RES = timer.getResult();                         \
    double* dummy_result = timer.dummy();               \
    DoNotOptimizePtr(dummy_result);                     \
  }

#define TOOL_BENCHMARK_REPEAT(CODE, NAME, REPEAT, STREAM) {      \
    if (time) {                                                  \
      TACO_TIME_REPEAT(CODE,REPEAT,timevalue,false);             \
      STREAM << timevalue << std::endl;                          \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

#define TOOL_BENCHMARK_REPEAT_COLD(CODE, NAME, REPEAT, STREAM) { \
    if (time) {                                                  \
      TACO_TIME_REPEAT(CODE,{},REPEAT,timevalue,true);              \
      STREAM << timevalue << std::endl;                          \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}

#define TOOL_BENCHMARK_REPEAT_COLD_TD(CODE, TEARDOWN, NAME, REPEAT, STREAM) { \
    if (time) {                                                  \
      TACO_TIME_REPEAT(CODE,TEARDOWN,REPEAT,timevalue,true);              \
      STREAM << timevalue << std::endl;                          \
    }                                                            \
    else {                                                       \
      CODE;                                                      \
    }                                                            \
}



std::string getEnvVar(std::string varname) {
  auto path = std::getenv(varname.c_str());
  if (path == nullptr) {
    return "";
  }
  return std::string(path);
}

int NUM_IMGS = 340;
std::string artifact_root = "/home/artifact/artifact";


int getNumRepetitions(int r){
  auto rep = getEnvVar("REPETITIONS");
  return rep == "" ? r :  std::stoi(rep);
}

void updateNumImgs(){
  auto rep = getEnvVar("NUM_IMGS");
  NUM_IMGS = rep == "" ? NUM_IMGS :  std::stoi(rep);
}

void writeHeader(std::ostream& outputFile, int repetition){
    outputFile << "index,mean,stddev,median,";
    for (int i=0; i<repetition-1; i++){
      outputFile << i << ",";
    }
    outputFile << repetition-1 << std::endl;
}

void mri_cold(std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(100);
    std::ofstream outputFile(outfile);

    writeHeader(outputFile, repetition);

    std::string folder = artifact_root + "/data/mri/";

    std::vector<std::vector<double>> times;
    for (int i = 0; i < 253; i++){
      times.push_back({});
    }

    for (int r=0; r<repetition; r++){
      for (int i=1; i<=253; i++){
        std::cout << "MRI: " << i << std::endl;
        Mat t1  = imread(folder + "img_t1_DENSE_" +  std::to_string(i) + ".png", IMREAD_GRAYSCALE);
        t1 -= 254;
        Mat t2  = imread(folder + "img_t2_DENSE_" +  std::to_string(i) + ".png", IMREAD_GRAYSCALE);
        t2 -= 254;
        Mat roi = imread(folder + "roi_DENSE_" +  std::to_string(i) + ".png", IMREAD_GRAYSCALE);
        roi -= 254;

        TACO_TIME_REPEAT({
            Mat out;
            out = (t1 & roi) ^ (t2 & roi);
            DoNotOptimize(out);
        }, {}, 1, timevalue, true);
        times[i].push_back(timevalue.mean);
        std::cout << "time " << r << ", " << i << " : " << timevalue.mean << std::endl;
      }
    }

    for (int i=1; i<=253; i++){
      int repeat = static_cast<int>(times[i].size());
      double mean=0.0;
      mean = accumulate(times[i].begin(), times[i].end(), 0.0);
      mean = mean/repeat;
      outputFile << std::to_string(i) << "," << mean << ",,,";
      for (int j=0; j< repeat-1; j++){
        outputFile << times[i][j] << ",";
      }
      outputFile << times[i][repeat-1] << std::endl;
    }
}

void mri(std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(1000);
    std::ofstream outputFile(outfile);

    writeHeader(outputFile, repetition);

    std::string folder = artifact_root + "/data/mri/";
    for (int i=1; i<=253; i++){
        std::cout << "MRI: " << i << std::endl;
        Mat t1  = imread(folder + "img_t1_DENSE_" +  std::to_string(i) + ".png", IMREAD_GRAYSCALE);
        t1 -= 254;
        Mat t2  = imread(folder + "img_t2_DENSE_" +  std::to_string(i) + ".png", IMREAD_GRAYSCALE);
        t2 -= 254;
        Mat roi = imread(folder + "roi_DENSE_" +  std::to_string(i) + ".png", IMREAD_GRAYSCALE);
        roi -= 254;

        outputFile << std::to_string(i) << ",";
        TOOL_BENCHMARK_REPEAT_COLD({
            Mat out;
            out = (t1 & roi) ^ (t2 & roi);
        }, "Compute", repetition, outputFile);

        // out *= 255;
        // imwrite(artifact_root + "/out/opencv/mri/" + std::to_string(i) + ".png", out);
    }
}

std::string numToPath(int i, std::string folder){
    std::string z = std::to_string(0);
    std::string path = folder;
    if (i<10) path += z + z;
    else if (i<100) path += z;
    path += std::to_string(i);
    path += ".png";
    return path;
}

Mat global;

void brighten(std::string folder, std::string validation, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(10);
    std::ofstream outputFile(outfile);

    writeHeader(outputFile, repetition);

    std::vector<std::vector<double>> times;
    for (int i = 0; i < NUM_IMGS; i++){
      times.push_back({});
    }

    for (int r=0; r<repetition; r++){
      for (int i=1; i<=NUM_IMGS; i++){
        auto path = numToPath(i, folder);
        std::cout << "brighten(" << r << "): " << path << std::endl;
        Mat img = imread(path, IMREAD_COLOR);

        TACO_TIME_REPEAT({
          Mat new_image;
          img.convertTo(new_image, -1, 1, 20);
          DoNotOptimize(new_image);
        }, {}, 1, timevalue, true);
        times[i].push_back(timevalue.mean);
        std::cout << "time " << r << ", " << i << " : " << timevalue.mean << std::endl;
      }
    }

    for (int i=1; i<=NUM_IMGS; i++){
      int repeat = static_cast<int>(times[i].size());
      double mean=0.0;
      mean = accumulate(times[i].begin(), times[i].end(), 0.0);
      mean = mean/repeat;
      outputFile << std::to_string(i) << "," << mean << ",,,";
      for (int j=0; j< repeat-1; j++){
        outputFile << times[i][j] << ",";
      }
      outputFile << times[i][repeat-1] << std::endl;
    }
}

void  __attribute__ ((noinline)) alpha_kernel(int repetition, Mat& img1, Mat& img2, double alpha, double beta, std::ostream& outputFile){
  TimeResults timevalue{};
  bool time = true;
  TOOL_BENCHMARK_REPEAT_COLD({
    Mat dst;
    addWeighted( img1, alpha, img2, beta, 0.0, dst);
    DoNotOptimize(dst);
  }, "Compute", repetition, outputFile);
}

void alpha(std::string folder1, std::string folder2, std::string validation, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(100);
    // std::ofstream outputFile(outfile);
    std::ostream& outputFile = std::cout;

    writeHeader(outputFile, repetition);

    for (int i=1; i<=NUM_IMGS; i++){
        auto path1 = numToPath(i, folder1);
        Mat img1 = imread(path1, IMREAD_COLOR);
        auto path2 = numToPath(i, folder2);
        Mat img2 = imread(path2, IMREAD_COLOR);
        std::cout << "alpha: " << path1 << std::endl;

        auto alpha = 0.7;
        auto beta = ( 1.0 - alpha );
        outputFile << std::to_string(i) << ",";
        alpha_kernel(repetition,img1,img2,alpha,beta,outputFile);
        // TOOL_BENCHMARK_REPEAT_COLD({
        //   Mat dst;
        //   addWeighted( img1, alpha, img2, beta, 0.0, dst);
        //   DoNotOptimize(dst);
        // }, "Compute", repetition, outputFile);

        Mat dst;
        addWeighted( img1, alpha, img2, beta, 0.0, dst);
    }
}

Mat generateMask(const Mat& img1){
  Mat dst = Mat::zeros( img1.size(), CV_8U );
  for( int y = 0; y < img1.rows/2; y++ ) {
    for( int x = 0; x < img1.cols/2; x++ ) {
            dst.at<uchar>(y,x) = 1;
    }
  }
  for( int y = img1.rows/2; y < img1.rows; y++ ) {
    for( int x = img1.cols/2; x < img1.cols; x++ ) {
            dst.at<uchar>(y,x) = 1;
    }
  }
  return dst;
}

void mask(std::string folder1, std::string folder2, std::string validation, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(100);
    // std::ofstream outputFile(outfile);
    std::ostream& outputFile = std::cout;

    writeHeader(outputFile, repetition);

    for (int i=1; i<=NUM_IMGS; i++){
        auto path1 = numToPath(i, folder1);
        Mat img1 = imread(path1, IMREAD_COLOR);
        auto path2 = numToPath(i, folder2);
        Mat img2 = imread(path2, IMREAD_COLOR);
        Mat m = generateMask(img1);
        std::cout << "mask: " << path1 << std::endl;

        Mat dst = Mat::zeros( img1.size(), img1.type() );

        std::vector<cv::Mat> a_channels;
        std::vector<cv::Mat> b_channels;

        cv::split(img1, a_channels);
        cv::split(img2, b_channels);
        outputFile << std::to_string(i) << ",";
        TOOL_BENCHMARK_REPEAT_COLD({
          std::vector<cv::Mat> c_channels;
          Mat not_m;
          bitwise_not(m, not_m);
          for(int i = 0; i < a_channels.size(); i++)
          {
              c_channels.push_back(
                a_channels[i].mul(m) + b_channels[i].mul(not_m)
              );
          }
          Mat dst;
          cv::merge(c_channels, dst);
        }, "Compute", repetition, outputFile);
        // imwrite(validation + std::to_string(i) + ".png", dst);
    }
}

void read(std::string folder, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(25);
    std::ofstream outputFile(outfile);

    writeHeader(outputFile, repetition);

    for (int i=1; i<=NUM_IMGS; i++){
      auto path = numToPath(i, folder);
      std::cout << "READING: " << path << std::endl;
      Mat img = imread(path, IMREAD_COLOR);
      std::vector<uchar> bytes;
      imencode(".png", img, bytes, {IMWRITE_PNG_STRATEGY_DEFAULT});

      outputFile << std::to_string(i) << ",";
      TOOL_BENCHMARK_REPEAT_COLD({
        Mat dst = imdecode(bytes, IMREAD_UNCHANGED);
        DoNotOptimize(dst);
      }, "Compute", repetition, outputFile);

      // imwrite(validation + std::to_string(i) + ".png", img);
    }
}

void compress(std::string folder, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(100);
    std::ofstream outputFile(outfile);

    writeHeader(outputFile, repetition);

    for (int i=1; i<=NUM_IMGS; i++){
      auto path = numToPath(i, folder);
      std::cout << "COMPRESSING: " << path << std::endl;
      Mat img = imread(path, IMREAD_COLOR);

      outputFile << std::to_string(i) << ",";
      TOOL_BENCHMARK_REPEAT_COLD({
        std::vector<uchar> bytes;
        imencode(".png", img, bytes, {IMWRITE_PNG_STRATEGY_DEFAULT});
        DoNotOptimize(bytes);
      }, "Compute", repetition, outputFile);
    }
}

void subtitle(std::string folder1, std::string validation, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(10);
    std::ofstream outputFile(outfile);

    auto path1 = numToPath(1, folder1);
    Mat img1 = imread(path1, IMREAD_COLOR);
    int width = img1.size().width;
    int height = img1.size().height;
    std::string subtitlePath = artifact_root + "/data/clips/subtitle_" + std::to_string(width) + "_" + std::to_string(height) + ".png";
    Mat s = imread(subtitlePath, IMREAD_UNCHANGED);
    Mat subtitle = Mat::zeros( cv::Size(width, height), CV_8U );
    Mat mask = Mat::zeros( cv::Size(width, height), CV_8U );

    for (int y = 0; y < s.rows; ++y) {
      for (int x = 0; x < s.cols; ++x) {
        cv::Vec4b & pixel = s.at<cv::Vec4b>(y, x);
        if (!pixel[3]){
          mask.at<cv::Scalar>(y,x) = 1;
          subtitle.at<cv::Scalar>(y,x) = pixel[0] ? 255 : 0;
        }
      }
    }

    writeHeader(outputFile, repetition);

    for (int i=1; i<=NUM_IMGS; i++){
      std::cout << "Img: " << i << std::endl;
        auto path1 = numToPath(i, folder1);
        Mat img1 = imread(path1, IMREAD_COLOR);
        Mat dst = Mat::zeros( img1.size(), img1.type() );

        std::vector<cv::Mat> a_channels;
        cv::split(img1, a_channels);
        outputFile << std::to_string(i) << ",";
        TOOL_BENCHMARK_REPEAT_COLD({
          std::vector<cv::Mat> c_channels;
          Mat not_m;
          bitwise_not(mask, not_m);
          for(int i = 0; i < a_channels.size(); i++)
          {
              c_channels.push_back(
                a_channels[i].mul(mask) + subtitle.mul(not_m)
              );
          }
          Mat dst;
          cv::merge(c_channels, dst);
        }, "Compute", repetition, outputFile);
        // imwrite(validation + std::to_string(i) + ".png", dst);
    }
}

void alpha_sketch_cold(std::string folder, std::string validation, std::string outfile){
    TimeResults timevalue{};
    bool time = true;
    int repetition = getNumRepetitions(10);
    std::ofstream outputFile(outfile);

    writeHeader(outputFile, repetition);

    auto alpha = 0.5;

    int start = 0;
    auto rep = getEnvVar("NUM_IMGS");
    NUM_IMGS = rep == "" ? 1000 :  std::stoi(rep);

    std::vector<std::vector<double>> times;
    for (int i = start; i < NUM_IMGS; i++){
      times.push_back({});
    }

    for (int r=0; r<repetition; r++){
      for (int i=1; i<=NUM_IMGS; i++){
        std::ostringstream stringStream;
        stringStream << folder;
        stringStream << i << ".png";
        std::string path1 = stringStream.str();

        std::ostringstream stringStream2;
        stringStream2 << folder;
        stringStream2 << i+1000 << ".png";
        std::string path2 = stringStream2.str();

        std::cout << "alpha sketch (" << r << ") : " << path1 << std::endl;
        Mat img1 = imread(path1, IMREAD_GRAYSCALE);
        std::cout << "alpha sketch (" << r << ") : " << path2 << std::endl;
        Mat img2 = imread(path2, IMREAD_GRAYSCALE);

        TACO_TIME_REPEAT({
          Mat dst;
          addWeighted(img1, alpha, img2, alpha, 0.0, dst);
          DoNotOptimize(dst);
        }, {}, 1, timevalue, true);
        times[i].push_back(timevalue.mean);
        std::cout << "time " << r << ", " << i << " : " << timevalue.mean << std::endl;
      }
    }

    for (int i=1; i<=NUM_IMGS; i++){
      int repeat = static_cast<int>(times[i].size());
      double mean=0.0;
      mean = accumulate(times[i].begin(), times[i].end(), 0.0);
      mean = mean/repeat;
      outputFile << std::to_string(i) << "," << mean << ",,,";
      for (int j=0; j< repeat-1; j++){
        outputFile << times[i][j] << ",";
      }
      outputFile << times[i][repeat-1] << std::endl;
    }
}

int main(){
  auto bench = getEnvVar("BENCH");
  auto folder1 = getEnvVar("PATH1");
  auto folder2 = getEnvVar("PATH2");
  auto name = getEnvVar("NAME");

  updateNumImgs();
  if (getEnvVar("ARTIFACT_ROOT") != ""){
    artifact_root = getEnvVar("ARTIFACT_ROOT");
  }


  if (bench == "mri"){
    mri_cold(artifact_root + "/out/opencv/mri.csv");
  } else if (bench == "brighten"){
    brighten(folder1, "", artifact_root + "/out/opencv/brighten/" + name + ".csv");
  } else if (bench == "alpha"){ 
    // alpha(folder1, folder2, "", artifact_root + "/out/opencv/alpha/" + name + ".csv");
    alpha_sketch_cold(artifact_root + "/data/sketches/", "",  artifact_root + "/out/opencv/alpha_sketch.csv");
  } else if (bench == "mask"){
    mask(folder1, folder2, "", artifact_root + "/out/opencv/mask/" + name + ".csv");
  } else if (bench == "read"){
    read(folder1, artifact_root + "/out/opencv/read/" + name + ".csv");
  } else if (bench == "compress"){
    compress(folder1, artifact_root + "/out/opencv/compress/" + name + ".csv");
  } else if (bench == "subtitle"){
    subtitle(folder1, "", artifact_root + "/out/opencv/subtitle/" + name + ".csv");
  }
}