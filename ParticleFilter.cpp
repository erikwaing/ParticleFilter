//
//  ParticleFilter.cpp
//
//  Created by Erik Waingarten on 7/31/13.
//
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <assert.h>
#include "ParticleFilter.h"

class ParticleFilter {
    
private:
  int height;
  int width;
  std::vector<cv::Point2d> particles;
  std::vector<double> weights;
  double mean;
  double standarddev;
  boost::random::mt19937 rng;
  boost::random::normal_distribution<> nd;
  boost::random::variate_generator<boost::random::mt19937&,
				   boost::random::normal_distribution<> > var_nor;
  double random_number() {
    return ((double) rand() / RAND_MAX );
  }

  double sampleFromGaussian() {
    double number = var_nor();
    return number;
  }
    
  cv::Point2d zeroMeanNoiseFunctionSampler() {
    cv::Point2d point;
    point.x = sampleFromGaussian();
    point.y = sampleFromGaussian();
    return point;
  }
    
  double probabilityOfObservationGivenPoint(cv::Mat observation, cv::Point2d x) {
    double h = (double)observation.at<uchar>(x.y, x.x);
    double value = 0.0078 / 255 * abs(h);
    return value;
  }

  cv::Point2d getNextPoint(cv::Point2d currentParticle) {
    cv::Point2d noise = zeroMeanNoiseFunctionSampler();
    cv::Point2d next;
    next.x = currentParticle.x + noise.x;
    next.y = currentParticle.y + noise.y;
    if(next.x > width) {
      next.x = width - 1;
    }
    if(next.x < 0) {
      next.x = 0;
    }
    if(next.y > height) {
      next.y = height - 1;
    }
    if(next.y < 0) {
      next.y = 0;
    }
    return next;
  }
    
  cv::Point2d sampleUniformly(int width, int height) {
    cv::Point2d point(random_number() * width, random_number() * height);
    return point;
  }
    
  void initializeParticles(int numOfParticles, int width, int height) {
    // sample from initial.. assume initial is uniform...
    for(int i = 0; i < numOfParticles; i++) {
      cv::Point2d particle = sampleUniformly(width, height);
      particles.push_back(particle);
    }
  }
    
  void initializeWeights(int numOfParticles) {
    for(int i = 0; i < numOfParticles; i++) {
      weights.push_back(i/numOfParticles);
    }
  }
    
  void predictionStage() {
    for(int i = 0; i < particles.size(); i++){
      particles[i] = getNextPoint(particles[i]);
    }
  }
    
  void updateWeights(cv::Mat observation) {
    double sum = 0;
    for(int j = 0; j < particles.size(); j++) {
      sum += probabilityOfObservationGivenPoint(observation, particles[j]);
    }
    for(int i = 0; i < particles.size(); i++) {
      double p_obs_given_x = probabilityOfObservationGivenPoint(observation, particles[i]);
      weights[i] = p_obs_given_x / sum;
    }
    double checksum = 0;
    for(int i = 0; i < weights.size(); i++ ){
      checksum += weights[i];
    }
    assert(abs(checksum - 1) < 0.01); 
  }
    
  void resampleFromWeights() {
    assert(particles.size() == weights.size());
    for(int i = 0; i < particles.size(); i++) {
      double sample = random_number();
      double sum = weights[0];
      int counter = 0;
      while(sum < sample && counter < weights.size() - 1) {
	counter ++;
	sum += weights[counter];
      }
      cv::Point2d picked = particles[counter];
      cv::Point2d clone(picked.x, picked.y);
      particles[i] = clone;
    }
  }
    
public:
  void runAlgorithm(cv::Mat observation) {
    predictionStage();
    updateWeights(observation);
    resampleFromWeights();
  }
  ParticleFilter(int numOfParticles, int widthOfImage, int heightOfImage, double standarddev) : nd(0, standarddev), var_nor(rng, nd) {
    initializeParticles(numOfParticles, widthOfImage, heightOfImage);
    initializeWeights(numOfParticles);
    width = widthOfImage;
    height = heightOfImage;
  }
  std::vector<cv::Point2d> getParticles() {
    return particles;
  }
  std::vector<double> getWeights() {
    return weights;
  }
};

int main(int argc, char** argv) {

  cvNamedWindow( "Finding Light", CV_WINDOW_AUTOSIZE );
  CvCapture* capture;

  capture = cvCreateCameraCapture( -1 );
  assert( capture != NULL );

  IplImage* frame;
  frame = cvQueryFrame( capture );
  cv::Mat img = frame;
  ParticleFilter filter(300, img.cols, img.rows, 20);
  std::vector<cv::Point2d> particles;
  while(1){
    frame = cvQueryFrame( capture );
    img = frame;
    if( !frame ) break;
    char c = cvWaitKey(10);
    if ( c == 27 ) break;
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_BGR2GRAY);
    filter.runAlgorithm(gray);
    particles = filter.getParticles();
    for(int i = 0; i < particles.size(); i++){
      cv::circle(img, particles[i], 2, cv::Scalar(255, 255, 255));
    }
    cv::imshow("image", img);
  }
  return 0;
}
