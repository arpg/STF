#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <calibu/cam/camera_crtp.h>
#include <calibu/cam/camera_models_crtp.h>
#include <calibu/cam/camera_crtp_interop.h>

#include <calibu/cam/rectify_crtp.h>

#include <HAL/Camera/CameraDevice.h>

#include "tags.h"

namespace Eigen
{
  typedef Matrix<double,6,1>  Vector6d;
}

using namespace std;

DEFINE_string(cam,
    "-cam",
    "hal camera specifier.");

DEFINE_string(cmod,
    "-cmod",
    "calibu camera model xml file.");

DEFINE_string( map,
    "-map",
    "Survey map file.");

#define USAGE\
      "This program will process images with visible AR tags and produce\n"\
      "a text file with the following format:\n"\
      "-----------------------------------------------------\n"\
      " m, the number of poses that see any landmarks\n"\
      " n, the number of unique landmarks seen\n"\
      " k, number of measurements\n"\
      " pose_id, landmark_id, u, v\n"\
      "   .\n"\
      "   .\n"\
      "   .\n"\
      " pose_id, landmark_id, u, v\n"\
      " pose_1\n"\
      " pose_2\n"\
      "   .\n"\
      "   .\n"\
      "   .\n"\
      " pose_m\n"\
      " landmark_1\n"\
      " landmark_2\n"\
      "   .\n"\
      "   .\n"\
      "   .\n"\
      " landmark_n\n"\
      "-----------------------------------------------------\n"\
      "This file is suitable as an initialization for ground-truth pose\n"\
      "estimation. NB:\n"\
      "  each pose is a 1x6 vector:  x,y,z,p,q,r\n"\
      "  each landmark is 1x4 vector:  id,x,y,z\n"\
      "\n\n"

/////////////////////////////////////////////////////////////////////////
struct measurement_t
{
  int pose_idx;
  int lm_idx;
  double u;
  double v;
};

/////////////////////////////////////////////////////////////////////////
void ParseCameraUriOrDieComplaining( const std::string& sUri, hal::Camera& cam )
{
  try{
    cam = hal::Camera( hal::Uri(sUri) );
  }
  catch( hal::DeviceException e ){
    printf("Error parsing camera URI: '%s' -- %s\n", sUri.c_str(), e.what() );
    printf("Perhaps you meant something like one of these:\n");
    printf("    rectify:[file=cameras.xml]//deinterlace://uvc://\n");
    printf("    file:[loop=1]//~/Data/CityBlock-Noisy/[left*,right*].pgm\n" );
    printf("    trailmix:[narrow=0,depth=0]//file:[startframe=30]//~/Data/stairwell/superframes/[*.pgm]\n");
    exit(-1);
  }
}

/////////////////////////////////////////////////////////////////////////
void ParseSurveyMapFile( 
    const std::string& filename,
    std::map<int,Eigen::Vector3d>& survey_map
    )
{
  std::ifstream ifs( filename );
  std::string line;
  while( ifs.good() ){
    std::getline ( ifs, line );
    int uid; // uniquely encodes tag id and landmark id
    double x, y, z;
    sscanf( line.c_str(), "%d, %lf, %lf, %lf", &uid, &x, &y, &z );
    survey_map[uid] = Eigen::Vector3d( x, y, z );

    // first two digits are tag id, secnod two are landmark id:
    int lmid = (uid%10) + ((uid-uid%10) % 100);
    int tagid = tagid / 100;
//    printf(" got tag %d, lm %d, at %f, %f, %f\n", tagid, lmid, x, y, z );
  }
}

/////////////////////////////////////////////////////////////////////////
Eigen::Vector6d CalcPose(
    const double pts_2d[4][2],
    const Eigen::Vector3d pts_3d[4],
    const Eigen::Matrix3d& K
    )
{
  std::vector<cv::Point3f> cv_obj;
  std::vector<cv::Point2f> cv_img;

  cv_obj.push_back( cv::Point3f(pts_3d[0][0],pts_3d[0][1],pts_3d[0][2]) );
  cv_obj.push_back( cv::Point3f(pts_3d[1][0],pts_3d[1][1],pts_3d[1][2]) );
  cv_obj.push_back( cv::Point3f(pts_3d[2][0],pts_3d[2][1],pts_3d[2][2]) );
  cv_obj.push_back( cv::Point3f(pts_3d[3][0],pts_3d[3][1],pts_3d[3][2]) );

  cv_img.push_back( cv::Point2f(pts_2d[0][0],pts_2d[0][1]) );
  cv_img.push_back( cv::Point2f(pts_2d[1][0],pts_2d[1][1]) );
  cv_img.push_back( cv::Point2f(pts_2d[2][0],pts_2d[2][1]) );
  cv_img.push_back( cv::Point2f(pts_2d[3][0],pts_2d[3][1]) );

  cv::Mat cv_K(3,3,CV_64F);
  cv_K.at<double>(0,0) = K(0,0);
  cv_K.at<double>(1,0) = K(1,0);
  cv_K.at<double>(2,0) = K(2,0);
  cv_K.at<double>(0,1) = K(0,1);
  cv_K.at<double>(1,1) = K(1,1);
  cv_K.at<double>(2,1) = K(2,1);
  cv_K.at<double>(0,2) = K(0,2);
  cv_K.at<double>(1,2) = K(1,2);
  cv_K.at<double>(2,2) = K(2,2);

  cv::Mat cv_coeff;
  cv::Mat cv_rot(3,1,CV_64F);
  cv::Mat cv_trans(3,1,CV_64F);

  cv::solvePnP( cv_obj, cv_img, cv_K, cv_coeff, cv_rot, cv_trans, false );

  Eigen::Vector6d pose;
  pose[0] = cv_trans.at<double>(0); 
  pose[1] = cv_trans.at<double>(1); 
  pose[2] = cv_trans.at<double>(2); 
  pose[3] = cv_rot.at<double>(0);
  pose[4] = cv_rot.at<double>(1);
  pose[5] = cv_rot.at<double>(2);

  return pose;
}

/////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
  if( argc <= 2 ){
    puts(USAGE);
    return -1;
  }
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  hal::Camera cam;
  ParseCameraUriOrDieComplaining( FLAGS_cam, cam );

  std::map<int,Eigen::Vector3d> survey_map;
  ParseSurveyMapFile( FLAGS_map, survey_map );

  // TODO condense this LoadRig into LoadCamera
  calibu::Rig<double> rig;
  calibu::LoadRig( FLAGS_cmod, &rig );
  calibu::CameraInterface<double> *cmod = rig.cameras_[0];
  Eigen::Matrix3d K;
  double* p = cmod->GetParams(); 
  K << p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1;

  // TODO add AttachLUT funtionality to camera models?
  calibu::LookupTable lut;
  calibu::CreateLookupTable( *cmod, lut );
  assert( cmod->Width() == cam.Width() && cmod->Height() == cam.Height() );
  cv::Mat rect( cam.Height(), cam.Width(), CV_8UC1 ); // rectified image
  cv::Mat rgb;

  // get a tag detector
  TagDetector td;

  cv::namedWindow( "Tag Viewer", CV_WINDOW_AUTOSIZE );

  std::vector<cv::Mat> vImages;

  int local_pose_id = 0;
  int local_landmark_id = 0;
  std::map<int,int> local_to_survey; // map from unique id to local 0-based id
  std::map<int,int> survey_to_local; // map from unique id to local 0-based id
  std::map<int,bool> tag_seen; // tags_seen[id] is true if tag id has been seen 
  std::vector<measurement_t> measurements;
  std::vector<Eigen::Vector6d> poses;

  for( int ii = 0; ii<40; ii++ ){

    // 1) Capture and rectify
    if( !cam.Capture( vImages ) ){
      printf("Finished after image %d\n", ii );
      return 0;
    }
    calibu::Rectify( lut, vImages[0].data, rect.data, rect.cols, rect.rows );
    cvtColor( rect, rgb, CV_GRAY2RGB );

    // 2) Run tag detector and get tag corners
    std::vector<april_tag_detection_t> vDetections;
    td.Detect( rect, vDetections );
    if( vDetections.empty() ){
      continue;
    }


    // 3) estimate rough pose from this first seen target
    Eigen::Vector3d pts_3d[4];
    for( int cidx = 0; cidx < 4; cidx++ ){
      pts_3d[cidx] = survey_map[ vDetections[0].id*100+cidx ];
    }
    poses.push_back( CalcPose( vDetections[0].p, pts_3d, K ) );

    // 4) Extract measurements of the 4 corners of each detected target
    for( size_t ii = 0; ii < vDetections.size(); ii++ ){
      april_tag_detection_t* p = &vDetections[ii];

      // target not seen before? then add 4 new landmarks
      if( tag_seen.find(p->id) == tag_seen.end() ){
        // ok, remap unique survey landmark id to new 0-based index
        for( int cidx = 0; cidx < 4; cidx++ ){
          int survey_id = p->id*100+cidx;
          local_to_survey[ local_landmark_id ] = survey_id;
          survey_to_local[ survey_id ] = local_landmark_id;
          local_landmark_id++;
        }
        tag_seen[p->id] = true;
      }

      // add 4 measurements for this tag's corners
      for( int cidx = 0; cidx < 4; cidx++ ){
        int survey_id = p->id*100+cidx;
        int lid = survey_to_local[ survey_id ]; // look up local id
        measurement_t z = { local_pose_id, lid, p->p[cidx][0], p->p[cidx][1] };
        measurements.push_back( z );
      }

      // draw rectangle around tag 
      cv::line( rgb,
          cv::Point( p->p[0][0], p->p[0][1] ),
          cv::Point( p->p[1][0], p->p[1][1] ),
          cv::Scalar( 255, 0, 0 ), 1, 8 );

      cv::line( rgb,
          cv::Point( p->p[1][0], p->p[1][1] ),
          cv::Point( p->p[2][0], p->p[2][1] ),
          cv::Scalar( 0, 255, 0 ), 1, 8 );

      cv::line( rgb,
          cv::Point( p->p[2][0], p->p[2][1] ),
          cv::Point( p->p[3][0], p->p[3][1] ),
          cv::Scalar( 0, 0, 255 ), 1, 8 );

      cv::line( rgb,
          cv::Point( p->p[3][0], p->p[3][1] ),
          cv::Point( p->p[0][0], p->p[0][1] ),
          cv::Scalar( 255, 0, 255 ), 1, 8 );
    }

    cv::imshow( "Tag Viewer", rgb );
//    cv::waitKey(0);
    local_pose_id++;
  }

  // ok, now print everything:
  printf( "%d\n", local_pose_id ); 
  printf( "%lu\n", local_to_survey.size() ); 
  printf( "%lu\n", measurements.size() ); 
  for( size_t ii = 0; ii < measurements.size(); ii++ ){
    measurement_t& z = measurements[ii];
    printf("%d, %d, %f, %f\n", z.pose_idx, z.lm_idx, z.u, z.v ); 
  }
  for( size_t ii = 0; ii < poses.size(); ii++ ){
    printf( "%f, %f, %f, %f, %f, %f\n", poses[ii][0],  poses[ii][1],
        poses[ii][2],  poses[ii][3], poses[ii][4],  poses[ii][5] );
  }
  for( size_t ii = 0; ii < local_to_survey.size(); ii++ ){
     int sid = local_to_survey[ii];
     Eigen::Vector3d& p3d = survey_map[sid];
     printf("%d, %f, %f, %f\n", sid, p3d[0], p3d[1], p3d[2] );
  }

  return 0;
}

