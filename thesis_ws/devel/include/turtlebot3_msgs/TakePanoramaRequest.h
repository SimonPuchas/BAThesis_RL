// Generated by gencpp from file turtlebot3_msgs/TakePanoramaRequest.msg
// DO NOT EDIT!


#ifndef TURTLEBOT3_MSGS_MESSAGE_TAKEPANORAMAREQUEST_H
#define TURTLEBOT3_MSGS_MESSAGE_TAKEPANORAMAREQUEST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace turtlebot3_msgs
{
template <class ContainerAllocator>
struct TakePanoramaRequest_
{
  typedef TakePanoramaRequest_<ContainerAllocator> Type;

  TakePanoramaRequest_()
    : mode(0)
    , pano_angle(0.0)
    , snap_interval(0.0)
    , rot_vel(0.0)  {
    }
  TakePanoramaRequest_(const ContainerAllocator& _alloc)
    : mode(0)
    , pano_angle(0.0)
    , snap_interval(0.0)
    , rot_vel(0.0)  {
  (void)_alloc;
    }



   typedef uint8_t _mode_type;
  _mode_type mode;

   typedef float _pano_angle_type;
  _pano_angle_type pano_angle;

   typedef float _snap_interval_type;
  _snap_interval_type snap_interval;

   typedef float _rot_vel_type;
  _rot_vel_type rot_vel;



// reducing the odds to have name collisions with Windows.h 
#if defined(_WIN32) && defined(SNAPANDROTATE)
  #undef SNAPANDROTATE
#endif
#if defined(_WIN32) && defined(CONTINUOUS)
  #undef CONTINUOUS
#endif
#if defined(_WIN32) && defined(STOP)
  #undef STOP
#endif

  enum {
    SNAPANDROTATE = 0u,
    CONTINUOUS = 1u,
    STOP = 2u,
  };


  typedef boost::shared_ptr< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> const> ConstPtr;

}; // struct TakePanoramaRequest_

typedef ::turtlebot3_msgs::TakePanoramaRequest_<std::allocator<void> > TakePanoramaRequest;

typedef boost::shared_ptr< ::turtlebot3_msgs::TakePanoramaRequest > TakePanoramaRequestPtr;
typedef boost::shared_ptr< ::turtlebot3_msgs::TakePanoramaRequest const> TakePanoramaRequestConstPtr;

// constants requiring out of line definition

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator1> & lhs, const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator2> & rhs)
{
  return lhs.mode == rhs.mode &&
    lhs.pano_angle == rhs.pano_angle &&
    lhs.snap_interval == rhs.snap_interval &&
    lhs.rot_vel == rhs.rot_vel;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator1> & lhs, const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace turtlebot3_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "f52c694c82704221735cc576c7587ecc";
  }

  static const char* value(const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xf52c694c82704221ULL;
  static const uint64_t static_value2 = 0x735cc576c7587eccULL;
};

template<class ContainerAllocator>
struct DataType< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "turtlebot3_msgs/TakePanoramaRequest";
  }

  static const char* value(const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# mode for taking the pictures\n"
"uint8 mode\n"
"# rotate, stop, snapshot, rotate, stop, snapshot, ...\n"
"uint8 SNAPANDROTATE=0\n"
"# keep rotating while taking snapshots\n"
"uint8 CONTINUOUS=1\n"
"# stop an ongoing panorama creation\n"
"uint8 STOP=2\n"
"# total angle of panorama picture\n"
"float32 pano_angle\n"
"# angle interval when creating the panorama picture in snap&rotate mode, time interval otherwise \n"
"float32 snap_interval\n"
"# rotating velocity\n"
"float32 rot_vel\n"
"\n"
;
  }

  static const char* value(const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.mode);
      stream.next(m.pano_angle);
      stream.next(m.snap_interval);
      stream.next(m.rot_vel);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TakePanoramaRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::turtlebot3_msgs::TakePanoramaRequest_<ContainerAllocator>& v)
  {
    s << indent << "mode: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.mode);
    s << indent << "pano_angle: ";
    Printer<float>::stream(s, indent + "  ", v.pano_angle);
    s << indent << "snap_interval: ";
    Printer<float>::stream(s, indent + "  ", v.snap_interval);
    s << indent << "rot_vel: ";
    Printer<float>::stream(s, indent + "  ", v.rot_vel);
  }
};

} // namespace message_operations
} // namespace ros

#endif // TURTLEBOT3_MSGS_MESSAGE_TAKEPANORAMAREQUEST_H
