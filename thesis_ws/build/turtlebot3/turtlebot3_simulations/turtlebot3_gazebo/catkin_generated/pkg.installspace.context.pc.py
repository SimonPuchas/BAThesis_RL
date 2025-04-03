# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "geometry_msgs;nav_msgs;roscpp;sensor_msgs;std_msgs;tf;gazebo_ros".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lturtlebot3_gazebo".split(';') if "-lturtlebot3_gazebo" != "" else []
PROJECT_NAME = "turtlebot3_gazebo"
PROJECT_SPACE_DIR = "/home/simon/PathFinderRL/thesis_ws/install"
PROJECT_VERSION = "0.1.7"
