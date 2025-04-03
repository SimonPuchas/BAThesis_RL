# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "open_manipulator_msgs: 2 messages, 2 services")

set(MSG_I_FLAGS "-Iopen_manipulator_msgs:/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(open_manipulator_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" NAME_WE)
add_custom_target(_open_manipulator_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "open_manipulator_msgs" "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" ""
)

get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" NAME_WE)
add_custom_target(_open_manipulator_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "open_manipulator_msgs" "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" "geometry_msgs/Point:geometry_msgs/Quaternion:geometry_msgs/Pose"
)

get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" NAME_WE)
add_custom_target(_open_manipulator_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "open_manipulator_msgs" "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" "open_manipulator_msgs/JointPose"
)

get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" NAME_WE)
add_custom_target(_open_manipulator_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "open_manipulator_msgs" "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" "geometry_msgs/Point:geometry_msgs/Quaternion:open_manipulator_msgs/KinematicsPose:geometry_msgs/Pose"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs
)
_generate_msg_cpp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Services
_generate_srv_cpp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv"
  "${MSG_I_FLAGS}"
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs
)
_generate_srv_cpp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Module File
_generate_module_cpp(open_manipulator_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(open_manipulator_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(open_manipulator_msgs_generate_messages open_manipulator_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_cpp _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_cpp _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_cpp _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_cpp _open_manipulator_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(open_manipulator_msgs_gencpp)
add_dependencies(open_manipulator_msgs_gencpp open_manipulator_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS open_manipulator_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs
)
_generate_msg_eus(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Services
_generate_srv_eus(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv"
  "${MSG_I_FLAGS}"
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs
)
_generate_srv_eus(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Module File
_generate_module_eus(open_manipulator_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(open_manipulator_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(open_manipulator_msgs_generate_messages open_manipulator_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_eus _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_eus _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_eus _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_eus _open_manipulator_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(open_manipulator_msgs_geneus)
add_dependencies(open_manipulator_msgs_geneus open_manipulator_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS open_manipulator_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs
)
_generate_msg_lisp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Services
_generate_srv_lisp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv"
  "${MSG_I_FLAGS}"
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs
)
_generate_srv_lisp(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Module File
_generate_module_lisp(open_manipulator_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(open_manipulator_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(open_manipulator_msgs_generate_messages open_manipulator_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_lisp _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_lisp _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_lisp _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_lisp _open_manipulator_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(open_manipulator_msgs_genlisp)
add_dependencies(open_manipulator_msgs_genlisp open_manipulator_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS open_manipulator_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs
)
_generate_msg_nodejs(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Services
_generate_srv_nodejs(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv"
  "${MSG_I_FLAGS}"
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs
)
_generate_srv_nodejs(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Module File
_generate_module_nodejs(open_manipulator_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(open_manipulator_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(open_manipulator_msgs_generate_messages open_manipulator_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_nodejs _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_nodejs _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_nodejs _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_nodejs _open_manipulator_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(open_manipulator_msgs_gennodejs)
add_dependencies(open_manipulator_msgs_gennodejs open_manipulator_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS open_manipulator_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs
)
_generate_msg_py(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Services
_generate_srv_py(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv"
  "${MSG_I_FLAGS}"
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs
)
_generate_srv_py(open_manipulator_msgs
  "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs
)

### Generating Module File
_generate_module_py(open_manipulator_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(open_manipulator_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(open_manipulator_msgs_generate_messages open_manipulator_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/JointPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_py _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/msg/KinematicsPose.msg" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_py _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetJointPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_py _open_manipulator_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/simon/PathFinderRL/thesis_ws/src/turtlebot3/open_manipulator/open_manipulator_msgs/srv/GetKinematicsPose.srv" NAME_WE)
add_dependencies(open_manipulator_msgs_generate_messages_py _open_manipulator_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(open_manipulator_msgs_genpy)
add_dependencies(open_manipulator_msgs_genpy open_manipulator_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS open_manipulator_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/open_manipulator_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(open_manipulator_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(open_manipulator_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/open_manipulator_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(open_manipulator_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(open_manipulator_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/open_manipulator_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(open_manipulator_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(open_manipulator_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/open_manipulator_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(open_manipulator_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(open_manipulator_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/open_manipulator_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(open_manipulator_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(open_manipulator_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
