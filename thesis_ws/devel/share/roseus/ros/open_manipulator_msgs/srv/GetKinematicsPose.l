;; Auto-generated. Do not edit!


(when (boundp 'open_manipulator_msgs::GetKinematicsPose)
  (if (not (find-package "OPEN_MANIPULATOR_MSGS"))
    (make-package "OPEN_MANIPULATOR_MSGS"))
  (shadow 'GetKinematicsPose (find-package "OPEN_MANIPULATOR_MSGS")))
(unless (find-package "OPEN_MANIPULATOR_MSGS::GETKINEMATICSPOSE")
  (make-package "OPEN_MANIPULATOR_MSGS::GETKINEMATICSPOSE"))
(unless (find-package "OPEN_MANIPULATOR_MSGS::GETKINEMATICSPOSEREQUEST")
  (make-package "OPEN_MANIPULATOR_MSGS::GETKINEMATICSPOSEREQUEST"))
(unless (find-package "OPEN_MANIPULATOR_MSGS::GETKINEMATICSPOSERESPONSE")
  (make-package "OPEN_MANIPULATOR_MSGS::GETKINEMATICSPOSERESPONSE"))

(in-package "ROS")





(defclass open_manipulator_msgs::GetKinematicsPoseRequest
  :super ros::object
  :slots ())

(defmethod open_manipulator_msgs::GetKinematicsPoseRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass open_manipulator_msgs::GetKinematicsPoseResponse
  :super ros::object
  :slots (_position_ctrl_kinematics_pose ))

(defmethod open_manipulator_msgs::GetKinematicsPoseResponse
  (:init
   (&key
    ((:position_ctrl_kinematics_pose __position_ctrl_kinematics_pose) (instance open_manipulator_msgs::KinematicsPose :init))
    )
   (send-super :init)
   (setq _position_ctrl_kinematics_pose __position_ctrl_kinematics_pose)
   self)
  (:position_ctrl_kinematics_pose
   (&rest __position_ctrl_kinematics_pose)
   (if (keywordp (car __position_ctrl_kinematics_pose))
       (send* _position_ctrl_kinematics_pose __position_ctrl_kinematics_pose)
     (progn
       (if __position_ctrl_kinematics_pose (setq _position_ctrl_kinematics_pose (car __position_ctrl_kinematics_pose)))
       _position_ctrl_kinematics_pose)))
  (:serialization-length
   ()
   (+
    ;; open_manipulator_msgs/KinematicsPose _position_ctrl_kinematics_pose
    (send _position_ctrl_kinematics_pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; open_manipulator_msgs/KinematicsPose _position_ctrl_kinematics_pose
       (send _position_ctrl_kinematics_pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; open_manipulator_msgs/KinematicsPose _position_ctrl_kinematics_pose
     (send _position_ctrl_kinematics_pose :deserialize buf ptr-) (incf ptr- (send _position_ctrl_kinematics_pose :serialization-length))
   ;;
   self)
  )

(defclass open_manipulator_msgs::GetKinematicsPose
  :super ros::object
  :slots ())

(setf (get open_manipulator_msgs::GetKinematicsPose :md5sum-) "c8178071274fe9988bb6f690a77025ca")
(setf (get open_manipulator_msgs::GetKinematicsPose :datatype-) "open_manipulator_msgs/GetKinematicsPose")
(setf (get open_manipulator_msgs::GetKinematicsPose :request) open_manipulator_msgs::GetKinematicsPoseRequest)
(setf (get open_manipulator_msgs::GetKinematicsPose :response) open_manipulator_msgs::GetKinematicsPoseResponse)

(defmethod open_manipulator_msgs::GetKinematicsPoseRequest
  (:response () (instance open_manipulator_msgs::GetKinematicsPoseResponse :init)))

(setf (get open_manipulator_msgs::GetKinematicsPoseRequest :md5sum-) "c8178071274fe9988bb6f690a77025ca")
(setf (get open_manipulator_msgs::GetKinematicsPoseRequest :datatype-) "open_manipulator_msgs/GetKinematicsPoseRequest")
(setf (get open_manipulator_msgs::GetKinematicsPoseRequest :definition-)
      "
---
KinematicsPose position_ctrl_kinematics_pose


================================================================================
MSG: open_manipulator_msgs/KinematicsPose
string    	        group_name
geometry_msgs/Pose  pose
float64             move_time

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")

(setf (get open_manipulator_msgs::GetKinematicsPoseResponse :md5sum-) "c8178071274fe9988bb6f690a77025ca")
(setf (get open_manipulator_msgs::GetKinematicsPoseResponse :datatype-) "open_manipulator_msgs/GetKinematicsPoseResponse")
(setf (get open_manipulator_msgs::GetKinematicsPoseResponse :definition-)
      "
---
KinematicsPose position_ctrl_kinematics_pose


================================================================================
MSG: open_manipulator_msgs/KinematicsPose
string    	        group_name
geometry_msgs/Pose  pose
float64             move_time

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
")



(provide :open_manipulator_msgs/GetKinematicsPose "c8178071274fe9988bb6f690a77025ca")


