;; Auto-generated. Do not edit!


(when (boundp 'open_manipulator_msgs::GetJointPose)
  (if (not (find-package "OPEN_MANIPULATOR_MSGS"))
    (make-package "OPEN_MANIPULATOR_MSGS"))
  (shadow 'GetJointPose (find-package "OPEN_MANIPULATOR_MSGS")))
(unless (find-package "OPEN_MANIPULATOR_MSGS::GETJOINTPOSE")
  (make-package "OPEN_MANIPULATOR_MSGS::GETJOINTPOSE"))
(unless (find-package "OPEN_MANIPULATOR_MSGS::GETJOINTPOSEREQUEST")
  (make-package "OPEN_MANIPULATOR_MSGS::GETJOINTPOSEREQUEST"))
(unless (find-package "OPEN_MANIPULATOR_MSGS::GETJOINTPOSERESPONSE")
  (make-package "OPEN_MANIPULATOR_MSGS::GETJOINTPOSERESPONSE"))

(in-package "ROS")





(defclass open_manipulator_msgs::GetJointPoseRequest
  :super ros::object
  :slots ())

(defmethod open_manipulator_msgs::GetJointPoseRequest
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

(defclass open_manipulator_msgs::GetJointPoseResponse
  :super ros::object
  :slots (_position_ctrl_joint_pose ))

(defmethod open_manipulator_msgs::GetJointPoseResponse
  (:init
   (&key
    ((:position_ctrl_joint_pose __position_ctrl_joint_pose) (instance open_manipulator_msgs::JointPose :init))
    )
   (send-super :init)
   (setq _position_ctrl_joint_pose __position_ctrl_joint_pose)
   self)
  (:position_ctrl_joint_pose
   (&rest __position_ctrl_joint_pose)
   (if (keywordp (car __position_ctrl_joint_pose))
       (send* _position_ctrl_joint_pose __position_ctrl_joint_pose)
     (progn
       (if __position_ctrl_joint_pose (setq _position_ctrl_joint_pose (car __position_ctrl_joint_pose)))
       _position_ctrl_joint_pose)))
  (:serialization-length
   ()
   (+
    ;; open_manipulator_msgs/JointPose _position_ctrl_joint_pose
    (send _position_ctrl_joint_pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; open_manipulator_msgs/JointPose _position_ctrl_joint_pose
       (send _position_ctrl_joint_pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; open_manipulator_msgs/JointPose _position_ctrl_joint_pose
     (send _position_ctrl_joint_pose :deserialize buf ptr-) (incf ptr- (send _position_ctrl_joint_pose :serialization-length))
   ;;
   self)
  )

(defclass open_manipulator_msgs::GetJointPose
  :super ros::object
  :slots ())

(setf (get open_manipulator_msgs::GetJointPose :md5sum-) "d90103d910754a636e20bdfefafe3642")
(setf (get open_manipulator_msgs::GetJointPose :datatype-) "open_manipulator_msgs/GetJointPose")
(setf (get open_manipulator_msgs::GetJointPose :request) open_manipulator_msgs::GetJointPoseRequest)
(setf (get open_manipulator_msgs::GetJointPose :response) open_manipulator_msgs::GetJointPoseResponse)

(defmethod open_manipulator_msgs::GetJointPoseRequest
  (:response () (instance open_manipulator_msgs::GetJointPoseResponse :init)))

(setf (get open_manipulator_msgs::GetJointPoseRequest :md5sum-) "d90103d910754a636e20bdfefafe3642")
(setf (get open_manipulator_msgs::GetJointPoseRequest :datatype-) "open_manipulator_msgs/GetJointPoseRequest")
(setf (get open_manipulator_msgs::GetJointPoseRequest :definition-)
      "
---
JointPose position_ctrl_joint_pose


================================================================================
MSG: open_manipulator_msgs/JointPose
string[]   joint_name
float64[]  position
float64    move_time
")

(setf (get open_manipulator_msgs::GetJointPoseResponse :md5sum-) "d90103d910754a636e20bdfefafe3642")
(setf (get open_manipulator_msgs::GetJointPoseResponse :datatype-) "open_manipulator_msgs/GetJointPoseResponse")
(setf (get open_manipulator_msgs::GetJointPoseResponse :definition-)
      "
---
JointPose position_ctrl_joint_pose


================================================================================
MSG: open_manipulator_msgs/JointPose
string[]   joint_name
float64[]  position
float64    move_time
")



(provide :open_manipulator_msgs/GetJointPose "d90103d910754a636e20bdfefafe3642")


