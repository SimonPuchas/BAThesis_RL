; Auto-generated. Do not edit!


(cl:in-package open_manipulator_msgs-srv)


;//! \htmlinclude GetKinematicsPose-request.msg.html

(cl:defclass <GetKinematicsPose-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GetKinematicsPose-request (<GetKinematicsPose-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetKinematicsPose-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetKinematicsPose-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name open_manipulator_msgs-srv:<GetKinematicsPose-request> is deprecated: use open_manipulator_msgs-srv:GetKinematicsPose-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetKinematicsPose-request>) ostream)
  "Serializes a message object of type '<GetKinematicsPose-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetKinematicsPose-request>) istream)
  "Deserializes a message object of type '<GetKinematicsPose-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetKinematicsPose-request>)))
  "Returns string type for a service object of type '<GetKinematicsPose-request>"
  "open_manipulator_msgs/GetKinematicsPoseRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetKinematicsPose-request)))
  "Returns string type for a service object of type 'GetKinematicsPose-request"
  "open_manipulator_msgs/GetKinematicsPoseRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetKinematicsPose-request>)))
  "Returns md5sum for a message object of type '<GetKinematicsPose-request>"
  "c8178071274fe9988bb6f690a77025ca")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetKinematicsPose-request)))
  "Returns md5sum for a message object of type 'GetKinematicsPose-request"
  "c8178071274fe9988bb6f690a77025ca")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetKinematicsPose-request>)))
  "Returns full string definition for message of type '<GetKinematicsPose-request>"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetKinematicsPose-request)))
  "Returns full string definition for message of type 'GetKinematicsPose-request"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetKinematicsPose-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetKinematicsPose-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetKinematicsPose-request
))
;//! \htmlinclude GetKinematicsPose-response.msg.html

(cl:defclass <GetKinematicsPose-response> (roslisp-msg-protocol:ros-message)
  ((position_ctrl_kinematics_pose
    :reader position_ctrl_kinematics_pose
    :initarg :position_ctrl_kinematics_pose
    :type open_manipulator_msgs-msg:KinematicsPose
    :initform (cl:make-instance 'open_manipulator_msgs-msg:KinematicsPose)))
)

(cl:defclass GetKinematicsPose-response (<GetKinematicsPose-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetKinematicsPose-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetKinematicsPose-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name open_manipulator_msgs-srv:<GetKinematicsPose-response> is deprecated: use open_manipulator_msgs-srv:GetKinematicsPose-response instead.")))

(cl:ensure-generic-function 'position_ctrl_kinematics_pose-val :lambda-list '(m))
(cl:defmethod position_ctrl_kinematics_pose-val ((m <GetKinematicsPose-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader open_manipulator_msgs-srv:position_ctrl_kinematics_pose-val is deprecated.  Use open_manipulator_msgs-srv:position_ctrl_kinematics_pose instead.")
  (position_ctrl_kinematics_pose m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetKinematicsPose-response>) ostream)
  "Serializes a message object of type '<GetKinematicsPose-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'position_ctrl_kinematics_pose) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetKinematicsPose-response>) istream)
  "Deserializes a message object of type '<GetKinematicsPose-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'position_ctrl_kinematics_pose) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetKinematicsPose-response>)))
  "Returns string type for a service object of type '<GetKinematicsPose-response>"
  "open_manipulator_msgs/GetKinematicsPoseResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetKinematicsPose-response)))
  "Returns string type for a service object of type 'GetKinematicsPose-response"
  "open_manipulator_msgs/GetKinematicsPoseResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetKinematicsPose-response>)))
  "Returns md5sum for a message object of type '<GetKinematicsPose-response>"
  "c8178071274fe9988bb6f690a77025ca")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetKinematicsPose-response)))
  "Returns md5sum for a message object of type 'GetKinematicsPose-response"
  "c8178071274fe9988bb6f690a77025ca")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetKinematicsPose-response>)))
  "Returns full string definition for message of type '<GetKinematicsPose-response>"
  (cl:format cl:nil "KinematicsPose position_ctrl_kinematics_pose~%~%~%================================================================================~%MSG: open_manipulator_msgs/KinematicsPose~%string    	        group_name~%geometry_msgs/Pose  pose~%float64             move_time~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetKinematicsPose-response)))
  "Returns full string definition for message of type 'GetKinematicsPose-response"
  (cl:format cl:nil "KinematicsPose position_ctrl_kinematics_pose~%~%~%================================================================================~%MSG: open_manipulator_msgs/KinematicsPose~%string    	        group_name~%geometry_msgs/Pose  pose~%float64             move_time~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetKinematicsPose-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'position_ctrl_kinematics_pose))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetKinematicsPose-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetKinematicsPose-response
    (cl:cons ':position_ctrl_kinematics_pose (position_ctrl_kinematics_pose msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetKinematicsPose)))
  'GetKinematicsPose-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetKinematicsPose)))
  'GetKinematicsPose-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetKinematicsPose)))
  "Returns string type for a service object of type '<GetKinematicsPose>"
  "open_manipulator_msgs/GetKinematicsPose")