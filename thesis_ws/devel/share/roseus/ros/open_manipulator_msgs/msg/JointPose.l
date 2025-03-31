;; Auto-generated. Do not edit!


(when (boundp 'open_manipulator_msgs::JointPose)
  (if (not (find-package "OPEN_MANIPULATOR_MSGS"))
    (make-package "OPEN_MANIPULATOR_MSGS"))
  (shadow 'JointPose (find-package "OPEN_MANIPULATOR_MSGS")))
(unless (find-package "OPEN_MANIPULATOR_MSGS::JOINTPOSE")
  (make-package "OPEN_MANIPULATOR_MSGS::JOINTPOSE"))

(in-package "ROS")
;;//! \htmlinclude JointPose.msg.html


(defclass open_manipulator_msgs::JointPose
  :super ros::object
  :slots (_joint_name _position _move_time ))

(defmethod open_manipulator_msgs::JointPose
  (:init
   (&key
    ((:joint_name __joint_name) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:position __position) (make-array 0 :initial-element 0.0 :element-type :float))
    ((:move_time __move_time) 0.0)
    )
   (send-super :init)
   (setq _joint_name __joint_name)
   (setq _position __position)
   (setq _move_time (float __move_time))
   self)
  (:joint_name
   (&optional __joint_name)
   (if __joint_name (setq _joint_name __joint_name)) _joint_name)
  (:position
   (&optional __position)
   (if __position (setq _position __position)) _position)
  (:move_time
   (&optional __move_time)
   (if __move_time (setq _move_time __move_time)) _move_time)
  (:serialization-length
   ()
   (+
    ;; string[] _joint_name
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _joint_name)) 4
    ;; float64[] _position
    (* 8    (length _position)) 4
    ;; float64 _move_time
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _joint_name
     (write-long (length _joint_name) s)
     (dolist (elem _joint_name)
       (write-long (length elem) s) (princ elem s)
       )
     ;; float64[] _position
     (write-long (length _position) s)
     (dotimes (i (length _position))
       (sys::poke (elt _position i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64 _move_time
       (sys::poke _move_time (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _joint_name
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _joint_name (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _joint_name i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; float64[] _position
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _position (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _position i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;; float64 _move_time
     (setq _move_time (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get open_manipulator_msgs::JointPose :md5sum-) "217de178e1835082dfab23527f9a9de9")
(setf (get open_manipulator_msgs::JointPose :datatype-) "open_manipulator_msgs/JointPose")
(setf (get open_manipulator_msgs::JointPose :definition-)
      "string[]   joint_name
float64[]  position
float64    move_time

")



(provide :open_manipulator_msgs/JointPose "217de178e1835082dfab23527f9a9de9")


