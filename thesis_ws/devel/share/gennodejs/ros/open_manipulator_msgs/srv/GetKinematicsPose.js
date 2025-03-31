// Auto-generated. Do not edit!

// (in-package open_manipulator_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let KinematicsPose = require('../msg/KinematicsPose.js');

//-----------------------------------------------------------

class GetKinematicsPoseRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetKinematicsPoseRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetKinematicsPoseRequest
    let len;
    let data = new GetKinematicsPoseRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'open_manipulator_msgs/GetKinematicsPoseRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetKinematicsPoseRequest(null);
    return resolved;
    }
};

class GetKinematicsPoseResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.position_ctrl_kinematics_pose = null;
    }
    else {
      if (initObj.hasOwnProperty('position_ctrl_kinematics_pose')) {
        this.position_ctrl_kinematics_pose = initObj.position_ctrl_kinematics_pose
      }
      else {
        this.position_ctrl_kinematics_pose = new KinematicsPose();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetKinematicsPoseResponse
    // Serialize message field [position_ctrl_kinematics_pose]
    bufferOffset = KinematicsPose.serialize(obj.position_ctrl_kinematics_pose, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetKinematicsPoseResponse
    let len;
    let data = new GetKinematicsPoseResponse(null);
    // Deserialize message field [position_ctrl_kinematics_pose]
    data.position_ctrl_kinematics_pose = KinematicsPose.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += KinematicsPose.getMessageSize(object.position_ctrl_kinematics_pose);
    return length;
  }

  static datatype() {
    // Returns string type for a service object
    return 'open_manipulator_msgs/GetKinematicsPoseResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c8178071274fe9988bb6f690a77025ca';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetKinematicsPoseResponse(null);
    if (msg.position_ctrl_kinematics_pose !== undefined) {
      resolved.position_ctrl_kinematics_pose = KinematicsPose.Resolve(msg.position_ctrl_kinematics_pose)
    }
    else {
      resolved.position_ctrl_kinematics_pose = new KinematicsPose()
    }

    return resolved;
    }
};

module.exports = {
  Request: GetKinematicsPoseRequest,
  Response: GetKinematicsPoseResponse,
  md5sum() { return 'c8178071274fe9988bb6f690a77025ca'; },
  datatype() { return 'open_manipulator_msgs/GetKinematicsPose'; }
};
