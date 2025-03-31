// Auto-generated. Do not edit!

// (in-package open_manipulator_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class JointPose {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.joint_name = null;
      this.position = null;
      this.move_time = null;
    }
    else {
      if (initObj.hasOwnProperty('joint_name')) {
        this.joint_name = initObj.joint_name
      }
      else {
        this.joint_name = [];
      }
      if (initObj.hasOwnProperty('position')) {
        this.position = initObj.position
      }
      else {
        this.position = [];
      }
      if (initObj.hasOwnProperty('move_time')) {
        this.move_time = initObj.move_time
      }
      else {
        this.move_time = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type JointPose
    // Serialize message field [joint_name]
    bufferOffset = _arraySerializer.string(obj.joint_name, buffer, bufferOffset, null);
    // Serialize message field [position]
    bufferOffset = _arraySerializer.float64(obj.position, buffer, bufferOffset, null);
    // Serialize message field [move_time]
    bufferOffset = _serializer.float64(obj.move_time, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type JointPose
    let len;
    let data = new JointPose(null);
    // Deserialize message field [joint_name]
    data.joint_name = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [position]
    data.position = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [move_time]
    data.move_time = _deserializer.float64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.joint_name.forEach((val) => {
      length += 4 + _getByteLength(val);
    });
    length += 8 * object.position.length;
    return length + 16;
  }

  static datatype() {
    // Returns string type for a message object
    return 'open_manipulator_msgs/JointPose';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '217de178e1835082dfab23527f9a9de9';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string[]   joint_name
    float64[]  position
    float64    move_time
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new JointPose(null);
    if (msg.joint_name !== undefined) {
      resolved.joint_name = msg.joint_name;
    }
    else {
      resolved.joint_name = []
    }

    if (msg.position !== undefined) {
      resolved.position = msg.position;
    }
    else {
      resolved.position = []
    }

    if (msg.move_time !== undefined) {
      resolved.move_time = msg.move_time;
    }
    else {
      resolved.move_time = 0.0
    }

    return resolved;
    }
};

module.exports = JointPose;
