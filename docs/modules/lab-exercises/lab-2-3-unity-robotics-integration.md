---
sidebar_label: 'Lab 2.3: Unity Robotics Integration'
---

# Lab Exercise 2.3: Unity Robotics Integration

This lab exercise covers integrating Unity with ROS 2 for digital twin applications.

## Objectives

- Set up Unity with ROS-TCP-Connector
- Import robot models from URDF
- Implement basic communication between Unity and ROS 2
- Create a simple digital twin environment

## Prerequisites

- Unity Hub and Unity 2022.3 LTS
- ROS 2 Humble Hawksbill
- Basic Unity knowledge

## Unity Robotics Setup

### Installation

1. Download and install Unity Hub
2. Install Unity 2022.3 LTS
3. Download Unity Robotics Hub from Unity's website
4. Install the following packages:
   - ROS-TCP-Connector
   - Unity-Robotics-Helpers
   - URDF-Importer

### ROS-TCP-Connector Setup

The ROS-TCP-Connector enables communication between Unity and ROS 2:

1. In Unity, go to Window → Package Manager
2. Install ROS-TCP-Connector
3. Add the ROSConnection prefab to your scene

### Basic ROS Connection

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityROSConnection : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        // Get the ROS connection object
        ros = ROSConnection.GetOrCreateInstance();

        // Set the IP address and port for ROS communication
        ros.Initialize("127.0.0.1", 10000);
    }

    void Update()
    {
        // Send a message every second
        if (Time.time % 1.0f < Time.deltaTime)
        {
            var message = new UInt8Msg((byte)1);
            ros.Publish("unity_status", message);
        }
    }
}
```

## URDF Import

### Importing Robot Models

1. Place your URDF file in Unity's Assets folder
2. Use the URDF Importer: GameObject → 3D Object → URDF Robot
3. Select your URDF file
4. Configure import settings

### Custom URDF Import Script

```csharp
using UnityEngine;
using Unity.Robotics.UrdfImporter;

public class CustomUrdfImporter : MonoBehaviour
{
    public string urdfPath;
    public GameObject robotPrefab;

    void Start()
    {
        if (!string.IsNullOrEmpty(urdfPath))
        {
            robotPrefab = UrdfRobotExtensions.CreateRobot(urdfPath);
        }
    }

    public void ImportUrdf(string path)
    {
        if (robotPrefab != null)
        {
            DestroyImmediate(robotPrefab);
        }
        robotPrefab = UrdfRobotExtensions.CreateRobot(path);
    }
}
```

## Unity-ROS Communication

### Publisher Implementation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityPublisher : MonoBehaviour
{
    ROSConnection ros;
    public GameObject robot;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>("unity_joint_states");
        ros.RegisterPublisher<OdometryMsg>("unity_odom");
    }

    void Update()
    {
        // Publish joint states
        PublishJointStates();

        // Publish odometry
        PublishOdometry();
    }

    void PublishJointStates()
    {
        var jointState = new JointStateMsg();
        jointState.header.stamp = new TimeStamp(0, 0);
        jointState.header.frame_id = "unity_base";

        // Get joint positions from Unity
        // This would depend on your robot structure
        jointState.name = new string[] { "joint1", "joint2", "joint3" };
        jointState.position = new double[] { 0.1, 0.2, 0.3 };

        ros.Publish("unity_joint_states", jointState);
    }

    void PublishOdometry()
    {
        var odom = new OdometryMsg();
        odom.header.stamp = new TimeStamp(0, 0);
        odom.header.frame_id = "world";
        odom.child_frame_id = "unity_base";

        // Set position and orientation
        odom.pose.pose.position = new Vector3Msg(robot.transform.position.x,
                                               robot.transform.position.y,
                                               robot.transform.position.z);
        odom.pose.pose.orientation = new QuaternionMsg(robot.transform.rotation.x,
                                                      robot.transform.rotation.y,
                                                      robot.transform.rotation.z,
                                                      robot.transform.rotation.w);

        ros.Publish("unity_odom", odom);
    }
}
```

### Subscriber Implementation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnitySubscriber : MonoBehaviour
{
    public GameObject robot;
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to ROS topics
        ros.Subscribe<JointStateMsg>("joint_states", OnJointStateReceived);
        ros.Subscribe<TwistMsg>("cmd_vel", OnCmdVelReceived);
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Update robot joints in Unity
        for (int i = 0; i < jointState.name.Length && i < jointState.position.Length; i++)
        {
            UpdateJoint(jointState.name[i], jointState.position[i]);
        }
    }

    void OnCmdVelReceived(TwistMsg cmdVel)
    {
        // Move robot based on velocity command
        Vector3 movement = new Vector3((float)cmdVel.linear.x, 0, (float)cmdVel.linear.y);
        robot.transform.Translate(movement * Time.deltaTime);
    }

    void UpdateJoint(string jointName, double position)
    {
        // Find the joint in the robot hierarchy
        Transform joint = robot.transform.Find(jointName);
        if (joint != null)
        {
            // Update joint position (this is simplified)
            joint.localEulerAngles = new Vector3(0, 0, (float)position * Mathf.Rad2Deg);
        }
    }
}
```

## Digital Twin Environment

### Creating the Environment

1. Create a new Unity scene
2. Set up the environment (floor, walls, objects)
3. Import and position your robot
4. Configure lighting and materials

### Environment Synchronization

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class EnvironmentSync : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<StringMsg>("unity_environment_state", OnEnvironmentState);
    }

    void OnEnvironmentState(StringMsg state)
    {
        // Update Unity environment based on ROS state
        UpdateEnvironment(state.data);
    }

    void UpdateEnvironment(string state)
    {
        // Parse state string and update Unity objects
        // For example, move objects, change colors, etc.
        string[] parts = state.Split(',');
        if (parts.Length >= 3)
        {
            float x = float.Parse(parts[0]);
            float y = float.Parse(parts[1]);
            float z = float.Parse(parts[2]);

            // Move an object to the specified position
            transform.position = new Vector3(x, y, z);
        }
    }
}
```

## ROS 2 Bridge Setup

### ROS 2 Node for Bridge

Create a ROS 2 node to bridge Unity and other ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import socket
import json

class UnityBridgeNode(Node):
    def __init__(self):
        super().__init__('unity_bridge')

        # ROS 2 publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, 'unity_joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, 'unity_odom', 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Timer for publishing Unity data
        self.timer = self.create_timer(0.1, self.publish_unity_data)

        # Unity connection (TCP socket)
        self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.unity_socket.connect(('localhost', 10000))

    def joint_state_callback(self, msg):
        # Forward joint states to Unity
        unity_msg = {
            'type': 'joint_states',
            'data': {
                'names': list(msg.name),
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort)
            }
        }
        self.send_to_unity(json.dumps(unity_msg))

    def cmd_vel_callback(self, msg):
        # Forward velocity commands to Unity
        unity_msg = {
            'type': 'cmd_vel',
            'linear': {'x': msg.linear.x, 'y': msg.linear.y, 'z': msg.linear.z},
            'angular': {'x': msg.angular.x, 'y': msg.angular.y, 'z': msg.angular.z}
        }
        self.send_to_unity(json.dumps(unity_msg))

    def publish_unity_data(self):
        # Receive data from Unity and publish to ROS 2
        try:
            data = self.unity_socket.recv(1024).decode('utf-8')
            unity_data = json.loads(data)

            if unity_data['type'] == 'joint_states':
                # Publish joint states received from Unity
                msg = JointState()
                msg.name = unity_data['data']['names']
                msg.position = unity_data['data']['positions']
                self.joint_state_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error receiving Unity data: {e}')

    def send_to_unity(self, msg):
        try:
            self.unity_socket.send(msg.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f'Error sending to Unity: {e}')

def main(args=None):
    rclpy.init(args=args)
    unity_bridge = UnityBridgeNode()
    rclpy.spin(unity_bridge)
    unity_bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing the Integration

### Setup Steps

1. Start ROS 2:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Run the Unity bridge node:
   ```bash
   ros2 run my_robot_unity_bridge unity_bridge_node
   ```

3. Start Unity with your scene
4. Verify communication between Unity and ROS 2

### Verification Commands

Check ROS 2 topics:
```bash
# List available topics
ros2 topic list

# Echo Unity joint states
ros2 topic echo /unity_joint_states

# Send joint commands to Unity
ros2 topic pub /joint_states sensor_msgs/msg/JointState "name: ['joint1', 'joint2'] position: [0.5, -0.3]"
```

## Exercise Tasks

1. Set up Unity with ROS-TCP-Connector
2. Import a simple robot model from URDF
3. Implement a publisher to send Unity robot state to ROS 2
4. Implement a subscriber to receive ROS 2 commands in Unity
5. Create a simple environment in Unity
6. Test bidirectional communication between Unity and ROS 2

## Troubleshooting

### Common Issues

- **Connection failures**: Check IP addresses and ports
- **Message format issues**: Verify message types match
- **Performance problems**: Optimize update rates and data sizes
- **URDF import errors**: Check URDF file format and dependencies

### Debugging Tips

- Use Unity's console for error messages
- Monitor ROS 2 topics with `ros2 topic echo`
- Check network connectivity between Unity and ROS 2

## Summary

In this lab, you learned to integrate Unity with ROS 2 for digital twin applications. You implemented bidirectional communication, imported robot models, and created a synchronized environment between Unity and ROS 2.