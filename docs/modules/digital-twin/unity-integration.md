# Unity Integration for Humanoid Robotics

## Introduction to Unity for Robotics

Unity is a powerful 3D development platform that, when combined with robotics frameworks, provides high-fidelity visualization and simulation capabilities. For humanoid robotics, Unity offers photorealistic rendering, advanced physics simulation, and intuitive development tools that complement traditional robotics simulation environments like Gazebo.

## Unity Robotics Setup

### Prerequisites

Before integrating Unity with your robotics project, ensure you have:
- Unity Hub installed (version 3.0 or later)
- Unity 2022.3 LTS installed (as specified in the requirements)
- Basic understanding of Unity's interface and concepts
- A robotics development environment with ROS 2

### Installing Unity Robotics Tools

Unity provides several tools specifically designed for robotics development:

1. **Unity Robotics Hub**: Centralized package management for robotics tools
2. **Unity Robotics Package**: Core components for robotics simulation
3. **ROS-TCP-Connector**: Communication bridge between Unity and ROS 2
4. **Unity Perception Package**: Tools for generating synthetic training data

### Setting up Unity Robotics Hub

1. Open Unity Hub
2. Go to the "Packages" tab
3. Install "Unity Robotics Hub" from the package manager
4. This will provide access to all robotics-specific packages

### Creating a Robotics Project

1. Open Unity Hub and create a new 3D project
2. Import the required robotics packages:
   - ROS-TCP-Connector
   - Unity Perception
   - Unity Robotics Package

## ROS-TCP-Connector Integration

The ROS-TCP-Connector package enables communication between Unity and ROS 2 through TCP/IP sockets.

### Installing ROS-TCP-Connector

```bash
# In your Unity project, use the Package Manager:
# Window → Package Manager → Add package from git URL
# Add: https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector
```

### Basic Unity-ROS Communication Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class UnityROSConnection : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        // Get the ROS connection asset
        ros = ROSConnection.GetOrCreateInstance();

        // Register a publisher
        ros.RegisterPublisher<StringMsg>("unity_status");

        // Register a subscriber
        ros.Subscribe<Float32Msg>("unity_joint_positions", OnJointPositionReceived);
    }

    void OnJointPositionReceived(Float32Msg msg)
    {
        // Handle incoming joint position data
        Debug.Log("Received joint position: " + msg.data);
    }

    void Update()
    {
        // Publish status message
        var statusMsg = new StringMsg();
        statusMsg.data = "Unity simulation running";
        ros.Publish("unity_status", statusMsg);
    }
}
```

## Creating a Humanoid Robot in Unity

### Importing Robot Models

Unity can import robot models in several formats:
- **URDF Importer**: Direct import of URDF files (if available)
- **FBX/GLTF**: Standard 3D model formats
- **Custom**: Manually created Unity prefabs

### Setting up Robot Hierarchy

A humanoid robot in Unity should follow a proper transform hierarchy:

```
HumanoidRobot (Root)
├── Torso
│   ├── Head
│   ├── LeftShoulder
│   │   ├── LeftUpperArm
│   │   ├── LeftLowerArm
│   │   └── LeftHand
│   ├── RightShoulder
│   │   ├── RightUpperArm
│   │   ├── RightLowerArm
│   │   └── RightHand
│   ├── LeftHip
│   │   ├── LeftUpperLeg
│   │   ├── LeftLowerLeg
│   │   └── LeftFoot
│   └── RightHip
│       ├── RightUpperLeg
│       ├── RightLowerLeg
│       └── RightFoot
```

### Configuring Joints and Constraints

Unity's built-in physics system can simulate robotic joints:

```csharp
using UnityEngine;

public class RobotJointController : MonoBehaviour
{
    public ConfigurableJoint joint;
    public float targetPosition = 0f;
    public float stiffness = 1000f;
    public float damping = 100f;

    void Start()
    {
        // Configure joint properties
        joint.xMotion = ConfigurableJointMotion.Locked;
        joint.yMotion = ConfigurableJointMotion.Locked;
        joint.zMotion = ConfigurableJointMotion.Locked;

        // Allow rotation around specific axis
        joint.angularXMotion = ConfigurableJointMotion.Limited;
        joint.angularYMotion = ConfigurableJointMotion.Limited;
        joint.angularZMotion = ConfigurableJointMotion.Limited;
    }

    void Update()
    {
        // Apply target position using spring force
        joint.targetRotation = Quaternion.Euler(0, 0, targetPosition);

        // Configure joint drive for position control
        JointDrive drive = joint.angularXDrive;
        drive.positionSpring = stiffness;
        drive.positionDamper = damping;
        joint.angularXDrive = drive;
    }

    public void SetJointPosition(float position)
    {
        targetPosition = position;
    }
}
```

## Unity Perception Package Integration

The Unity Perception package enables generation of synthetic training data for AI models.

### Installing Unity Perception

1. In Unity Package Manager, add package from git URL:
   `https://github.com/Unity-Technologies/Unity-Perception.git`

2. The package provides:
   - Synthetic data generation tools
   - Sensor simulation (cameras, LiDAR, etc.)
   - Annotation tools for training data

### Setting up Synthetic Data Generation

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;

public class PerceptionSetup : MonoBehaviour
{
    void Start()
    {
        // Add synthetic camera sensor
        var camera = GetComponent<Camera>();
        var syntheticCamera = camera.gameObject.AddComponent<SyntheticCamera>();

        // Configure synthetic camera properties
        syntheticCamera.captureRgb = true;
        syntheticCamera.captureDepth = true;
        syntheticCamera.captureSegmentation = true;
        syntheticCamera.captureOpticalFlow = true;

        // Add object annotation
        var annotationManager = FindObjectOfType<AnnotationManager>();
        if (annotationManager == null)
        {
            var managerObject = new GameObject("AnnotationManager");
            annotationManager = managerObject.AddComponent<AnnotationManager>();
        }
    }
}
```

## Creating Unity-ROS Bridge

### Python Bridge Script

Create a Python script to bridge Unity and ROS 2:

```python
#!/usr/bin/env python3
"""
Unity-ROS Bridge for Humanoid Robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import socket
import json
import threading
import time

class UnityROSBridge(Node):
    def __init__(self):
        super().__init__('unity_ros_bridge')

        # ROS publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, '/unity/joint_states', 10)
        self.robot_cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.robot_cmd_callback,
            10
        )

        # Unity connection setup
        self.unity_socket = None
        self.unity_host = 'localhost'
        self.unity_port = 10000  # Default Unity TCP port

        # Start Unity connection in separate thread
        self.unity_thread = threading.Thread(target=self.connect_to_unity)
        self.unity_thread.daemon = True
        self.unity_thread.start()

    def connect_to_unity(self):
        """Establish connection with Unity application"""
        while rclpy.ok():
            try:
                self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.unity_socket.connect((self.unity_host, self.unity_port))
                self.get_logger().info('Connected to Unity')

                # Start listening for Unity messages
                self.listen_to_unity()

            except Exception as e:
                self.get_logger().error(f'Unity connection failed: {e}')
                time.sleep(2)  # Wait before retrying

    def listen_to_unity(self):
        """Listen for messages from Unity"""
        while rclpy.ok():
            try:
                data = self.unity_socket.recv(4096)
                if data:
                    message = json.loads(data.decode('utf-8'))
                    self.process_unity_message(message)
            except Exception as e:
                self.get_logger().error(f'Error receiving from Unity: {e}')
                break

    def process_unity_message(self, message):
        """Process messages received from Unity"""
        msg_type = message.get('type', '')

        if msg_type == 'joint_states':
            self.publish_joint_states(message)
        elif msg_type == 'sensor_data':
            self.publish_sensor_data(message)

    def publish_joint_states(self, message):
        """Publish joint states from Unity to ROS"""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = message.get('joint_names', [])
        joint_msg.position = message.get('joint_positions', [])
        joint_msg.velocity = message.get('joint_velocities', [])
        joint_msg.effort = message.get('joint_efforts', [])

        self.joint_state_pub.publish(joint_msg)

    def robot_cmd_callback(self, msg):
        """Send robot commands to Unity"""
        if self.unity_socket:
            command = {
                'type': 'robot_command',
                'linear_x': msg.linear.x,
                'angular_z': msg.angular.z
            }

            try:
                self.unity_socket.send(json.dumps(command).encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f'Error sending to Unity: {e}')

    def publish_sensor_data(self, message):
        """Publish sensor data received from Unity"""
        # Implement sensor data publishing based on message content
        pass

def main(args=None):
    rclpy.init(args=args)

    bridge = UnityROSBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Setting up Visualization Pipeline

### Real-time Robot State Visualization

To visualize robot state changes in real-time:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class RobotStateVisualizer : MonoBehaviour
{
    [System.Serializable]
    public class JointMapping
    {
        public string jointName;
        public Transform jointTransform;
    }

    public List<JointMapping> jointMappings = new List<JointMapping>();
    private Dictionary<string, Transform> jointDict = new Dictionary<string, Transform>();

    void Start()
    {
        // Create dictionary for quick lookup
        foreach (var mapping in jointMappings)
        {
            jointDict[mapping.jointName] = mapping.jointTransform;
        }
    }

    public void UpdateRobotPose(List<string> jointNames, List<float> jointPositions)
    {
        for (int i = 0; i < jointNames.Count && i < jointPositions.Count; i++)
        {
            string jointName = jointNames[i];
            float jointPos = jointPositions[i];

            if (jointDict.ContainsKey(jointName))
            {
                Transform jointTransform = jointDict[jointName];

                // Apply joint position to transform
                // This depends on joint type and axis
                jointTransform.localEulerAngles = new Vector3(0, jointPos * Mathf.Rad2Deg, 0);
            }
        }
    }
}
```

## Unity Scene Setup for Humanoid Robotics

### Creating a Basic Scene

1. **Environment Setup**:
   - Create a ground plane with realistic materials
   - Add lighting (directional light for sun simulation)
   - Set up skybox for realistic environment

2. **Robot Placement**:
   - Import humanoid robot as prefab
   - Position in the center of the scene
   - Configure physics properties

3. **Camera Setup**:
   - Main camera for visualization
   - Additional cameras for sensor simulation
   - Configure camera properties for realistic rendering

### Example Scene Structure

```
Main Scene
├── Environment
│   ├── Ground Plane
│   ├── Directional Light (Sun)
│   ├── Skybox
│   └── Environment Objects
├── Humanoid Robot
│   ├── Robot Prefab
│   ├── Robot Controller Script
│   └── Joint Controllers
├── Cameras
│   ├── Main Camera
│   ├── RGB Camera (for sensor simulation)
│   └── Depth Camera
└── ROS Bridge
    ├── ROS Connection Manager
    └── Message Handlers
```

## Performance Optimization

For complex humanoid robot visualization:

1. **LOD (Level of Detail)**: Use simplified models at distance
2. **Occlusion Culling**: Hide objects not visible to camera
3. **Texture Compression**: Optimize textures for real-time rendering
4. **Batching**: Combine similar meshes for better performance
5. **Shader Optimization**: Use efficient shaders for robot materials

## Troubleshooting Unity Integration

### Common Issues and Solutions

1. **Connection Problems**:
   - Verify Unity and ROS are on same network
   - Check firewall settings
   - Confirm correct IP addresses and ports

2. **Performance Issues**:
   - Reduce polygon count of robot models
   - Lower texture resolutions
   - Simplify physics calculations

3. **Synchronization Problems**:
   - Verify time synchronization between systems
   - Check message frequency and buffering
   - Confirm data format compatibility

## Best Practices

1. **Modular Design**: Create reusable robot prefabs
2. **Configuration Management**: Use scriptable objects for robot parameters
3. **Testing**: Validate Unity-ROS communication independently
4. **Documentation**: Document all Unity-ROS interfaces
5. **Version Control**: Use appropriate version control for Unity assets

## Integration with Gazebo

Unity can complement Gazebo by providing high-fidelity visualization while Gazebo handles physics simulation:

```
ROS 2 Ecosystem
├── Gazebo (Physics & Sensor Simulation)
├── Unity (Visualization & Perception)
├── Robot Controllers (Motion Planning)
└── AI Components (Perception & Decision Making)
```

This dual-simulation approach allows for realistic physics in Gazebo while providing photorealistic visualization in Unity.

## Summary

Unity integration provides high-fidelity visualization capabilities for humanoid robotics, complementing physics simulation environments like Gazebo. The Unity-ROS bridge enables seamless communication between the visualization layer and the ROS 2 ecosystem, allowing for realistic rendering of robot states, sensor data, and environmental interactions. Proper setup involves configuring the ROS-TCP-Connector, creating appropriate robot models, and optimizing performance for real-time visualization.

## Learning Check

After studying this section, you should be able to:
- Set up Unity for robotics applications with appropriate packages
- Configure ROS-TCP-Connector for Unity-ROS communication
- Create and animate humanoid robot models in Unity
- Implement visualization pipelines for robot state
- Troubleshoot common Unity-ROS integration issues