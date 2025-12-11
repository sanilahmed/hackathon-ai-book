# Lab 2.3: Unity Robotics Integration

## Overview

In this lab, you will learn how to integrate Unity with ROS for robotics applications. You'll set up the Unity Robotics Hub, create a robot model in Unity, implement perception systems, and establish communication with ROS. This lab focuses on creating a digital twin environment that mirrors the Gazebo simulation.

## Objectives

By the end of this lab, you will be able to:
- Install and configure Unity Robotics Hub
- Set up ROS-TCP-Connector for Unity-ROS communication
- Create robot models and scenes in Unity for robotics
- Implement perception systems using Unity's Perception package
- Establish bidirectional communication between Unity and ROS
- Create synthetic data generation pipelines for computer vision

## Prerequisites

- Unity 2022.3 LTS installed
- Basic Unity development knowledge
- Completion of Module 1 (ROS 2 basics)
- Completion of Lab 2.1 and 2.2 (Gazebo and robot modeling)
- ROS 2 Humble Hawksbill installed
- Ubuntu 20.04 or 22.04 LTS

## Duration

4-5 hours

## Exercise 1: Install Unity Robotics Hub

### Step 1: Download and Install Unity Hub

1. Go to https://unity.com/download and download Unity Hub
2. Install Unity Hub following the Ubuntu installation instructions
3. Launch Unity Hub and sign in with your Unity ID

### Step 2: Install Unity 2022.3 LTS

1. In Unity Hub, go to the "Installs" tab
2. Click "Add" and select Unity 2022.3.x LTS version
3. Select the modules: Linux Build Support, Visual Scripting (optional)

### Step 3: Install Unity Robotics Hub packages

1. Create a new 3D project in Unity Hub
2. In Unity, go to Window → Package Manager
3. Install the following packages:
   - ROS-TCP-Connector
   - Perception package
   - Robotics Simulation Library (if available)

### Step 4: Alternative installation via Git URL

If packages aren't available in Package Manager:

1. In Package Manager, click the "+" button → "Add package from git URL..."
2. Add these repositories:
   - `https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector`
   - `https://github.com/Unity-Technologies/com.unity.perception.git`

## Exercise 2: Set up ROS-TCP-Connector

### Step 1: Create a new Unity project

1. Open Unity Hub
2. Create a new 3D Core project named "RoboticsSimulation"
3. Open the project in Unity

### Step 2: Configure ROS-TCP-Connector

1. In Unity, go to GameObject → Unity Robotics → ROS Settings
2. In the Inspector, set the ROS Connection settings:
   - ROS IP: `127.0.0.1` (localhost)
   - ROS Port: `10000`
   - Use SSL: false
   - Reconnection Delay: 1000 ms
   - Max Reconnection Attempts: 10

### Step 3: Create a basic ROS connection script

Create a C# script `Assets/Scripts/ROSConnectionManager.cs`:

```csharp
using System.Collections;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class ROSConnectionManager : MonoBehaviour
{
    ROSConnection ros;
    string rosTopic = "unity_message";

    // Start is called before the first frame update
    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.instance;

        // Subscribe to a topic
        ros.Subscribe<StringMsg>(rosTopic, OnMessageReceived);

        // Start sending messages
        StartCoroutine(SendMessage());
    }

    // Callback invoked when a message is received from ROS
    void OnMessageReceived(StringMsg msg)
    {
        Debug.Log("Received from ROS: " + msg.data);
    }

    // Send a message to ROS
    IEnumerator SendMessage()
    {
        while (true)
        {
            // Create a new message
            StringMsg msg = new StringMsg();
            msg.data = "Hello from Unity! Time: " + Time.time;

            // Send the message
            ros.Publish(rosTopic, msg);

            // Wait 1 second
            yield return new WaitForSeconds(1.0f);
        }
    }
}
```

### Step 4: Attach the script to Main Camera

1. In the Hierarchy, select the Main Camera
2. In the Inspector, click "Add Component"
3. Search for and add the "ROSConnectionManager" script

## Exercise 3: Create a Robot Model in Unity

### Step 1: Import the robot model

1. In Unity, create a new folder: `Assets/Models/Robot`
2. Create a simple robot using primitives:
   - Create a cube for the body
   - Create spheres for the head
   - Create cylinders for arms and legs

### Step 2: Create a robot hierarchy

Create a C# script `Assets/Scripts/RobotController.cs`:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string cmdVelTopic = "cmd_vel";
    string jointStatesTopic = "joint_states";

    // Robot components
    public Transform body;
    public Transform leftArm;
    public Transform rightArm;
    public Transform leftLeg;
    public Transform rightLeg;

    // Joint angles
    float leftArmAngle = 0f;
    float rightArmAngle = 0f;
    float leftLegAngle = 0f;
    float rightLegAngle = 0f;

    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to command velocity topic
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);
    }

    // Callback for velocity commands
    void OnCmdVelReceived(TwistMsg cmd)
    {
        // Move the robot based on linear and angular velocity
        transform.Translate(new Vector3((float)cmd.linear.x, 0, (float)cmd.linear.y) * Time.deltaTime);
        transform.Rotate(0, (float)cmd.angular.z * Time.deltaTime * 50f, 0);
    }

    // Update is called once per frame
    void Update()
    {
        // Animate joints (for demonstration)
        AnimateJoints();

        // Publish joint states periodically
        if (Time.frameCount % 60 == 0) // Every second at 60 FPS
        {
            PublishJointStates();
        }
    }

    void AnimateJoints()
    {
        // Simple joint animation
        leftArmAngle = Mathf.Sin(Time.time) * 30f;
        rightArmAngle = Mathf.Cos(Time.time) * 30f;
        leftLegAngle = Mathf.Sin(Time.time * 0.5f) * 15f;
        rightLegAngle = Mathf.Cos(Time.time * 0.5f) * 15f;

        // Apply rotations
        if (leftArm != null)
            leftArm.localRotation = Quaternion.Euler(leftArmAngle, 0, 0);
        if (rightArm != null)
            rightArm.localRotation = Quaternion.Euler(rightArmAngle, 0, 0);
        if (leftLeg != null)
            leftLeg.localRotation = Quaternion.Euler(leftLegAngle, 0, 0);
        if (rightLeg != null)
            rightLeg.localRotation = Quaternion.Euler(rightLegAngle, 0, 0);
    }

    void PublishJointStates()
    {
        // Create joint state message
        JointStateMsg jointState = new JointStateMsg();
        jointState.name = new string[] { "left_arm_joint", "right_arm_joint", "left_leg_joint", "right_leg_joint" };
        jointState.position = new double[] {
            leftArmAngle * Mathf.Deg2Rad,
            rightArmAngle * Mathf.Deg2Rad,
            leftLegAngle * Mathf.Deg2Rad,
            rightLegAngle * Mathf.Deg2Rad
        };
        jointState.velocity = new double[] { 0, 0, 0, 0 };
        jointState.effort = new double[] { 0, 0, 0, 0 };
        jointState.header.stamp = new TimeStamp(Time.time);
        jointState.header.frame_id = "base_link";

        ros.Publish(jointStatesTopic, jointState);
    }
}
```

### Step 3: Set up the robot hierarchy

1. Create an empty GameObject named "Robot"
2. Add the RobotController script to it
3. Create child objects for body, head, arms, and legs
4. Assign the transforms in the RobotController script

## Exercise 4: Implement Perception System

### Step 1: Install and configure Perception package

1. In Package Manager, install the Perception package
2. Create a new Perception Camera:
   - GameObject → Perception → Perception Camera
   - Position it appropriately on your robot

### Step 2: Create a perception script

Create `Assets/Scripts/PerceptionManager.cs`:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using System.Linq;

public class PerceptionManager : MonoBehaviour
{
    ROSConnection ros;
    string cameraInfoTopic = "camera_info";
    string imageTopic = "image_raw";

    public Camera perceptionCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;

    RenderTexture renderTexture;
    PerceptionCamera perceptionCamComponent;

    void Start()
    {
        ros = ROSConnection.instance;

        // Set up render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        perceptionCamera.targetTexture = renderTexture;

        // Add perception camera component if not already present
        perceptionCamComponent = perceptionCamera.GetComponent<PerceptionCamera>();
        if (perceptionCamComponent == null)
        {
            perceptionCamComponent = perceptionCamera.gameObject.AddComponent<PerceptionCamera>();
        }

        // Configure perception camera
        perceptionCamComponent.captureRgbImages = true;
        perceptionCamComponent.rgbImageCaptureSettings.frequency = CaptureFrequency.Variable;
        perceptionCamComponent.rgbImageCaptureSettings.targetFrameRate = 30;
    }

    void Update()
    {
        // Capture and publish image periodically
        if (Time.frameCount % 2 == 0) // Every other frame at 60 FPS = 30 FPS
        {
            PublishCameraImage();
        }
    }

    void PublishCameraImage()
    {
        // Create texture to read from
        Texture2D texture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        // Set the active render texture
        RenderTexture.active = renderTexture;

        // Copy the pixels from the render texture to the texture
        texture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture.Apply();

        // Convert to byte array (this is a simplified approach)
        byte[] imageBytes = texture.EncodeToJPG();

        // Create ROS image message (simplified - in practice, you'd use proper encoding)
        // For this example, we'll just publish a placeholder
        StringMsg imageMsg = new StringMsg();
        imageMsg.data = "Image data would go here";

        ros.Publish(imageTopic, imageMsg);

        // Clean up
        Destroy(texture);
    }

    void OnDestroy()
    {
        if (renderTexture != null)
            Destroy(renderTexture);
    }
}
```

## Exercise 5: Create a Complete Unity Robot Scene

### Step 1: Set up the scene

1. Create a new scene: File → New Scene
2. Save it as "RobotScene" in Assets/Scenes/
3. Add lighting: GameObject → Light → Directional Light
4. Add ground plane: GameObject → 3D Object → Plane

### Step 2: Create the complete robot

Create a more detailed robot with proper joint setup:

Create `Assets/Scripts/DetailedRobotController.cs`:

```csharp
using System.Collections;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Nav_msgs;

public class DetailedRobotController : MonoBehaviour
{
    ROSConnection ros;

    // ROS topics
    string cmdVelTopic = "/cmd_vel";
    string jointStatesTopic = "/joint_states";
    string odomTopic = "/odom";
    string imuTopic = "/imu";
    string laserScanTopic = "/scan";

    // Robot configuration
    public Transform baseLink;
    public Transform[] joints; // Array of joint transforms
    public string[] jointNames; // Corresponding joint names

    // Robot state
    Vector3 position;
    Quaternion rotation;
    Vector3 velocity;
    Vector3 angularVelocity;

    // IMU simulation
    float imuRoll, imuPitch, imuYaw;

    void Start()
    {
        ros = ROSConnection.instance;

        // Initialize robot state
        position = transform.position;
        rotation = transform.rotation;

        // Subscribe to command topics
        ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);

        // Start publishing sensor data
        StartCoroutine(PublishSensorData());
    }

    void OnCmdVelReceived(TwistMsg cmd)
    {
        // Apply velocity commands to robot
        Vector3 linear = new Vector3((float)cmd.linear.x, 0, (float)cmd.linear.y);
        float angular = (float)cmd.angular.z;

        // Convert to world space and apply
        position += transform.TransformDirection(linear) * Time.deltaTime;
        transform.position = position;

        // Apply angular rotation
        transform.Rotate(0, angular * Time.deltaTime * Mathf.Rad2Deg, 0);

        // Update state
        velocity = linear;
        angularVelocity = new Vector3(0, angular, 0);
    }

    IEnumerator PublishSensorData()
    {
        while (true)
        {
            // Publish joint states
            PublishJointStates();

            // Publish odometry
            PublishOdometry();

            // Publish IMU data
            PublishIMUData();

            // Publish laser scan (simulated)
            PublishLaserScan();

            yield return new WaitForSeconds(0.1f); // 10 Hz
        }
    }

    void PublishJointStates()
    {
        JointStateMsg jointState = new JointStateMsg();
        jointState.name = jointNames;
        jointState.position = new double[joints.Length];
        jointState.velocity = new double[joints.Length];
        jointState.effort = new double[joints.Length];

        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] != null)
            {
                // Get joint angle (simplified - you'd need to extract the relevant rotation)
                jointState.position[i] = joints[i].localEulerAngles.y * Mathf.Deg2Rad; // Example
            }
        }

        jointState.header.stamp = new TimeStamp(Time.time);
        jointState.header.frame_id = "base_link";

        ros.Publish(jointStatesTopic, jointState);
    }

    void PublishOdometry()
    {
        OdometryMsg odom = new OdometryMsg();

        odom.header.stamp = new TimeStamp(Time.time);
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";

        // Position
        odom.pose.pose.position.x = transform.position.x;
        odom.pose.pose.position.y = transform.position.z; // Unity Z -> ROS Y
        odom.pose.pose.position.z = transform.position.y; // Unity Y -> ROS Z

        // Orientation (convert Unity to ROS coordinate system)
        Quaternion unityRot = transform.rotation;
        odom.pose.pose.orientation.x = unityRot.x;
        odom.pose.pose.orientation.y = unityRot.z; // Swap Y and Z
        odom.pose.pose.orientation.z = unityRot.y;
        odom.pose.pose.orientation.w = unityRot.w;

        // Velocity
        odom.twist.twist.linear.x = velocity.x;
        odom.twist.twist.linear.y = velocity.z;
        odom.twist.twist.linear.z = velocity.y;

        odom.twist.twist.angular.x = angularVelocity.x;
        odom.twist.twist.angular.y = angularVelocity.z;
        odom.twist.twist.angular.z = angularVelocity.y;

        ros.Publish(odomTopic, odom);
    }

    void PublishIMUData()
    {
        ImuMsg imu = new ImuMsg();

        imu.header.stamp = new TimeStamp(Time.time);
        imu.header.frame_id = "imu_link";

        // Simulate IMU data
        imu.orientation.x = transform.rotation.x;
        imu.orientation.y = transform.rotation.z;
        imu.orientation.z = transform.rotation.y;
        imu.orientation.w = transform.rotation.w;

        // Angular velocity (simplified)
        imu.angular_velocity.x = angularVelocity.x;
        imu.angular_velocity.y = angularVelocity.z;
        imu.angular_velocity.z = angularVelocity.y;

        // Linear acceleration (include gravity)
        imu.linear_acceleration.x = Physics.gravity.x;
        imu.linear_acceleration.y = Physics.gravity.z;
        imu.linear_acceleration.z = Physics.gravity.y;

        ros.Publish(imuTopic, imu);
    }

    void PublishLaserScan()
    {
        LaserScanMsg scan = new LaserScanMsg();

        scan.header.stamp = new TimeStamp(Time.time);
        scan.header.frame_id = "laser_link";

        // Laser scan parameters
        scan.angle_min = -Mathf.PI / 2; // -90 degrees
        scan.angle_max = Mathf.PI / 2;  // 90 degrees
        scan.angle_increment = Mathf.PI / 180; // 1 degree
        scan.time_increment = 0.0;
        scan.scan_time = 0.1f;
        scan.range_min = 0.1f;
        scan.range_max = 10.0f;

        // Simulate ranges (simplified)
        int numReadings = Mathf.RoundToInt((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1;
        scan.ranges = new float[numReadings];

        for (int i = 0; i < numReadings; i++)
        {
            // Simulate distance readings (in a real implementation, you'd use raycasting)
            scan.ranges[i] = Random.Range(1.0f, 5.0f); // Random distances for simulation
        }

        ros.Publish(laserScanTopic, scan);
    }
}
```

### Step 3: Configure the robot controller

1. Add the DetailedRobotController script to your robot root object
2. Assign the base link and joint transforms in the Inspector
3. Set up joint names array in the Inspector

## Exercise 6: Test Unity-ROS Communication

### Step 1: Set up ROS bridge

First, let's create a simple ROS node to test communication:

Create a ROS 2 package for testing:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python unity_test_py --dependencies rclpy geometry_msgs sensor_msgs
```

Create `~/ros2_ws/src/unity_test_py/unity_test_py/test_subscriber.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu, LaserScan
from nav_msgs.msg import Odometry


class UnityTestSubscriber(Node):
    def __init__(self):
        super().__init__('unity_test_subscriber')

        # Create subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info('Unity test subscriber started')

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f'Command velocity received: linear=({msg.linear.x}, {msg.linear.y}, {msg.linear.z}), angular=({msg.angular.x}, {msg.angular.y}, {msg.angular.z})')

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Joint states received: {len(msg.name)} joints')

    def odom_callback(self, msg):
        self.get_logger().info(f'Odometry received: pos=({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}, {msg.pose.pose.position.z:.2f})')

    def imu_callback(self, msg):
        self.get_logger().info(f'IMU data received')

    def scan_callback(self, msg):
        self.get_logger().info(f'Laser scan received: {len(msg.ranges)} readings')


def main(args=None):
    rclpy.init(args=args)
    subscriber = UnityTestSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Update the setup.py:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'unity_test_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Unity-ROS test package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_subscriber = unity_test_py.test_subscriber:main',
        ],
    },
)
```

### Step 2: Build and run the test

```bash
cd ~/ros2_ws
colcon build --packages-select unity_test_py
source ~/ros2_ws/install/setup.bash
ros2 run unity_test_py test_subscriber
```

### Step 3: Configure Unity to connect to ROS

1. In Unity, make sure your ROS settings point to the correct IP and port
2. If running ROS on the same machine: IP = 127.0.0.1, Port = 10000
3. If running ROS on a different machine, use the appropriate IP address

## Exercise 7: Create a Digital Twin Scene

### Step 1: Create a synchronized environment

Create `Assets/Scripts/DigitalTwinManager.cs`:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Tf2_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class DigitalTwinManager : MonoBehaviour
{
    ROSConnection ros;

    // TF topics
    string tfTopic = "/tf";
    string tfStaticTopic = "/tf_static";

    // Robot synchronization
    public GameObject robotPrefab;
    Dictionary<string, GameObject> robotInstances = new Dictionary<string, GameObject>();

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to TF topics to synchronize transforms
        ros.Subscribe<TFMessage>(tfTopic, OnTFReceived);
    }

    void OnTFReceived(TFMessage tfMsg)
    {
        foreach (var transform in tfMsg.transforms)
        {
            string frameId = transform.child_frame_id;

            // Find or create robot instance for this frame
            if (!robotInstances.ContainsKey(frameId))
            {
                GameObject robotObj = Instantiate(robotPrefab);
                robotObj.name = frameId;
                robotInstances[frameId] = robotObj;
            }

            // Update the robot's position and rotation
            GameObject robot = robotInstances[frameId];
            if (robot != null)
            {
                // Convert ROS transform to Unity coordinates
                Vector3 position = new Vector3(
                    (float)transform.transform.translation.x,
                    (float)transform.transform.translation.z, // ROS Z -> Unity Y
                    (float)transform.transform.translation.y  // ROS Y -> Unity Z
                );

                Quaternion rotation = new Quaternion(
                    (float)transform.transform.rotation.x,
                    (float)transform.transform.rotation.z, // ROS Z -> Unity Y
                    (float)transform.transform.rotation.y, // ROS Y -> Unity Z
                    (float)transform.transform.rotation.w
                );

                robot.transform.position = position;
                robot.transform.rotation = rotation;
            }
        }
    }

    // Method to publish Unity transforms to ROS (for synchronization)
    public void PublishTransformToROS(string frameId, string parentFrameId, Transform unityTransform)
    {
        TFMessage tfMsg = new TFMessage();

        Geometry_msgs.TransformStamped transformStamped = new Geometry_msgs.TransformStamped();
        transformStamped.header.stamp = new TimeStamp(Time.time);
        transformStamped.header.frame_id = parentFrameId;
        transformStamped.child_frame_id = frameId;

        // Convert Unity transform to ROS coordinates
        transformStamped.transform.translation.x = unityTransform.position.x;
        transformStamped.transform.translation.y = unityTransform.position.z; // Unity Z -> ROS Y
        transformStamped.transform.translation.z = unityTransform.position.y; // Unity Y -> ROS Z

        transformStamped.transform.rotation.x = unityTransform.rotation.x;
        transformStamped.transform.rotation.y = unityTransform.rotation.z; // Unity Z -> ROS Y
        transformStamped.transform.rotation.z = unityTransform.rotation.y; // Unity Y -> ROS Z
        transformStamped.transform.rotation.w = unityTransform.rotation.w;

        tfMsg.transforms = new Geometry_msgs.TransformStamped[] { transformStamped };

        ros.Publish(tfTopic, tfMsg);
    }
}
```

### Step 2: Create a synchronized scene

1. Create a new scene: "DigitalTwinScene"
2. Add the DigitalTwinManager script to an empty GameObject
3. Set up the scene to mirror your Gazebo environment
4. Add synchronized objects that update based on ROS TF data

## Exercise 8: Implement Synthetic Data Generation

### Step 1: Set up Perception package for synthetic data

1. In Unity, go to Window → Unity Perception → Perception Camera Manager
2. Add a Perception Camera to your scene
3. Configure it for synthetic data generation

### Step 2: Create synthetic data generation script

Create `Assets/Scripts/SyntheticDataGenerator.cs`:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using Unity.Robotics.ROSTCPConnector;

public class SyntheticDataGenerator : MonoBehaviour
{
    public PerceptionCamera perceptionCamera;
    public int captureFrequency = 10; // Hz
    public int imageWidth = 640;
    public int imageHeight = 480;

    private RenderTexture m_RenderTexture;
    private int m_FrameCounter = 0;
    private float m_LastCaptureTime = 0f;

    void Start()
    {
        // Create render texture for synthetic data
        m_RenderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        perceptionCamera.GetComponent<Camera>().targetTexture = m_RenderTexture;

        // Configure perception camera
        var perceptionCam = perceptionCamera.GetComponent<PerceptionCamera>();
        perceptionCam.rgbImageCaptureSettings.frequency = CaptureFrequency.Variable;
        perceptionCam.rgbImageCaptureSettings.targetFrameRate = captureFrequency;

        // Add semantic segmentation if needed
        var labeler = perceptionCamera.gameObject.AddComponent<SegmentationLabeler>();
    }

    void Update()
    {
        // Additional synthetic data processing can go here
        float timeInterval = 1f / captureFrequency;

        if (Time.time - m_LastCaptureTime >= timeInterval)
        {
            m_LastCaptureTime = Time.time;

            // Process synthetic data capture
            ProcessSyntheticDataCapture();
        }
    }

    void ProcessSyntheticDataCapture()
    {
        // This is where you would process and potentially send
        // synthetic data to ROS or save it locally
        Debug.Log($"Synthetic data captured at frame {m_FrameCounter}");
        m_FrameCounter++;
    }

    void OnDestroy()
    {
        if (m_RenderTexture != null)
            Destroy(m_RenderTexture);
    }
}
```

## Exercise 9: Testing and Validation

### Step 1: Create a test scene

1. Create a new scene with a simple environment
2. Add your robot model
3. Include the ROS connection and perception components
4. Test the basic functionality

### Step 2: Verify communication

1. Run Unity with the robot scene
2. Start ROS nodes that publish to the topics your Unity scene subscribes to
3. Verify that Unity receives and responds to commands
4. Check that Unity publishes sensor data that ROS nodes can receive

### Step 3: Test with RViz

Create a ROS launch file to visualize the Unity data:

Create `~/ros2_ws/src/unity_test_py/unity_test_py/launch/unity_visualization.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # RViz node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('unity_test_py'), 'config', 'unity.rviz')],
        output='screen'
    )

    # Robot state publisher (for TF visualization)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        rviz,
        robot_state_publisher
    ])
```

## Troubleshooting

### Common Issues and Solutions

1. **Connection issues between Unity and ROS**:
   - Verify IP address and port settings
   - Check firewall settings
   - Ensure ROS bridge is running
   - Use `telnet <ip> <port>` to test connectivity

2. **Coordinate system mismatches**:
   - Unity uses left-handed coordinate system, ROS uses right-handed
   - Convert between coordinate systems when publishing/subscribing
   - Pay attention to axis mappings (Y/Z swaps)

3. **Performance issues**:
   - Reduce rendering quality for real-time performance
   - Limit the frequency of data publishing
   - Use appropriate image resolutions

4. **TF synchronization problems**:
   - Ensure consistent frame naming
   - Check TF tree validity
   - Verify timestamp synchronization

5. **Perception package errors**:
   - Ensure Perception package is properly installed
   - Check for required dependencies
   - Verify camera configurations

## Assessment Questions

1. What are the key differences between Unity's and ROS's coordinate systems?
2. How do you establish communication between Unity and ROS?
3. What is the purpose of the Perception package in Unity?
4. How would you synchronize robot states between Unity and Gazebo?
5. What are the advantages of using Unity for robotics simulation?

## Extension Exercises

1. Create a complete humanoid robot model in Unity with all joints
2. Implement a navigation system that works with both Unity and ROS
3. Add realistic physics to the Unity simulation
4. Create a user interface for controlling the Unity robot
5. Implement machine learning training using Unity's synthetic data

## Summary

In this lab, you successfully:
- Installed and configured Unity Robotics Hub
- Set up ROS-TCP-Connector for Unity-ROS communication
- Created robot models and scenes in Unity for robotics
- Implemented perception systems using Unity's Perception package
- Established bidirectional communication between Unity and ROS
- Created synthetic data generation pipelines

These skills enable you to create sophisticated digital twin environments that complement your Gazebo simulations, providing a more immersive visualization and perception development platform. The Unity-ROS integration opens up possibilities for advanced human-robot interaction, realistic sensor simulation, and synthetic data generation for AI training.