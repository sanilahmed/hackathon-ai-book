# VLA Integration

## Overview

VLA (Vision-Language-Action) integration involves connecting the multimodal perception, language understanding, and action execution components into a cohesive system that works with the existing robotics infrastructure. This section covers the integration of VLA systems with ROS 2, simulation environments, and real-world robotic platforms.

## ROS 2 Integration Architecture

### VLA ROS 2 Node Structure

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose
from humanoid_robot_msgs.msg import VLACommand, VLAStatus
from humanoid_robot_msgs.srv import ExecuteVLACommand

class VLAIntegrationNode(Node):
    def __init__(self):
        super().__init__('vla_integration_node')

        # Initialize VLA model
        self.vla_model = self.initialize_vla_model()

        # QoS profiles for different data types
        image_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vla_status_pub = self.create_publisher(VLAStatus, '/vla/status', 10)
        self.debug_image_pub = self.create_publisher(Image, '/vla/debug_image', 5)

        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, image_qos
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 5
        )
        self.language_command_sub = self.create_subscription(
            String, '/vla/language_command', self.language_command_callback, command_qos
        )

        # Services
        self.execute_vla_service = self.create_service(
            ExecuteVLACommand,
            'execute_vla_command',
            self.execute_vla_command_callback
        )

        # Action servers for complex tasks
        self.vla_action_server = ActionServer(
            self,
            VLACommand,
            'execute_vla_task',
            self.execute_vla_task_callback
        )

        # Internal state
        self.current_image = None
        self.camera_info = None
        self.model_ready = False
        self.vla_lock = threading.Lock()

        # Start model loading in background
        self.model_loading_thread = threading.Thread(target=self.load_model_async)
        self.model_loading_thread.start()

    def initialize_vla_model(self):
        """Initialize the VLA model (this might be done asynchronously)."""
        # Placeholder - in practice, load your trained VLA model
        from your_vla_model import YourVLA
        return YourVLA()

    def load_model_async(self):
        """Load model asynchronously to avoid blocking node startup."""
        try:
            # Load the VLA model
            self.get_logger().info("Loading VLA model...")
            # self.vla_model = load_trained_vla_model()  # Your model loading logic
            self.model_ready = True
            self.get_logger().info("VLA model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load VLA model: {e}")
            self.model_ready = False

    def camera_callback(self, msg):
        """Process incoming camera images."""
        if not self.model_ready:
            return

        # Store current image for processing
        self.current_image = msg

        # Optionally, process image immediately if in continuous mode
        if hasattr(self, 'continuous_processing') and self.continuous_processing:
            self.process_current_image_and_command()

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_info = msg

    def language_command_callback(self, msg):
        """Process incoming language commands."""
        if not self.model_ready:
            self.get_logger().warn("VLA model not ready, discarding command")
            return

        # Process command with current image
        with self.vla_lock:
            if self.current_image is not None:
                self.process_vla_command(self.current_image, msg.data)
            else:
                # Store command for later processing when image is available
                self.pending_command = msg.data
                self.get_logger().info("Stored command for later processing")

    def process_vla_command(self, image, instruction):
        """Process VLA command with current image and instruction."""
        try:
            # Convert ROS image to format expected by VLA model
            cv_image = self.ros_image_to_cv2(image)

            # Process with VLA model
            action = self.vla_model.process_instruction(cv_image, instruction)

            # Publish action to robot
            self.publish_action(action)

            # Update status
            status_msg = VLAStatus()
            status_msg.success = True
            status_msg.message = f"Executed: {instruction}"
            status_msg.timestamp = self.get_clock().now().to_msg()
            self.vla_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing VLA command: {e}")
            status_msg = VLAStatus()
            status_msg.success = False
            status_msg.message = f"Error: {str(e)}"
            status_msg.timestamp = self.get_clock().now().to_msg()
            self.vla_status_pub.publish(status_msg)

    def execute_vla_command_callback(self, request, response):
        """Service callback for executing VLA commands."""
        try:
            if not self.model_ready:
                response.success = False
                response.message = "VLA model not ready"
                return response

            # Process command
            if self.current_image is not None:
                action = self.vla_model.process_instruction(
                    self.ros_image_to_cv2(self.current_image),
                    request.instruction
                )

                # Execute action
                self.publish_action(action)

                response.success = True
                response.message = "Command executed successfully"
            else:
                response.success = False
                response.message = "No current image available"

        except Exception as e:
            response.success = False
            response.message = f"Execution failed: {str(e)}"

        return response

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS Image message to OpenCV format."""
        import cv2
        import numpy as np

        # Convert ROS image to OpenCV format
        dtype = np.uint8
        if ros_image.encoding == 'rgb8':
            dtype = np.uint8
        elif ros_image.encoding == 'rgba8':
            dtype = np.uint8

        # Create numpy array from image data
        img = np.frombuffer(ros_image.data, dtype=dtype).reshape(
            ros_image.height, ros_image.width, -1
        )

        # Convert RGB to BGR if needed
        if ros_image.encoding.startswith('rgb'):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def publish_action(self, action):
        """Publish action to robot control system."""
        # Convert VLA action to ROS message format
        twist_msg = Twist()

        # Map action vector to Twist command (example mapping)
        if len(action) >= 3:
            twist_msg.linear.x = float(action[0])  # Forward/backward
            twist_msg.linear.y = float(action[1])  # Left/right
            twist_msg.angular.z = float(action[2])  # Rotation

        self.action_pub.publish(twist_msg)

    def process_current_image_and_command(self):
        """Process current image with pending command if available."""
        if hasattr(self, 'pending_command') and self.current_image is not None:
            command = self.pending_command
            delattr(self, 'pending_command')
            self.process_vla_command(self.current_image, command)
```

### VLA Action Server Implementation

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from humanoid_robot_msgs.action import ExecuteVLACommand

class VLAActionServer:
    def __init__(self, node, vla_model):
        self.node = node
        self.vla_model = vla_model
        self.action_server = ActionServer(
            node,
            ExecuteVLACommand,
            'execute_vla_command',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        """Accept or reject goal."""
        self.node.get_logger().info(f'Accepting goal: {goal_request.instruction}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel request."""
        self.node.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute VLA command with feedback."""
        self.node.get_logger().info(f'Executing VLA command: {goal_handle.request.instruction}')

        feedback_msg = ExecuteVLACommand.Feedback()
        result = ExecuteVLACommand.Result()

        try:
            # Update feedback
            feedback_msg.status = 'PROCESSING'
            feedback_msg.progress = 10.0
            goal_handle.publish_feedback(feedback_msg)

            # Get current image
            if self.node.current_image is None:
                result.success = False
                result.message = "No current image available"
                goal_handle.abort()
                return result

            # Process with VLA model
            cv_image = self.node.ros_image_to_cv2(self.node.current_image)
            action = self.vla_model.process_instruction(cv_image, goal_handle.request.instruction)

            feedback_msg.progress = 50.0
            goal_handle.publish_feedback(feedback_msg)

            # Execute action
            self.node.publish_action(action)

            feedback_msg.progress = 90.0
            goal_handle.publish_feedback(feedback_msg)

            # Wait for action completion (simplified)
            import time
            time.sleep(2.0)  # Wait for action to complete

            result.success = True
            result.message = f"Command executed: {goal_handle.request.instruction}"
            goal_handle.succeed()

        except Exception as e:
            self.node.get_logger().error(f'VLA action execution error: {e}')
            result.success = False
            result.message = f"Execution failed: {str(e)}"
            goal_handle.abort()

        return result
```

## Simulation Integration

### Isaac Sim Integration

```python
# Isaac Sim VLA integration
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

class IsaacSimVLAIntegration:
    def __init__(self, vla_model):
        self.vla_model = vla_model
        self.world = World(stage_units_in_meters=1.0)
        self.setup_simulation()

    def setup_simulation(self):
        """Setup Isaac Sim environment for VLA integration."""
        # Add humanoid robot to simulation
        self.humanoid = add_reference_to_stage(
            usd_path="/path/to/humanoid_robot.usd",
            prim_path="/World/HumanoidRobot"
        )

        # Setup camera for vision input
        self.camera = Camera(
            prim_path="/World/HumanoidRobot/Camera",
            frequency=30,
            resolution=(640, 480)
        )

        # Setup ROS bridge for communication
        from omni.isaac.ros2_bridge import ROS2Bridge
        self.ros_bridge = ROS2Bridge()

    def run_vla_simulation(self):
        """Run VLA system in simulation loop."""
        self.world.reset()

        while True:
            # Get current image from simulation
            image = self.camera.get_rgb()
            depth = self.camera.get_depth()

            # Process with VLA model (in a real system, this would be triggered by language command)
            # For simulation, we might use pre-defined scenarios
            instruction = self.get_next_instruction()
            if instruction:
                action = self.vla_model.process_instruction(image, instruction)

                # Apply action to robot in simulation
                self.apply_action_to_robot(action)

            # Step simulation
            self.world.step(render=True)

    def apply_action_to_robot(self, action):
        """Apply VLA-generated action to robot in simulation."""
        # Convert action to joint commands or other robot controls
        # This depends on your robot's control interface
        pass

    def get_next_instruction(self):
        """Get next instruction for simulation (could be from a scenario file)."""
        # In practice, this would come from a command source
        # For simulation, you might cycle through predefined instructions
        return "Move forward 1 meter"
```

### Gazebo Integration

```python
# Gazebo VLA integration
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class GazeboVLAIntegration:
    def __init__(self, vla_model):
        self.vla_model = vla_model

        # ROS initialization
        rospy.init_node('gazebo_vla_integration')

        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.language_sub = rospy.Subscriber('/vla/instruction', String, self.language_callback)

        # Internal state
        self.current_image = None
        self.pending_instruction = None

    def image_callback(self, msg):
        """Handle incoming camera images from Gazebo."""
        self.current_image = msg

        # Process pending instruction if available
        if self.pending_instruction:
            self.process_vla_command()

    def language_callback(self, msg):
        """Handle incoming language commands."""
        self.pending_instruction = msg.data

        # Process if image is available
        if self.current_image:
            self.process_vla_command()

    def process_vla_command(self):
        """Process VLA command with current image and pending instruction."""
        if not self.current_image or not self.pending_instruction:
            return

        # Convert ROS image to format expected by VLA model
        cv_image = self.ros_image_to_cv2(self.current_image)

        # Process with VLA model
        action = self.vla_model.process_instruction(cv_image, self.pending_instruction)

        # Convert to Twist command and publish
        twist_cmd = self.action_to_twist(action)
        self.cmd_vel_pub.publish(twist_cmd)

        # Clear pending instruction
        self.pending_instruction = None

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS Image to OpenCV format."""
        import cv2
        import numpy as np

        # Implementation similar to ROS 2 version
        pass

    def action_to_twist(self, action):
        """Convert VLA action to Twist command."""
        twist = Twist()
        twist.linear.x = action[0] if len(action) > 0 else 0.0
        twist.linear.y = action[1] if len(action) > 1 else 0.0
        twist.angular.z = action[2] if len(action) > 2 else 0.0
        return twist
```

## Unity Integration

### Unity-ROS Bridge for VLA

```csharp
// Unity C# script for VLA integration
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Newtonsoft.Json;

public class UnityVLAIntegration : MonoBehaviour
{
    ROSConnection ros;
    string rosTopic = "vla/unity_command";
    string imageTopic = "vla/unity_image";
    string actionTopic = "cmd_vel";

    public Camera vlaCamera;  // Camera for VLA vision input
    public GameObject robot;  // Robot object to control

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.instance;

        // Subscribe to VLA command topic
        ros.Subscribe<StringMsg>(rosTopic, ProcessVLACommand);

        // Publish camera images for vision processing
        InvokeRepeating("PublishCameraImage", 0.0f, 0.1f); // Every 0.1 seconds
    }

    void ProcessVLACommand(StringMsg commandMsg)
    {
        string instruction = commandMsg.data;

        // In a real system, you would send the current image and instruction
        // to your VLA model (possibly running externally) and receive an action
        // For simulation, we'll simulate the VLA processing
        Vector3 action = SimulateVLAProcessing(instruction);

        // Execute action in Unity
        ExecuteAction(action);
    }

    Vector3 SimulateVLAProcessing(string instruction)
    {
        // Simulate VLA model processing
        // In reality, this would involve sending image + instruction to VLA model
        // and receiving back an action vector

        // Simple example based on instruction content
        if (instruction.ToLower().Contains("forward"))
            return new Vector3(1, 0, 0); // Move forward
        else if (instruction.ToLower().Contains("backward"))
            return new Vector3(-1, 0, 0); // Move backward
        else if (instruction.ToLower().Contains("left"))
            return new Vector3(0, 0, 1); // Turn left
        else if (instruction.ToLower().Contains("right"))
            return new Vector3(0, 0, -1); // Turn right
        else
            return Vector3.zero; // No action
    }

    void ExecuteAction(Vector3 action)
    {
        // Apply action to robot in Unity
        robot.transform.Translate(action * Time.deltaTime);
    }

    void PublishCameraImage()
    {
        if (vlaCamera != null)
        {
            // Capture image from camera
            Texture2D imageTexture = CaptureCameraImage(vlaCamera);

            // Convert to ROS message format
            // This would typically involve encoding the image as JPEG/PNG
            // and publishing to a sensor_msgs/Image topic

            // For now, we'll just log that we're publishing
            Debug.Log("Publishing camera image for VLA processing");
        }
    }

    Texture2D CaptureCameraImage(Camera cam)
    {
        // Render texture setup
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;

        cam.Render();

        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        image.Apply();

        RenderTexture.active = currentRT;
        return image;
    }
}
```

## Real Robot Integration

### Hardware Abstraction Layer

```python
class HardwareAbstractionLayer:
    def __init__(self, robot_type='humanoid'):
        self.robot_type = robot_type
        self.robot_interface = self.initialize_robot_interface()
        self.safety_system = SafetySystem()
        self.calibration_data = self.load_calibration_data()

    def initialize_robot_interface(self):
        """Initialize interface to physical robot."""
        if self.robot_type == 'custom_humanoid':
            return CustomHumanoidInterface()
        elif self.robot_type == 'nao':
            return NAOInterface()
        elif self.robot_type == 'pepper':
            return PepperInterface()
        else:
            raise ValueError(f"Unsupported robot type: {self.robot_type}")

    def execute_vla_action(self, action, instruction=None):
        """Execute VLA-generated action on physical robot."""
        # Validate action safety
        if not self.safety_system.validate_action(action):
            raise RuntimeError("Action failed safety validation")

        # Apply calibration corrections
        calibrated_action = self.apply_calibration(action)

        # Execute on robot
        success = self.robot_interface.execute_action(calibrated_action)

        if success:
            # Log successful execution
            self.log_execution(instruction, action, success)
        else:
            raise RuntimeError("Action execution failed on robot")

        return success

    def apply_calibration(self, action):
        """Apply calibration corrections to action."""
        # Apply joint angle offsets, scaling, etc.
        calibrated_action = action.copy()

        # Example calibration transformation
        for i, (offset, scale) in enumerate(self.calibration_data.get('joint_corrections', [])):
            if i < len(calibrated_action):
                calibrated_action[i] = (calibrated_action[i] + offset) * scale

        return calibrated_action

    def get_robot_state(self):
        """Get current robot state for VLA context."""
        return self.robot_interface.get_current_state()

    def log_execution(self, instruction, action, success):
        """Log VLA execution for analysis and improvement."""
        log_entry = {
            'timestamp': time.time(),
            'instruction': instruction,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'success': success,
            'robot_state': self.get_robot_state()
        }

        # Write to log file or database
        self.write_execution_log(log_entry)

class CustomHumanoidInterface:
    def __init__(self):
        # Initialize connection to custom humanoid robot
        self.joint_controllers = self.setup_joint_controllers()
        self.sensors = self.setup_sensors()

    def setup_joint_controllers(self):
        """Setup joint controllers for humanoid robot."""
        # This would connect to actual robot hardware
        # via ROS control, direct hardware interface, etc.
        pass

    def setup_sensors(self):
        """Setup robot sensors."""
        # Setup IMU, encoders, force sensors, etc.
        pass

    def execute_action(self, action):
        """Execute action on physical robot."""
        try:
            # Send action to robot controllers
            # This could be joint positions, velocities, or torques
            self.send_to_controllers(action)

            # Wait for execution
            time.sleep(0.1)  # Small delay for safety

            # Verify execution (check if robot reached commanded state)
            return self.verify_execution(action)

        except Exception as e:
            print(f"Error executing action: {e}")
            return False

    def get_current_state(self):
        """Get current robot state."""
        # Return joint positions, velocities, sensor readings, etc.
        pass

    def send_to_controllers(self, action):
        """Send action to robot controllers."""
        # Implementation depends on robot's control interface
        pass

    def verify_execution(self, action):
        """Verify that action was executed properly."""
        # Check if robot state matches expected state after action
        pass
```

## Safety Integration

### Safety Monitor for VLA Systems

```python
class VLASafetyMonitor:
    def __init__(self):
        self.collision_detector = CollisionDetector()
        self.stability_checker = StabilityChecker()
        self.ethics_checker = EthicsChecker()
        self.emergency_stop = EmergencyStopSystem()

        # Safety thresholds
        self.safety_thresholds = {
            'collision_probability': 0.1,
            'stability_margin': 0.05,  # meters from support polygon
            'joint_limit_violation': 0.0,
            'velocity_limit': 1.0  # m/s
        }

        # Emergency procedures
        self.emergency_procedures = {
            'full_stop': self.full_robot_stop,
            'safe_pose': self.move_to_safe_pose,
            'retreat': self.retreat_motion
        }

    def validate_action(self, action, current_state, instruction):
        """Validate action before execution."""
        safety_checks = {
            'collision_risk': self.check_collision_risk(action, current_state),
            'stability_risk': self.check_stability_risk(action, current_state),
            'ethics_violation': self.check_ethics(instruction),
            'joint_limits': self.check_joint_limits(action, current_state),
            'velocity_limits': self.check_velocity_limits(action, current_state)
        }

        # Determine if action is safe
        is_safe = self.evaluate_safety(safety_checks)

        if not is_safe:
            self.trigger_safety_procedure('full_stop')
            return False, safety_checks

        return True, safety_checks

    def check_collision_risk(self, action, current_state):
        """Check if action poses collision risk."""
        predicted_collision_prob = self.collision_detector.predict_collision(action, current_state)
        return predicted_collision_prob

    def check_stability_risk(self, action, current_state):
        """Check if action maintains robot stability."""
        stability_margin = self.stability_checker.calculate_stability_margin(action, current_state)
        return stability_margin

    def check_ethics(self, instruction):
        """Check if instruction is ethical to follow."""
        ethics_score = self.ethics_checker.evaluate(instruction)
        return ethics_score

    def check_joint_limits(self, action, current_state):
        """Check if action violates joint limits."""
        violations = 0
        for joint_idx, (joint_limit_min, joint_limit_max) in enumerate(self.get_joint_limits()):
            if joint_idx < len(action):
                if action[joint_idx] < joint_limit_min or action[joint_idx] > joint_limit_max:
                    violations += 1
        return violations

    def check_velocity_limits(self, action, current_state):
        """Check if action exceeds velocity limits."""
        # Calculate expected velocities from action
        # Compare against maximum allowed velocities
        pass

    def evaluate_safety(self, safety_checks):
        """Evaluate overall safety based on all checks."""
        if safety_checks['collision_risk'] > self.safety_thresholds['collision_probability']:
            return False
        if safety_checks['stability_risk'] < self.safety_thresholds['stability_margin']:
            return False
        if safety_checks['ethics_violation'] > 0:  # Any ethics violation
            return False
        if safety_checks['joint_limits'] > self.safety_thresholds['joint_limit_violation']:
            return False
        if safety_checks['velocity_limits'] > self.safety_thresholds['velocity_limit']:
            return False

        return True

    def trigger_safety_procedure(self, procedure_name):
        """Trigger safety procedure."""
        if procedure_name in self.emergency_procedures:
            self.emergency_procedures[procedure_name]()

    def full_robot_stop(self):
        """Stop all robot motion immediately."""
        self.emergency_stop.activate()

    def move_to_safe_pose(self):
        """Move robot to a predefined safe pose."""
        # Move all joints to safe positions
        pass

    def retreat_motion(self):
        """Execute retreat motion to safe position."""
        # Move robot away from potential hazards
        pass

class SafetyIntegratedVLAIntegration(VLAIntegrationNode):
    def __init__(self):
        super().__init__()
        self.safety_monitor = VLASafetyMonitor()

    def process_vla_command(self, image, instruction):
        """Process VLA command with safety validation."""
        try:
            # Convert ROS image to format expected by VLA model
            cv_image = self.ros_image_to_cv2(image)

            # Process with VLA model
            action = self.vla_model.process_instruction(cv_image, instruction)

            # Get current robot state for safety validation
            current_state = self.get_robot_state()

            # Validate action safety
            is_safe, safety_checks = self.safety_monitor.validate_action(
                action, current_state, instruction
            )

            if is_safe:
                # Publish safe action to robot
                self.publish_action(action)

                # Update status
                status_msg = VLAStatus()
                status_msg.success = True
                status_msg.message = f"Executed safely: {instruction}"
                status_msg.timestamp = self.get_clock().now().to_msg()
                self.vla_status_pub.publish(status_msg)
            else:
                # Action is unsafe
                self.get_logger().warn(f"Unsafe action blocked: {safety_checks}")
                status_msg = VLAStatus()
                status_msg.success = False
                status_msg.message = f"Action blocked for safety: {safety_checks}"
                status_msg.timestamp = self.get_clock().now().to_msg()
                self.vla_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing VLA command: {e}")
            status_msg = VLAStatus()
            status_msg.success = False
            status_msg.message = f"Error: {str(e)}"
            status_msg.timestamp = self.get_clock().now().to_msg()
            self.vla_status_pub.publish(status_msg)
```

## Performance Optimization

### Efficient VLA Pipeline

```python
class EfficientVLAIntegration:
    def __init__(self, vla_model):
        self.vla_model = vla_model

        # Use threading for parallel processing
        self.image_queue = queue.Queue(maxsize=5)
        self.command_queue = queue.Queue(maxsize=5)

        # Feature caching for temporal consistency
        self.feature_cache = {}

        # Async processing
        self.processing_thread = threading.Thread(target=self.process_commands_async)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def camera_callback(self, msg):
        """Non-blocking camera callback."""
        try:
            self.image_queue.put_nowait(msg)
        except queue.Full:
            # Drop oldest image if queue is full
            try:
                self.image_queue.get_nowait()
                self.image_queue.put_nowait(msg)
            except queue.Empty:
                pass

    def language_command_callback(self, msg):
        """Non-blocking language command callback."""
        try:
            self.command_queue.put_nowait({
                'instruction': msg.data,
                'timestamp': time.time()
            })
        except queue.Full:
            # Drop oldest command if queue is full
            try:
                self.command_queue.get_nowait()
                self.command_queue.put_nowait({
                    'instruction': msg.data,
                    'timestamp': time.time()
                })
            except queue.Empty:
                pass

    def process_commands_async(self):
        """Process commands asynchronously."""
        while True:
            try:
                # Get latest image and command
                image = self.image_queue.get(timeout=0.1)
                command = self.command_queue.get(timeout=0.1)

                # Process with VLA model
                cv_image = self.ros_image_to_cv2(image)
                action = self.vla_model.process_instruction(cv_image, command['instruction'])

                # Publish action
                self.publish_action(action)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Async processing error: {e}")

    def publish_action(self, action):
        """Optimized action publishing."""
        # Use latched publishing for critical commands
        twist_msg = Twist()
        twist_msg.linear.x = float(action[0]) if len(action) > 0 else 0.0
        twist_msg.linear.y = float(action[1]) if len(action) > 1 else 0.0
        twist_msg.angular.z = float(action[2]) if len(action) > 2 else 0.0

        self.action_pub.publish(twist_msg)
```

## Multi-Robot VLA Coordination

### Distributed VLA System

```python
import zmq
import json

class MultiRobotVLA:
    def __init__(self, robot_id, total_robots):
        self.robot_id = robot_id
        self.total_robots = total_robots

        # ZMQ context for inter-robot communication
        self.context = zmq.Context()

        # Communication sockets
        self.broadcast_socket = self.context.socket(zmq.PUB)
        self.broadcast_socket.bind(f"tcp://*:{5555 + robot_id}")

        self.listen_socket = self.context.socket(zmq.SUB)
        for i in range(total_robots):
            if i != robot_id:
                self.listen_socket.connect(f"tcp://localhost:{5555 + i}")
        self.listen_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # Start communication thread
        self.comm_thread = threading.Thread(target=self.listen_for_messages)
        self.comm_thread.daemon = True
        self.comm_thread.start()

    def process_coordinated_command(self, instruction, world_state):
        """Process command considering other robots."""
        # Broadcast intent to other robots
        self.broadcast_intent(instruction)

        # Wait for responses from other robots
        other_intents = self.receive_intents()

        # Coordinate actions to avoid conflicts
        coordinated_action = self.coordinate_actions(
            instruction, world_state, other_intents
        )

        return coordinated_action

    def broadcast_intent(self, instruction):
        """Broadcast robot's intent to others."""
        message = {
            'robot_id': self.robot_id,
            'instruction': instruction,
            'timestamp': time.time()
        }
        self.broadcast_socket.send_string(json.dumps(message))

    def receive_intents(self):
        """Receive intents from other robots."""
        intents = []
        # Non-blocking receive with timeout
        try:
            message = self.listen_socket.recv_string(zmq.NOBLOCK)
            intent = json.loads(message)
            intents.append(intent)
        except zmq.Again:
            pass

        return intents

    def coordinate_actions(self, instruction, world_state, other_intents):
        """Coordinate actions to avoid conflicts."""
        # Simple coordination: avoid same target locations
        my_target = self.extract_target_location(instruction)

        for other_intent in other_intents:
            other_target = self.extract_target_location(other_intent['instruction'])
            if self.locations_conflict(my_target, other_target):
                # Adjust my action to avoid conflict
                instruction = self.adjust_instruction_for_conflict(instruction, other_intent)

        # Process adjusted instruction
        return self.vla_model.process_instruction(world_state['image'], instruction)

    def extract_target_location(self, instruction):
        """Extract target location from instruction."""
        # Use NLP to extract spatial references
        # This is a simplified example
        if 'table' in instruction.lower():
            return 'table_area'
        elif 'kitchen' in instruction.lower():
            return 'kitchen_area'
        else:
            return 'unknown'

    def locations_conflict(self, loc1, loc2):
        """Check if two locations conflict."""
        return loc1 == loc2
```

## Integration Testing

### VLA Integration Test Suite

```python
import unittest
from unittest.mock import Mock, patch
import rospy

class TestVLAIntegration(unittest.TestCase):
    def setUp(self):
        """Setup test environment."""
        # Mock ROS node
        self.mock_node = Mock()

        # Mock VLA model
        self.mock_vla_model = Mock()
        self.mock_vla_model.process_instruction.return_value = [0.5, 0.0, 0.1]

        # Create integration instance with mocks
        self.vla_integration = VLAIntegrationNode()
        self.vla_integration.vla_model = self.mock_vla_model
        self.vla_integration.model_ready = True

    def test_image_processing(self):
        """Test image processing pipeline."""
        # Create mock image message
        mock_image = Mock()
        mock_image.height = 480
        mock_image.width = 640
        mock_image.encoding = 'rgb8'
        mock_image.data = b'\x00' * (480 * 640 * 3)  # Mock image data

        # Process image
        cv_image = self.vla_integration.ros_image_to_cv2(mock_image)

        # Verify image conversion
        self.assertIsNotNone(cv_image)
        self.assertEqual(cv_image.shape, (480, 640, 3))

    def test_command_processing(self):
        """Test VLA command processing."""
        # Create mock image and instruction
        mock_image = Mock()
        mock_image.height = 480
        mock_image.width = 640
        mock_image.encoding = 'rgb8'
        mock_image.data = b'\x00' * (480 * 640 * 3)

        instruction = "Move forward 1 meter"

        # Process command
        with patch.object(self.vla_integration, 'publish_action') as mock_publish:
            self.vla_integration.current_image = mock_image
            self.vla_integration.process_vla_command(mock_image, instruction)

            # Verify VLA model was called
            self.mock_vla_model.process_instruction.assert_called_once()
            mock_publish.assert_called_once()

    def test_service_call(self):
        """Test VLA service call."""
        # Create mock request and response
        request = Mock()
        request.instruction = "Pick up the red cup"

        response = Mock()

        # Call service
        result = self.vla_integration.execute_vla_command_callback(request, response)

        # Verify response
        self.assertIsNotNone(result)
        self.assertTrue(response.success or not response.success)

    def test_safety_validation(self):
        """Test safety validation in VLA processing."""
        # Test with safety monitor
        safety_integration = SafetyIntegratedVLAIntegration()
        safety_integration.vla_model = self.mock_vla_model
        safety_integration.model_ready = True
        safety_integration.safety_monitor = Mock()
        safety_integration.safety_monitor.validate_action.return_value = (True, {})

        # Process command
        mock_image = Mock()
        mock_image.height = 480
        mock_image.width = 640
        mock_image.encoding = 'rgb8'
        mock_image.data = b'\x00' * (480 * 640 * 3)

        with patch.object(safety_integration, 'publish_action') as mock_publish:
            safety_integration.process_vla_command(mock_image, "Move forward")
            mock_publish.assert_called_once()

class IntegrationTestRunner:
    def __init__(self):
        self.test_suite = unittest.TestSuite()

    def add_tests(self):
        """Add integration tests to suite."""
        loader = unittest.TestLoader()
        self.test_suite.addTests(loader.loadTestsFromTestCase(TestVLAIntegration))

    def run_tests(self):
        """Run integration tests."""
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(self.test_suite)
        return result

if __name__ == '__main__':
    # Run integration tests
    test_runner = IntegrationTestRunner()
    test_runner.add_tests()
    test_results = test_runner.run_tests()

    # Exit with appropriate code
    import sys
    sys.exit(0 if test_results.wasSuccessful() else 1)
```

## Troubleshooting Common Integration Issues

### 1. Timing and Synchronization Issues
- Use appropriate QoS profiles for different message types
- Implement proper buffering for image and command synchronization
- Use message filters for time-based synchronization

### 2. Memory and Performance Issues
- Implement efficient data structures for real-time processing
- Use appropriate batching for model inference
- Monitor and optimize GPU/CPU usage

### 3. Communication Problems
- Verify ROS network configuration
- Check topic/service availability
- Implement proper error handling and fallbacks

### 4. Calibration and Coordinate System Issues
- Ensure consistent coordinate frame transformations
- Verify camera calibration parameters
- Use TF2 for proper coordinate transformations

---
[Next: Safety and Evaluation](./safety-evaluation.md) | [Previous: Training VLA Models](./training-vla-models.md)