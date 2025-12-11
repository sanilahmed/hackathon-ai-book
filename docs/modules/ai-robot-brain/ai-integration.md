# AI Integration

## Overview

This section covers the integration of AI components with ROS 2 and the digital twin established in previous modules. AI integration involves connecting perception, planning, control, and learning systems into a cohesive AI robot brain that can operate the humanoid robot effectively.

## Architecture Overview

### AI Robot Brain Architecture

The AI robot brain follows a hierarchical architecture:

```
┌─────────────────────────────────────────┐
│           AI Robot Brain               │
├─────────────────────────────────────────┤
│  ┌─────────────┐ ┌──────────────────┐   │
│  │ Perception  │ │ Planning &       │   │
│  │ System      │ │ Control          │   │
│  └─────────────┘ └──────────────────┘   │
│         │                   │           │
│         ▼                   ▼           │
│  ┌───────────────────────────────────┐   │
│  │        Decision Making            │   │
│  │      (Behavior Trees, RL)         │   │
│  └───────────────────────────────────┘   │
│                   │                     │
│                   ▼                     │
│  ┌───────────────────────────────────┐   │
│  │         Action Execution          │   │
│  │      (Motor Commands, etc.)       │   │
│  └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Integration with ROS 2 Ecosystem

The AI brain integrates with ROS 2 through:

- **Action Servers**: Long-running tasks with feedback
- **Services**: Synchronous request-response communication
- **Topics**: Asynchronous data streaming
- **Parameters**: Configuration management

## ROS 2 Action Integration

### AI Task Action Server

```python
# AI task action server for complex humanoid behaviors
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from humanoid_robot_msgs.action import ExecuteAITask
from humanoid_robot_msgs.msg import RobotState, TaskStatus
from std_msgs.msg import String

class AITaskActionServer(Node):
    def __init__(self):
        super().__init__('ai_task_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            ExecuteAITask,
            'execute_ai_task',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Publishers for monitoring
        self.task_status_pub = self.create_publisher(TaskStatus, 'task_status', 10)
        self.robot_state_sub = self.create_subscription(
            RobotState, 'robot_state', self.robot_state_callback, 10
        )

        # AI components
        self.perception_system = PerceptionSystem(self)
        self.planning_system = PlanningSystem(self)
        self.control_system = ControlSystem(self)

    def goal_callback(self, goal_request):
        """Accept or reject goal."""
        self.get_logger().info(f'Accepting goal: {goal_request.task_type}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute AI task with feedback."""
        self.get_logger().info(f'Executing task: {goal_handle.request.task_type}')

        feedback_msg = ExecuteAITask.Feedback()
        result = ExecuteAITask.Result()

        try:
            # Initialize task
            task_type = goal_handle.request.task_type
            task_params = goal_handle.request.parameters

            # Process task through AI pipeline
            success = await self.process_task(
                task_type, task_params, feedback_msg, goal_handle
            )

            if success:
                result.success = True
                result.message = f'Task {task_type} completed successfully'
                goal_handle.succeed()
            else:
                result.success = False
                result.message = f'Task {task_type} failed'
                goal_handle.abort()

        except Exception as e:
            self.get_logger().error(f'Task execution error: {e}')
            result.success = False
            result.message = f'Task failed with error: {str(e)}'
            goal_handle.abort()

        return result

    async def process_task(self, task_type, params, feedback_msg, goal_handle):
        """Process AI task through perception-planning-control pipeline."""
        # Update task status
        status_msg = TaskStatus()
        status_msg.task_type = task_type
        status_msg.status = 'PROCESSING'
        self.task_status_pub.publish(status_msg)

        # 1. Perception phase
        if goal_handle.is_cancel_requested:
            return False

        perception_result = await self.perception_system.process_environment()
        feedback_msg.current_phase = 'PERCEPTION'
        feedback_msg.percentage_complete = 25.0
        goal_handle.publish_feedback(feedback_msg)

        # 2. Planning phase
        if goal_handle.is_cancel_requested:
            return False

        plan = await self.planning_system.generate_plan(
            task_type, perception_result, params
        )
        feedback_msg.current_phase = 'PLANNING'
        feedback_msg.percentage_complete = 50.0
        goal_handle.publish_feedback(feedback_msg)

        # 3. Execution phase
        if goal_handle.is_cancel_requested:
            return False

        execution_result = await self.control_system.execute_plan(plan)
        feedback_msg.current_phase = 'EXECUTION'
        feedback_msg.percentage_complete = 75.0
        goal_handle.publish_feedback(feedback_msg)

        # 4. Verification phase
        if goal_handle.is_cancel_requested:
            return False

        verification_result = await self.verify_task_completion(
            task_type, execution_result
        )
        feedback_msg.percentage_complete = 100.0
        goal_handle.publish_feedback(feedback_msg)

        return verification_result

    def verify_task_completion(self, task_type, execution_result):
        """Verify that task was completed successfully."""
        # Implementation depends on task type
        # Check final state, sensor data, etc.
        return execution_result.success
```

### Perception System Integration

```python
# Perception system integrated with ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Header

class PerceptionSystem(Node):
    def __init__(self, parent_node):
        super().__init__('perception_system')
        self.parent_node = parent_node

        # Subscribers for sensor data
        self.image_sub = self.create_subscription(
            Image, '/humanoid/camera/image_raw', self.image_callback, 10
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/humanoid/scan', self.pointcloud_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/humanoid/imu', self.imu_callback, 10
        )

        # Publishers for processed data
        self.object_detection_pub = self.create_publisher(
            MarkerArray, '/detected_objects', 10
        )
        self.environment_map_pub = self.create_publisher(
            OccupancyGrid, '/environment_map', 10
        )

        # AI perception models
        self.object_detector = self.load_object_detector()
        self.segmentation_model = self.load_segmentation_model()
        self.slam_system = self.initialize_slam_system()

    def image_callback(self, msg):
        """Process camera image for object detection and segmentation."""
        # Convert ROS Image to OpenCV format
        cv_image = self.ros_image_to_cv2(msg)

        # Run object detection
        detections = self.object_detector.detect(cv_image)

        # Run semantic segmentation
        segmentation = self.segmentation_model.segment(cv_image)

        # Publish results
        self.publish_detections(detections, msg.header)

    def pointcloud_callback(self, msg):
        """Process point cloud for environment mapping."""
        # Convert PointCloud2 to numpy array
        points = self.pointcloud2_to_array(msg)

        # Build environment map
        occupancy_grid = self.slam_system.process_scan(points)

        # Publish map
        self.environment_map_pub.publish(occupancy_grid)

    def process_environment(self):
        """Process current environment state for AI decision making."""
        # Get latest sensor data
        latest_image = self.get_latest_image()
        latest_pointcloud = self.get_latest_pointcloud()
        latest_imu = self.get_latest_imu()

        # Process all sensor data
        environment_state = {
            'objects': self.detect_objects(latest_image),
            'obstacles': self.map_obstacles(latest_pointcloud),
            'orientation': self.get_orientation(latest_imu),
            'map': self.get_current_map(),
            'robot_pose': self.get_robot_pose()
        }

        return environment_state

    def detect_objects(self, image):
        """Detect objects in image using AI model."""
        # Run object detection model
        detections = self.object_detector(image)

        objects = []
        for detection in detections:
            obj = {
                'class': detection.class_name,
                'confidence': detection.confidence,
                'bbox': detection.bbox,
                'position': self.calculate_3d_position(detection)
            }
            objects.append(obj)

        return objects

    def map_obstacles(self, pointcloud):
        """Map obstacles from point cloud data."""
        # Process point cloud to identify obstacles
        obstacles = []
        for point in pointcloud:
            if self.is_obstacle_point(point):
                obstacle = {
                    'position': point[:3],
                    'size': self.estimate_obstacle_size(point)
                }
                obstacles.append(obstacle)

        return obstacles
```

## Planning System Integration

### AI Planning with ROS 2

```python
# AI planning system integrated with ROS 2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path
from humanoid_robot_msgs.srv import PlanPath

class PlanningSystem(Node):
    def __init__(self, parent_node):
        super().__init__('planning_system')
        self.parent_node = parent_node

        # Service server for path planning
        self.plan_path_service = self.create_service(
            PlanPath, 'plan_path', self.plan_path_callback
        )

        # Publishers for visualization
        self.global_plan_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)

        # AI planning components
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner()
        self.trajectory_optimizer = TrajectoryOptimizer()

    def plan_path_callback(self, request, response):
        """Handle path planning service request."""
        try:
            # Plan global path
            global_path = self.global_planner.plan(
                request.start_pose, request.goal_pose, request.map
            )

            # Optimize trajectory
            optimized_trajectory = self.trajectory_optimizer.optimize(
                global_path, request.constraints
            )

            # Set response
            response.path = optimized_trajectory
            response.success = True
            response.message = 'Path planning successful'

        except Exception as e:
            response.success = False
            response.message = f'Path planning failed: {str(e)}'

        return response

    async def generate_plan(self, task_type, environment_state, task_params):
        """Generate plan for specific task using AI planning."""
        if task_type == 'navigation':
            return await self.plan_navigation(environment_state, task_params)
        elif task_type == 'manipulation':
            return await self.plan_manipulation(environment_state, task_params)
        elif task_type == 'locomotion':
            return await self.plan_locomotion(environment_state, task_params)
        else:
            raise ValueError(f'Unknown task type: {task_type}')

    async def plan_navigation(self, env_state, params):
        """Plan navigation task."""
        start_pose = params.get('start_pose', env_state['robot_pose'])
        goal_pose = params['goal_pose']
        map_data = env_state['map']

        # Plan global path
        global_path = self.global_planner.plan(start_pose, goal_pose, map_data)

        # Generate local plans for execution
        local_plans = []
        for i in range(0, len(global_path.poses), 10):  # Every 10 waypoints
            local_plan = self.local_planner.plan(
                global_path.poses[i:i+20],  # Next 20 waypoints
                env_state['obstacles']
            )
            local_plans.append(local_plan)

        return {
            'global_path': global_path,
            'local_plans': local_plans,
            'execution_strategy': 'follow_path'
        }

    async def plan_manipulation(self, env_state, params):
        """Plan manipulation task."""
        target_object = params['target_object']
        grasp_pose = params.get('grasp_pose')

        # Find object in environment
        object_info = self.find_object_in_environment(env_state, target_object)

        if not object_info:
            raise ValueError(f'Target object {target_object} not found')

        # Plan grasp trajectory
        grasp_plan = self.plan_grasp_trajectory(
            object_info['position'],
            object_info['orientation']
        )

        # Plan approach and retreat
        approach_plan = self.plan_approach_trajectory(
            grasp_plan, env_state['obstacles']
        )

        return {
            'grasp_plan': grasp_plan,
            'approach_plan': approach_plan,
            'execution_strategy': 'grasp_and_place'
        }
```

## Control System Integration

### AI Control with ROS 2

```python
# AI control system integrated with ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from humanoid_robot_msgs.srv import ExecuteTrajectory

class ControlSystem(Node):
    def __init__(self, parent_node):
        super().__init__('control_system')
        self.parent_node = parent_node

        # Publishers for control commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_group_position_controller/commands', 10
        )
        self.velocity_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # Subscribers for state feedback
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Service for trajectory execution
        self.execute_traj_service = self.create_service(
            ExecuteTrajectory, 'execute_trajectory', self.execute_trajectory_callback
        )

        # AI control components
        self.impedance_controller = ImpedanceController()
        self.model_predictive_controller = ModelPredictiveController()
        self.reinforcement_learning_controller = RLController()

        # Current robot state
        self.current_joint_states = None
        self.current_pose = None

    def joint_state_callback(self, msg):
        """Update current joint states."""
        self.current_joint_states = msg

    def execute_trajectory_callback(self, request, response):
        """Execute trajectory service callback."""
        try:
            # Execute trajectory using AI controller
            success = self.execute_trajectory_with_ai(
                request.trajectory, request.execution_params
            )

            response.success = success
            response.message = 'Trajectory execution completed' if success else 'Execution failed'

        except Exception as e:
            response.success = False
            response.message = f'Execution failed: {str(e)}'

        return response

    async def execute_plan(self, plan):
        """Execute plan using AI control system."""
        execution_strategy = plan.get('execution_strategy', 'default')

        if execution_strategy == 'follow_path':
            return await self.execute_navigation_plan(plan)
        elif execution_strategy == 'grasp_and_place':
            return await self.execute_manipulation_plan(plan)
        elif execution_strategy == 'locomotion':
            return await self.execute_locomotion_plan(plan)
        else:
            return await self.execute_default_plan(plan)

    async def execute_navigation_plan(self, plan):
        """Execute navigation plan."""
        local_plans = plan['local_plans']
        execution_results = []

        for local_plan in local_plans:
            # Execute local plan segment
            result = await self.follow_path_segment(local_plan)
            execution_results.append(result)

            # Check for replanning needs
            if not result.success:
                # Replan based on new environment state
                new_plan = await self.replan_navigation(plan)
                return await self.execute_plan(new_plan)

        return {
            'success': all(r.success for r in execution_results),
            'execution_results': execution_results
        }

    async def follow_path_segment(self, path_segment):
        """Follow a path segment using AI control."""
        # Use MPC controller for path following
        for waypoint in path_segment.poses:
            # Calculate control command
            control_cmd = self.model_predictive_controller.compute_control(
                self.current_pose, waypoint.pose
            )

            # Execute control command
            self.publish_velocity_command(control_cmd)

            # Wait for execution
            await self.wait_for_pose_reach(waypoint.pose)

        return {'success': True, 'reached_waypoint': True}

    def publish_velocity_command(self, cmd):
        """Publish velocity command to robot."""
        twist_msg = Twist()
        twist_msg.linear.x = cmd[0]
        twist_msg.linear.y = cmd[1]
        twist_msg.angular.z = cmd[2]
        self.velocity_cmd_pub.publish(twist_msg)

    def wait_for_pose_reach(self, target_pose, timeout=10.0):
        """Wait for robot to reach target pose."""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_pose_reached(target_pose):
                return True
            time.sleep(0.1)

        return False

    def is_pose_reached(self, target_pose, tolerance=0.1):
        """Check if target pose is reached."""
        if self.current_pose is None:
            return False

        # Calculate distance to target
        distance = self.calculate_pose_distance(self.current_pose, target_pose)
        return distance < tolerance
```

## Deep Learning Model Integration

### TensorRT Integration with ROS 2

```python
# TensorRT integration for optimized AI inference
import rclpy
from rclpy.node import Node
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInferenceNode(Node):
    def __init__(self):
        super().__init__('tensorrt_inference_node')

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load optimized models
        self.object_detection_engine = self.load_engine('object_detection.plan')
        self.segmentation_engine = self.load_engine('segmentation.plan')
        self.control_policy_engine = self.load_engine('control_policy.plan')

        # CUDA streams for parallel inference
        self.stream = cuda.Stream()

    def load_engine(self, engine_path):
        """Load TensorRT engine from file."""
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = self.runtime.deserialize_cuda_engine(engine_data)
        return engine

    def run_inference(self, engine, input_data):
        """Run inference on TensorRT engine."""
        # Create execution context
        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings, stream = self.allocate_buffers(engine)

        # Copy input data to GPU
        cuda.memcpy_htod_async(inputs[0].data, input_data, stream)

        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy output data from GPU
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].data, stream)
        stream.synchronize()

        return outputs[0].host

    def allocate_buffers(self, engine):
        """Allocate input/output buffers for TensorRT."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append(DeviceBinding(device_mem, host_mem))
            else:
                outputs.append(DeviceBinding(device_mem, host_mem))

        return inputs, outputs, bindings, stream

class DeviceBinding:
    def __init__(self, device_mem, host_mem):
        self.data = device_mem
        self.host = host_mem
```

## Isaac Integration

### Isaac ROS Bridge Integration

```python
# Isaac ROS bridge integration
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

class IsaacROSIntegration(Node):
    def __init__(self):
        super().__init__('isaac_ros_integration')

        # Isaac publishers
        self.isaac_image_pub = self.create_publisher(Image, '/isaac/camera/image_raw', 10)
        self.isaac_imu_pub = self.create_publisher(Imu, '/isaac/imu', 10)
        self.isaac_odom_pub = self.create_publisher(Odometry, '/isaac/odom', 10)

        # Isaac subscribers
        self.isaac_cmd_sub = self.create_subscription(
            Twist, '/isaac/cmd_vel', self.isaac_cmd_callback, 10
        )

        # AI brain subscribers (from ROS 2 ecosystem)
        self.ai_cmd_sub = self.create_subscription(
            Twist, '/ai/cmd_vel', self.ai_cmd_callback, 10
        )

        # Initialize Isaac components
        self.initialize_isaac_components()

    def initialize_isaac_components(self):
        """Initialize Isaac-specific components."""
        # Connect to Isaac Sim
        from omni.isaac.ros2_bridge import ROS2Bridge
        self.isaac_bridge = ROS2Bridge()

        # Initialize Isaac sensors
        self.setup_isaac_sensors()

        # Initialize Isaac controllers
        self.setup_isaac_controllers()

    def isaac_cmd_callback(self, msg):
        """Handle commands from Isaac Sim."""
        # Process Isaac commands
        self.execute_isaac_command(msg)

    def ai_cmd_callback(self, msg):
        """Handle commands from AI brain."""
        # Forward AI commands to Isaac
        self.send_command_to_isaac(msg)

    def setup_isaac_sensors(self):
        """Setup Isaac sensors and bridge to ROS 2."""
        # Setup camera bridge
        self.isaac_bridge.create_camera_bridge(
            camera_topic='/isaac/camera/image_raw',
            camera_info_topic='/isaac/camera/camera_info'
        )

        # Setup IMU bridge
        self.isaac_bridge.create_imu_bridge(
            imu_topic='/isaac/imu'
        )

        # Setup LiDAR bridge
        self.isaac_bridge.create_lidar_bridge(
            lidar_topic='/isaac/scan'
        )

    def setup_isaac_controllers(self):
        """Setup Isaac controllers."""
        # Setup joint controllers
        self.isaac_bridge.create_joint_command_bridge(
            joint_command_topic='/isaac/joint_commands'
        )

        # Setup differential drive controller
        self.isaac_bridge.create_diff_drive_bridge(
            cmd_vel_topic='/isaac/cmd_vel',
            odom_topic='/isaac/odom'
        )

    def synchronize_with_isaac(self):
        """Synchronize AI brain with Isaac simulation."""
        # Get Isaac simulation time
        isaac_time = self.get_isaac_simulation_time()

        # Set ROS 2 time to match Isaac
        self.set_use_sim_time(True)
        self.set_simulation_time(isaac_time)

        # Synchronize transforms
        self.synchronize_transforms()
```

## Digital Twin Synchronization

### Multi-Environment Synchronization

```python
# Digital twin synchronization between Gazebo, Unity, and Isaac
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Time

class DigitalTwinSynchronizer(Node):
    def __init__(self):
        super().__init__('digital_twin_synchronizer')

        # Publishers for all environments
        self.gazebo_joint_pub = self.create_publisher(JointState, '/gazebo/joint_states', 10)
        self.unity_joint_pub = self.create_publisher(JointState, '/unity/joint_states', 10)
        self.isaac_joint_pub = self.create_publisher(JointState, '/isaac/joint_states', 10)

        # Subscribers for synchronization
        self.main_joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # TF broadcasters for each environment
        self.gazebo_broadcaster = TFMessage()
        self.unity_broadcaster = TFMessage()
        self.isaac_broadcaster = TFMessage()

        # Synchronization parameters
        self.sync_frequency = 50  # Hz
        self.sync_timer = self.create_timer(1.0/self.sync_frequency, self.synchronize_all)

    def joint_state_callback(self, msg):
        """Receive joint states from main system."""
        self.current_joint_states = msg

    def synchronize_all(self):
        """Synchronize all digital twin environments."""
        if hasattr(self, 'current_joint_states'):
            # Publish to Gazebo
            self.gazebo_joint_pub.publish(self.current_joint_states)

            # Publish to Unity
            self.unity_joint_pub.publish(self.current_joint_states)

            # Publish to Isaac
            self.isaac_joint_pub.publish(self.current_joint_states)

            # Synchronize transforms
            self.synchronize_transforms()

    def synchronize_transforms(self):
        """Synchronize transforms across all environments."""
        # Get current transforms
        transforms = self.get_current_transforms()

        # Create TF messages for each environment
        gazebo_tf_msg = self.create_environment_tf(transforms, 'gazebo')
        unity_tf_msg = self.create_environment_tf(transforms, 'unity')
        isaac_tf_msg = self.create_environment_tf(transforms, 'isaac')

        # Publish transforms to each environment
        self.publish_environment_tf(gazebo_tf_msg, 'gazebo')
        self.publish_environment_tf(unity_tf_msg, 'unity')
        self.publish_environment_tf(isaac_tf_msg, 'isaac')

    def create_environment_tf(self, transforms, environment):
        """Create TF message for specific environment."""
        tf_msg = TFMessage()
        for transform in transforms:
            # Modify frame IDs for environment-specific naming
            env_transform = TransformStamped()
            env_transform.header = transform.header
            env_transform.header.frame_id = f'{environment}_{transform.header.frame_id}'
            env_transform.child_frame_id = f'{environment}_{transform.child_frame_id}'
            env_transform.transform = transform.transform
            tf_msg.transforms.append(env_transform)

        return tf_msg

    def publish_environment_tf(self, tf_msg, environment):
        """Publish TF message to specific environment."""
        # Use environment-specific TF publisher
        if environment == 'gazebo':
            self.gazebo_tf_pub.publish(tf_msg)
        elif environment == 'unity':
            self.unity_tf_pub.publish(tf_msg)
        elif environment == 'isaac':
            self.isaac_tf_pub.publish(tf_msg)
```

## Performance Optimization

### Multi-Threaded AI Processing

```python
# Multi-threaded AI processing for real-time performance
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import threading
import queue
import asyncio

class MultiThreadedAIProcessor(Node):
    def __init__(self):
        super().__init__('multi_threaded_ai_processor')

        # Create callback groups for threading
        self.perception_group = MutuallyExclusiveCallbackGroup()
        self.planning_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()

        # Queues for inter-thread communication
        self.perception_queue = queue.Queue()
        self.planning_queue = queue.Queue()
        self.control_queue = queue.Queue()

        # Initialize AI components
        self.perception_worker = PerceptionWorker(self.perception_queue)
        self.planning_worker = PlanningWorker(self.planning_queue)
        self.control_worker = ControlWorker(self.control_queue)

        # Start worker threads
        self.perception_thread = threading.Thread(target=self.perception_worker.run)
        self.planning_thread = threading.Thread(target=self.planning_worker.run)
        self.control_thread = threading.Thread(target=self.control_worker.run)

        self.perception_thread.start()
        self.planning_thread.start()
        self.control_thread.start()

    def process_sensor_data_async(self, sensor_data):
        """Process sensor data asynchronously."""
        self.perception_queue.put(sensor_data)

    def get_perception_result(self):
        """Get perception result from queue."""
        try:
            return self.perception_queue.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self):
        """Clean shutdown of worker threads."""
        self.perception_worker.stop()
        self.planning_worker.stop()
        self.control_worker.stop()

        self.perception_thread.join()
        self.planning_thread.join()
        self.control_thread.join()

class PerceptionWorker:
    def __init__(self, queue):
        self.queue = queue
        self.running = True
        self.model = self.load_perception_model()

    def load_perception_model(self):
        """Load optimized perception model."""
        # Load TensorRT optimized model
        return TensorRTInferenceNode()

    def run(self):
        """Main worker loop."""
        while self.running:
            try:
                # Get sensor data from queue
                sensor_data = self.queue.get(timeout=0.1)

                # Process with AI model
                result = self.model.process(sensor_data)

                # Put result back in queue
                self.queue.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Perception worker error: {e}')

    def stop(self):
        """Stop the worker."""
        self.running = False
```

## Safety and Monitoring

### AI Safety Monitor

```python
# AI safety monitoring system
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from humanoid_robot_msgs.msg import SafetyStatus
from geometry_msgs.msg import Twist
import numpy as np

class AISafetyMonitor(Node):
    def __init__(self):
        super().__init__('ai_safety_monitor')

        # Publishers for safety status
        self.safety_status_pub = self.create_publisher(SafetyStatus, '/safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Subscribers for monitoring
        self.robot_state_sub = self.create_subscription(
            RobotState, '/robot_state', self.robot_state_callback, 10
        )
        self.command_sub = self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10
        )

        # Safety parameters
        self.safety_thresholds = {
            'max_velocity': 2.0,      # m/s
            'max_angular_velocity': 1.0,  # rad/s
            'max_torque': 100.0,      # Nm
            'max_acceleration': 5.0,  # m/s²
            'com_stability': 0.1      # m from support polygon
        }

        # Safety monitoring timer
        self.safety_timer = self.create_timer(0.1, self.check_safety)

        # Emergency stop flag
        self.emergency_stop_active = False

    def robot_state_callback(self, msg):
        """Monitor robot state for safety violations."""
        self.current_state = msg

    def command_callback(self, msg):
        """Monitor commands for safety violations."""
        self.current_command = msg

    def check_safety(self):
        """Check for safety violations."""
        if self.emergency_stop_active:
            return

        safety_status = SafetyStatus()
        safety_status.timestamp = self.get_clock().now().to_msg()

        # Check velocity limits
        if hasattr(self, 'current_command'):
            vel_violation = self.check_velocity_limits(self.current_command)
            if vel_violation:
                safety_status.safety_violations.append(vel_violation)

        # Check robot state safety
        if hasattr(self, 'current_state'):
            state_violations = self.check_state_safety(self.current_state)
            safety_status.safety_violations.extend(state_violations)

        # Check for critical violations requiring emergency stop
        critical_violations = [v for v in safety_status.safety_violations
                              if v.severity == 'CRITICAL']

        if critical_violations:
            self.trigger_emergency_stop(critical_violations)
        else:
            # Publish safety status
            self.safety_status_pub.publish(safety_status)

    def check_velocity_limits(self, cmd):
        """Check if velocity commands are within limits."""
        violations = []

        if abs(cmd.linear.x) > self.safety_thresholds['max_velocity']:
            violations.append({
                'type': 'VELOCITY_EXCEEDED',
                'severity': 'WARNING',
                'description': f'Linear velocity {cmd.linear.x} exceeds limit {self.safety_thresholds["max_velocity"]}'
            })

        if abs(cmd.angular.z) > self.safety_thresholds['max_angular_velocity']:
            violations.append({
                'type': 'ANGULAR_VELOCITY_EXCEEDED',
                'severity': 'WARNING',
                'description': f'Angular velocity {cmd.angular.z} exceeds limit {self.safety_thresholds["max_angular_velocity"]}'
            })

        return violations

    def check_state_safety(self, state):
        """Check robot state for safety issues."""
        violations = []

        # Check if robot is upright
        if hasattr(state, 'orientation'):
            # Check if robot is within safe orientation bounds
            roll, pitch, yaw = self.quaternion_to_euler(state.orientation)
            if abs(roll) > 1.0 or abs(pitch) > 1.0:  # 57 degrees
                violations.append({
                    'type': 'ORIENTATION_UNSAFE',
                    'severity': 'CRITICAL',
                    'description': f'Robot orientation unsafe: roll={roll}, pitch={pitch}'
                })

        # Check joint limits
        if hasattr(state, 'joint_positions'):
            for i, pos in enumerate(state.joint_positions):
                if abs(pos) > self.safety_thresholds['max_torque']:
                    violations.append({
                        'type': 'JOINT_LIMIT_EXCEEDED',
                        'severity': 'WARNING',
                        'description': f'Joint {i} position {pos} exceeds safe limits'
                    })

        return violations

    def trigger_emergency_stop(self, violations):
        """Trigger emergency stop due to safety violations."""
        self.emergency_stop_active = True

        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        self.get_logger().error(f'EMERGENCY STOP TRIGGERED: {violations}')

    def quaternion_to_euler(self, quaternion):
        """Convert quaternion to Euler angles."""
        import math
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
```

## Testing and Validation

### AI Integration Test Suite

```python
#!/usr/bin/env python3
# AI integration test suite
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class TestAIIntegration(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = Node('ai_integration_tester')

    def tearDown(self):
        rclpy.shutdown()

    def test_perception_system_integration(self):
        """Test perception system integration with ROS 2."""
        # Subscribe to perception outputs
        perception_sub = self.node.create_subscription(
            String, '/detected_objects', lambda msg: setattr(self, 'objects', msg), 10
        )

        # Wait for perception data
        rclpy.spin_once(self.node, timeout_sec=5.0)

        # Verify perception is working
        self.assertTrue(hasattr(self, 'objects'))
        self.assertIsNotNone(self.objects)

    def test_planning_service(self):
        """Test planning service availability."""
        # Create client for planning service
        client = self.node.create_client(
            PlanPath, 'plan_path'
        )

        # Wait for service
        self.assertTrue(client.wait_for_service(timeout_sec=5.0))

        # Send planning request
        request = PlanPath.Request()
        # Set up request parameters
        future = client.call_async(request)

        # Wait for response
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
        self.assertTrue(future.done())

    def test_control_system_response(self):
        """Test control system response to commands."""
        # Publisher for velocity commands
        cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Send test command
        cmd = Twist()
        cmd.linear.x = 1.0
        cmd_pub.publish(cmd)

        # Verify command was sent
        self.assertEqual(cmd.linear.x, 1.0)

if __name__ == '__main__':
    unittest.main()
```

## Troubleshooting

### Common Integration Issues

1. **Timing Issues**
   - Ensure all systems use the same time source
   - Check for buffer overflows in high-frequency data
   - Use appropriate queue sizes for message passing

2. **Data Synchronization**
   - Use message filters for time synchronization
   - Implement proper TF tree management
   - Handle different update rates appropriately

3. **Performance Bottlenecks**
   - Profile AI model inference times
   - Optimize data transfer between systems
   - Use multi-threading where appropriate

4. **Communication Failures**
   - Verify ROS 2 network configuration
   - Check topic/service availability
   - Implement proper error handling and recovery

---
[Next: References](./references.md) | [Previous: Simulation-to-Reality](./sim2real.md)