/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['modules/ros2-nervous-system/index'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 Robotic Nervous System',
      items: [
        'modules/ros2-nervous-system/index',
        'modules/ros2-nervous-system/architecture',
        'modules/ros2-nervous-system/nodes-topics',
        'modules/ros2-nervous-system/urdf-modeling',
        'modules/ros2-nervous-system/ai-integration',
        {
          type: 'category',
          label: 'Lab Exercises',
          items: [
            'modules/lab-exercises/lab-1-1-ros2-basics',
            'modules/lab-exercises/lab-1-1-ros2-setup',
            'modules/lab-exercises/lab-1-2-ros2-services-actions',
            'modules/lab-exercises/lab-1-3-robot-state-management'
          ],
        },
        'modules/ros2-nervous-system/references'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Gazebo + Unity)',
      items: [
        'modules/digital-twin/index',
        'modules/digital-twin/gazebo-setup',
        'modules/digital-twin/unity-integration',
        'modules/digital-twin/sensor-simulation',
        'modules/digital-twin/ros2-sync',
        {
          type: 'category',
          label: 'Lab Exercises',
          items: [
            'modules/lab-exercises/lab-2-1-gazebo-setup',
            'modules/lab-exercises/lab-2-2-robot-model-integration',
            'modules/lab-exercises/lab-2-3-unity-robotics-integration',
            'modules/lab-exercises/lab-2-4-multi-environment-synchronization'
          ],
        },
        'modules/digital-twin/references'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'modules/ai-robot-brain/index',
        'modules/ai-robot-brain/isaac-sim-setup',
        'modules/ai-robot-brain/perception-systems',
        'modules/ai-robot-brain/planning-control',
        'modules/ai-robot-brain/reinforcement-learning',
        {
          type: 'category',
          label: 'Lab Exercises',
          items: [
            'modules/lab-exercises/lab-3-1-isaac-navigation',
            'modules/lab-exercises/lab-3-1-isaac-sim-setup',
            'modules/lab-exercises/lab-3-2-perception-systems',
            'modules/lab-exercises/lab-3-3-planning-control',
            'modules/lab-exercises/lab-3-4-reinforcement-learning',
            'modules/lab-exercises/lab-3-5-sim2real-transfer'
          ],
        },
        'modules/ai-robot-brain/references',
        'modules/ai-robot-brain/sim2real'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'modules/vla-system/index',
        'modules/vla-system/vla-fundamentals',
        'modules/vla-system/vla-architecture',
        'modules/vla-system/multimodal-perception',
        'modules/vla-system/language-action-mapping',
        'modules/vla-system/training-vla-models',
        'modules/vla-system/vla-integration',
        {
          type: 'category',
          label: 'Lab Exercises',
          items: [
            'modules/lab-exercises/lab-4-1-vla-fundamentals',
            'modules/lab-exercises/lab-4-2-multimodal-perception',
            'modules/lab-exercises/lab-4-3-action-mapping'
          ],
        },
        'modules/vla-system/references',
        'modules/vla-system/safety-evaluation'
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'modules/lab-exercises',
        'reference/glossary',
        'reference/citations'
      ],
    }
  ],
};

module.exports = sidebars;