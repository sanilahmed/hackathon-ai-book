#!/usr/bin/env python3
"""
Lab Validation Script for Physical AI & Humanoid Robotics Technical Book

This script provides validation functionality for lab exercises in the book.
Each lab can implement its own validation logic by defining specific checks.
"""

import os
import sys
import json
import argparse
from pathlib import Path


class LabValidator:
    """
    A class to validate lab exercise completion and correctness.
    """

    def __init__(self, lab_path):
        self.lab_path = Path(lab_path)
        self.results = {
            'passed': 0,
            'failed': 0,
            'total': 0,
            'checks': []
        }

    def validate_ros2_environment(self):
        """Validate that ROS 2 environment is properly set up."""
        try:
            import rclpy
            self._add_check("ROS 2 Python client library available", True)
        except ImportError:
            self._add_check("ROS 2 Python client library available", False,
                          "rclpy not found - ROS 2 may not be installed or sourced")
            return False

        # Check if ROS 2 is properly sourced by checking environment variables
        ros_domain_id = os.environ.get('ROS_DOMAIN_ID')
        if ros_domain_id is not None:
            self._add_check("ROS_DOMAIN_ID environment variable set", True)
        else:
            self._add_check("ROS_DOMAIN_ID environment variable set", False,
                          "ROS_DOMAIN_ID not set - ROS 2 may not be properly sourced")

        return True

    def validate_workspace_structure(self):
        """Validate ROS 2 workspace structure."""
        src_dir = self.lab_path.parent / "ros2-workspace" / "src"
        if src_dir.exists():
            self._add_check("ROS 2 workspace src directory exists", True)
        else:
            self._add_check("ROS 2 workspace src directory exists", False,
                          f"Expected directory does not exist: {src_dir}")
            return False

        # Check for basic workspace files
        setup_file = self.lab_path.parent / "ros2-workspace" / "setup.bash"
        if setup_file.exists():
            self._add_check("ROS 2 workspace setup.bash exists", True)
        else:
            self._add_check("ROS 2 workspace setup.bash exists", False)

        return True

    def validate_python_dependencies(self):
        """Validate that required Python dependencies are available."""
        dependencies = [
            'numpy', 'pandas', 'matplotlib', 'opencv-python', 'torch',
            'transformers', 'speechrecognition', 'pyaudio'
        ]

        missing_deps = []
        for dep in dependencies:
            try:
                __import__(dep)
                self._add_check(f"Python dependency '{dep}' available", True)
            except ImportError:
                missing_deps.append(dep)
                self._add_check(f"Python dependency '{dep}' available", False,
                               f"Missing dependency: {dep}")

        if missing_deps:
            return False
        return True

    def validate_lab_specific_files(self, expected_files):
        """Validate that specific lab files exist."""
        for file_path in expected_files:
            full_path = self.lab_path / file_path
            if full_path.exists():
                self._add_check(f"Lab file exists: {file_path}", True)
            else:
                self._add_check(f"Lab file exists: {file_path}", False,
                               f"Expected file does not exist: {full_path}")
                return False
        return True

    def _add_check(self, description, passed, message=""):
        """Add a validation check result."""
        self.results['total'] += 1
        if passed:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1

        self.results['checks'].append({
            'description': description,
            'passed': passed,
            'message': message
        })

    def run_validation(self, validation_type="basic"):
        """Run validation checks based on type."""
        if validation_type == "basic":
            self.validate_ros2_environment()
            self.validate_workspace_structure()
            self.validate_python_dependencies()
        elif validation_type == "lab":
            # Additional lab-specific validation would go here
            pass

        return self.results

    def print_results(self):
        """Print validation results in a formatted way."""
        print(f"\nLab Validation Results:")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Total:  {self.results['total']}")
        print(f"Success Rate: {self.results['passed']/self.results['total']*100:.1f}%")

        print(f"\nDetailed Results:")
        for check in self.results['checks']:
            status = "✓" if check['passed'] else "✗"
            print(f"  {status} {check['description']}")
            if check['message']:
                print(f"      {check['message']}")


def main():
    parser = argparse.ArgumentParser(description='Validate lab exercises for Physical AI book')
    parser.add_argument('lab_path', help='Path to the lab directory to validate')
    parser.add_argument('--type', choices=['basic', 'lab'], default='basic',
                       help='Type of validation to run')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')

    args = parser.parse_args()

    validator = LabValidator(args.lab_path)
    results = validator.run_validation(args.type)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        validator.print_results()

    # Exit with error code if validation failed
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()