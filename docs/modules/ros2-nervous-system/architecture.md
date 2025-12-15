---
sidebar_label: 'ROS 2 Architecture'
---

# ROS 2 Architecture

This document covers the architecture of the ROS 2 robotic nervous system.

## Core Components

The ROS 2 architecture consists of several key components:

1. **Nodes**: Individual processes that perform computation
2. **Topics**: Communication channels for publishing and subscribing to data streams
3. **Services**: Synchronous request/response communication
4. **Actions**: Asynchronous goal-oriented communication

## Communication Patterns

### Publishers and Subscribers
- Asynchronous communication pattern
- Data flows from publishers to subscribers
- Uses DDS (Data Distribution Service) as the underlying transport

### Services
- Synchronous request/response pattern
- Request-response model for immediate feedback
- Good for operations that have a clear start and end

### Actions
- Asynchronous goal-oriented communication
- Suitable for long-running tasks with feedback
- Supports preempting goals

## Quality of Service (QoS)

ROS 2 provides Quality of Service settings to control communication behavior:
- Reliability: Best effort or reliable
- Durability: Volatile or transient local
- History: Keep all or keep last N samples