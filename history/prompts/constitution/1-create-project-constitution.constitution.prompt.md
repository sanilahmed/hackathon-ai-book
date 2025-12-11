---
id: 1
title: Create Project Constitution
stage: constitution
date: 2025-12-10
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution,setup,initial]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
Project: AI/Spec-Driven Technical Book on Physical AI & Humanoid Robotics
Purpose:
Develop a structured, technically rigorous, and reproducible book using Docusaurus—authored collaboratively with Spec-Kit Plus and Claude Code—and deployed on GitHub Pages. The book teaches students how to design, simulate, control, and evaluate embodied AI systems and humanoid robots in both simulated and physical environments.

Core Principles

Technical accuracy through primary-source validation (official docs, research papers, standards).

Clarity for a mixed audience: early-career engineers, robotics students, and advanced learners.

Reproducibility: all code, algorithms, and simulations must be executable by the reader.

Engineering rigor: include mathematical derivations, algorithmic steps, and architectural details where relevant.

Ethical alignment: emphasize robotic safety, alignment, responsible AI deployment, and human–robot interaction principles.

Key Standards
Source & Citation Requirements

All claims must be backed by verifiable sources.

Allowed citation styles: IEEE or APA (consistent throughout each chapter).

Source types:

Peer-reviewed papers

Robotics textbooks

Standards (IEEE, ISO, ROS REP)

Credible official documentation (ROS 2, Unity, Isaac Sim, etc.)

Minimum 40% peer-reviewed or academic sources.

Zero tolerance for:

Fabricated citations

Hallucinated facts

Outdated or deprecated information unless explicitly marked

Technical & Writing Standards

Algorithms must include:

Pseudocode

Diagrams or flowcharts

Example implementations (Python, TypeScript, or C++)

Hardware and simulation examples must include:

Setup steps

Version numbers

Reproducible commands

Writing clarity: technically precise yet accessible for motivated beginners.

Glossary of robotics + AI terminology maintained throughout the book.

Book Structure Requirements

Each chapter must follow a modular structure:

Learning objectives

Concept overview

Technical explanation with diagrams

Mathematical formulations (where applicable)

Pseudocode or algorithm walkthrough

Code examples in Python / TypeScript / C++

Practical labs or exercises

Troubleshooting notes

Summary

Verified references

Additional structural requirements:

Dedicated section for robot safety & ethics

Full glossary of technical terms

Architecture maps, data flow diagrams, robot schematics as needed

Appendices for extended tutorials (optional)

Content Constraints

No speculative claims unless clearly labeled as theoretical or future work.

All robotic architectures must reflect current industry/state-of-the-art practices.

Avoid unverified performance claims about robots or algorithms.

Safety, alignment, ethical considerations must be explicitly discussed.

Technical Constraints

Entire book must be:

Built using Docusaurus

Managed via GitHub version control

Deployed to GitHub Pages

Authored collaboratively using Spec-Kit Plus + Claude Code

All diagrams must be compatible with Docusaurus (Mermaid, SVG, PNG).

Final build must pass without errors.

Success Criteria

All content technically correct and validated against sources.

Logical, coherent chapter flow with cross-topic consistency.

All code and simulation examples are reproducible and tested.

Zero fabricated citations and 0% plagiarism.

Appropriate for learners with basic robotics & programming knowledge.

Book successfully deployed to GitHub Pages with working navigation, links, and diagrams.

Stretch Goals

Interactive simulation links (WebGL, Unity Web, Isaac Sim recordings).

Real robot hardware integration examples.

Automatic citation generation pipeline.

Multilingual versions (Urdu, English).
```

## Response snapshot

Created the project constitution file (.specify/memory/constitution.md) with the specified principles and guidelines from the user input. The constitution includes:
- Core principles for technical accuracy, clarity, reproducibility, engineering rigor, and ethical alignment
- Technical & Writing Standards with requirements for algorithms, examples, and book structure
- Content and Technical Constraints for the AI/robotics book project
- Governance section with success criteria and stretch goals
- Version tracking (1.0.0) with ratification date

## Outcome

- ✅ Impact: Created initial project constitution with all specified principles and constraints
- 🧪 Tests: none
- 📁 Files: Created .specify/memory/constitution.md with all required content
- 🔁 Next prompts: Ready to proceed with project planning and specification
- 🧠 Reflection: Successfully transformed user requirements into a structured constitution document

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A