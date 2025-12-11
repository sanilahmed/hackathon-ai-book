<!--
Sync Impact Report:
Version change: N/A → 1.0.0 (Initial creation)
Added sections: All principles and sections from user input
Removed sections: None (new file)
Templates requiring updates:
- .specify/templates/plan-template.md ⚠ pending (check for constitution references)
- .specify/templates/spec-template.md ⚠ pending (check for constitution references)
- .specify/templates/tasks-template.md ⚠ pending (check for constitution references)
- .specify/templates/phr-template.prompt.md ⚠ pending (check for constitution references)
Follow-up TODOs: None
-->
# AI/Spec-Driven Technical Book on Physical AI & Humanoid Robotics Constitution

## Core Principles

### Technical Accuracy through Primary-Source Validation
Technical accuracy through primary-source validation (official docs, research papers, standards).

### Clarity for Mixed Audience
Clarity for a mixed audience: early-career engineers, robotics students, and advanced learners.

### Reproducibility
Reproducibility: all code, algorithms, and simulations must be executable by the reader.

### Engineering Rigor
Engineering rigor: include mathematical derivations, algorithmic steps, and architectural details where relevant.

### Ethical Alignment
Ethical alignment: emphasize robotic safety, alignment, responsible AI deployment, and human–robot interaction principles.

### Source & Citation Requirements

All claims must be backed by verifiable sources. Allowed citation styles: IEEE or APA (consistent throughout each chapter). Source types: Peer-reviewed papers, Robotics textbooks, Standards (IEEE, ISO, ROS REP), Credible official documentation (ROS 2, Unity, Isaac Sim, etc.). Minimum 40% peer-reviewed or academic sources. Zero tolerance for: Fabricated citations, Hallucinated facts, Outdated or deprecated information unless explicitly marked.

## Technical & Writing Standards

Algorithms must include: Pseudocode, Diagrams or flowcharts, Example implementations (Python, TypeScript, or C++). Hardware and simulation examples must include: Setup steps, Version numbers, Reproducible commands. Writing clarity: technically precise yet accessible for motivated beginners. Glossary of robotics + AI terminology maintained throughout the book. Book Structure Requirements: Each chapter must follow a modular structure: Learning objectives, Concept overview, Technical explanation with diagrams, Mathematical formulations (where applicable), Pseudocode or algorithm walkthrough, Code examples in Python / TypeScript / C++, Practical labs or exercises, Troubleshooting notes, Summary, Verified references. Additional structural requirements: Dedicated section for robot safety & ethics, Full glossary of technical terms, Architecture maps, data flow diagrams, robot schematics as needed, Appendices for extended tutorials (optional).

## Content and Technical Constraints

Content Constraints: No speculative claims unless clearly labeled as theoretical or future work. All robotic architectures must reflect current industry/state-of-the-art practices. Avoid unverified performance claims about robots or algorithms. Safety, alignment, ethical considerations must be explicitly discussed. Technical Constraints: Entire book must be: Built using Docusaurus, Managed via GitHub version control, Deployed to GitHub Pages, Authored collaboratively using Spec-Kit Plus + Claude Code, All diagrams must be compatible with Docusaurus (Mermaid, SVG, PNG). Final build must pass without errors.

## Governance

Success Criteria: All content technically correct and validated against sources. Logical, coherent chapter flow with cross-topic consistency. All code and simulation examples are reproducible and tested. Zero fabricated citations and 0% plagiarism. Appropriate for learners with basic robotics & programming knowledge. Book successfully deployed to GitHub Pages with working navigation, links, and diagrams. Stretch Goals: Interactive simulation links (WebGL, Unity Web, Isaac Sim recordings). Real robot hardware integration examples. Automatic citation generation pipeline. Multilingual versions (Urdu, English).

**Version**: 1.0.0 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-10