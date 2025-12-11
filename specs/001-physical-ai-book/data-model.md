# Data Model: Physical AI & Humanoid Robotics Technical Book

## Overview
This document defines the key data structures and entities for the Physical AI & Humanoid Robotics technical book. Since this is primarily a documentation project, the "data models" represent the structured content and metadata needed for the educational modules.

## Entity: LearningModule
A structured educational unit covering one of the four core topics.

**Fields**:
- id: string (unique identifier for the module)
- title: string (display title of the module)
- description: string (brief description of the module content)
- moduleNumber: integer (1-4, indicating the sequence)
- objectives: string[] (learning objectives for the module)
- topics: string[] (key topics covered in the module)
- prerequisites: string[] (required knowledge or completed modules)
- estimatedDuration: integer (time in hours to complete the module)
- languageSupport: string[] (programming languages covered in examples)
- diagrams: Diagram[] (list of diagrams used in the module)
- pseudocode: Pseudocode[] (list of pseudocode examples)
- labs: LabExercise[] (list of lab exercises)
- references: Citation[] (list of citations used in the module)
- learningOutcomes: string[] (specific outcomes students should achieve)

**Validation Rules**:
- moduleNumber must be between 1 and 4
- title and description are required
- objectives must contain at least 3 items
- estimatedDuration must be greater than 0
- languageSupport values must be from: ["Python", "C++", "TypeScript"]

## Entity: Diagram
Visual representation used in the educational content.

**Fields**:
- id: string (unique identifier)
- title: string (descriptive title)
- description: string (explanation of what the diagram illustrates)
- filePath: string (path to the diagram file relative to docs root)
- format: string (format of the diagram: "PNG", "SVG", "Mermaid")
- moduleRef: string (ID of the module this diagram belongs to)
- altText: string (alternative text for accessibility)

**Validation Rules**:
- filePath must exist and be accessible
- format must be one of the allowed values
- moduleRef must reference an existing module

## Entity: Pseudocode
Algorithm representation in the educational content.

**Fields**:
- id: string (unique identifier)
- title: string (descriptive title)
- algorithm: string (the pseudocode content)
- explanation: string (detailed explanation of the algorithm)
- moduleRef: string (ID of the module this pseudocode belongs to)
- complexity: string (time/space complexity if applicable)
- languageRef: string (reference to actual implementation language)

**Validation Rules**:
- algorithm must follow standard pseudocode conventions
- moduleRef must reference an existing module

## Entity: LabExercise
Hands-on exercise with step-by-step instructions.

**Fields**:
- id: string (unique identifier)
- title: string (descriptive title)
- description: string (brief overview of the lab exercise)
- objectives: string[] (what students will learn from this lab)
- prerequisites: string[] (what students need before starting)
- environmentRequirements: string[] (software, hardware requirements)
- stepByStepInstructions: string[] (detailed steps to complete the lab)
- expectedOutput: string (what students should see when completed correctly)
- troubleshootingTips: string[] (common issues and solutions)
- validationScript: string (path to script that validates completion)
- moduleRef: string (ID of the module this lab belongs to)
- estimatedDuration: integer (time in minutes to complete)

**Validation Rules**:
- environmentRequirements must be reproducible in Ubuntu 22.04
- stepByStepInstructions must have at least 3 steps
- moduleRef must reference an existing module

## Entity: Citation
Verified reference to academic, peer-reviewed, or official documentation.

**Fields**:
- id: string (unique identifier)
- type: string ("academic", "peer-reviewed", "official-documentation", "book", "standard")
- title: string (title of the cited work)
- authors: string[] (list of authors)
- publication: string (journal, conference, or publisher)
- year: integer (publication year)
- doi: string (Digital Object Identifier if available)
- url: string (URL to the source)
- accessedDate: string (date when the source was accessed, ISO format)
- relevance: string (explanation of how this citation is relevant to the content)
- moduleRef: string (ID of the module where this citation is used)

**Validation Rules**:
- type must be one of the allowed values
- at least 40% of citations in the book must be "academic" or "peer-reviewed"
- doi or url must be provided
- accessedDate must be in ISO format (YYYY-MM-DD)

## Entity: TechnicalBook
The complete educational resource containing all modules.

**Fields**:
- id: string (unique identifier)
- title: string (title of the book)
- subtitle: string (subtitle of the book)
- version: string (semantic version of the book)
- authors: string[] (list of book authors)
- modules: LearningModule[] (all 4 learning modules)
- glossary: GlossaryTerm[] (glossary of robotics and AI terminology)
- references: Citation[] (all citations used in the book)
- targetAudience: string[] (target audience segments)
- prerequisites: string[] (overall prerequisites for the book)
- totalDuration: integer (estimated total time to complete the book)
- lastUpdated: string (date when the book was last updated, ISO format)

**Validation Rules**:
- must contain exactly 4 modules
- targetAudience must include at least one of the specified audiences
- totalDuration must equal the sum of all module durations

## Entity: GlossaryTerm
Definition of a robotics or AI term used in the book.

**Fields**:
- id: string (unique identifier)
- term: string (the term being defined)
- definition: string (clear, concise definition)
- moduleRefs: string[] (IDs of modules where this term is used)
- relatedTerms: string[] (other related glossary terms)

**Validation Rules**:
- term and definition are required
- each term must be used in at least one module

## Entity: QuickReference
Quick reference material for common concepts or commands.

**Fields**:
- id: string (unique identifier)
- title: string (descriptive title)
- content: string (the reference content)
- moduleRef: string (ID of the module this reference belongs to)
- category: string (category of the reference: "command", "configuration", "formula", etc.)

**Validation Rules**:
- moduleRef must reference an existing module
- category must be one of the allowed values