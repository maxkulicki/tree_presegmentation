# TreeExtractor

Extract individual trees from TLS/MLS point clouds based on dendromatics library. Good for presegmentation before doing manual tree segmentation.

## Overview

This repository contains scripts for automated tree detection and extraction from Terrestrial Laser Scanning (TLS) and Mobile Laser Scanning (MLS) point cloud data. 

## Main Features

- Detect individual tree stems in point cloud data
- Extract individual trees by creating tilted cylinders around detected stems
- Process large point clouds efficiently with grid-based approach
- Export extracted trees as LAZ files

## Workflow

### Standard Pipeline (`pipeline.py`)

The standard workflow consists of:

1. Cropping the point cloud with a fixed radius around the origin (0,0)
2. Detecting tree stems using dendromatics algorithms
3. Extracting individual trees by creating cylinders around stem centers
   - Cylinders are tilted based on the detected stem angle
4. Saving each extracted tree as a separate LAZ file

Specify the input and output directories in the main function arguments.

### Grid Pipeline for Large Point Cloud Processing (`grid_pipeline.py`)

For memory-efficient processing of large point clouds:

1. Divides the point cloud into a rectangular grid
2. Processes each grid section individually
3. Combines the results

Specify the input and output directories in the main function arguments.

### Requirements
See requirements.txt for a list of dependencies.