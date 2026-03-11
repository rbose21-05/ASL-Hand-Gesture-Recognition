# ASL Hand Gesture Recognition 🚧 Work in Progress

A real-time American Sign Language (ASL) hand gesture recognition system using computer vision and deep learning. This project achieves **98.28% validation accuracy** on a 3-class gesture recognition task using transfer learning with MobileNetV2.

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **⚠️ Project Status**: Core training pipeline complete (98.28% accuracy achieved). Real-time inference demo currently in development.

## 🎯 Project Overview

This project demonstrates an end-to-end machine learning pipeline for hand gesture recognition, from data collection to model deployment. The system uses MediaPipe for hand detection and segmentation, combined with transfer learning on MobileNetV2 for accurate gesture classification.

### Key Features

-   ✅ Custom data collection pipeline via webcam
-   ✅ Intelligent hand detection and preprocessing using MediaPipe
-   ✅ Transfer learning with pre-trained MobileNetV2
-   ✅ Data augmentation for improved generalization
-   ✅ **98.28% validation accuracy achieved**
-   🚧 Real-time webcam inference (coming soon)
-   🚧 Deployment-ready optimization (planned)

## 📊 Current Results

-   **Training Accuracy**: 100%
-   **Validation Accuracy**: 98.28%
-   **Model Size**: ~9.24 MB
-   **Trainable Parameters**: 164,355 (only 6.8% of total)

## 🏗️ Architecture

### Model Pipeline
