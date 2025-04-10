Gomoku AI Game System (GomokuAdept) - Overview
Introduction
This codebase implements a complete Gomoku (Five in a Row) game system with an advanced AI opponent powered by deep learning. The system consists of two main components:

improved_gomoku2.py - The AI ​​training and backend engine
gomoku_gui2.py - The graphical user interface for playing against the AI
improved_gomoku2.py
This file contains the core AI and game logic implementation:

GomokuGame class: Implements the game rules, board state management, move validation, and win condition checking for the 15x15 board.

GomokuAI class: A sophisticated neural network-based AI that:

Uses convolutional neural networks to evaluate board positions
Implements strategic pattern recognition
Combines rule-based heuristics with deep learning
Can be trained through self-play and expert pattern recognition
Includes opponent move prediction capabilities
Training Functions: The code includes comprehensive training pipelines:

Strategic pattern generation and recognition
Self-play reinforcement learning
Expert pattern learning
Progressive training with evaluation
Evaluation and Testing: Functions for evaluating AI strength, playing sample games, and allowing human vs. AI gameplay through a console interface.

gomoku_gui2.py
This file provides a user-friendly graphical interface for playing against the trained AI (Saved models are added to the GUI):

GomokuGame class: A simplified version of the game logic adapted for GUI interaction.

GomokuAI class: A streamlined version of the AI ​​that loads pre-trained models and focuses on move selection.

GomokuGUI class: The graphical interface that:

Displays a 15x15 Gomoku board with traditional markings
Handles user interactions and move input
Visualizes game state with X and O markers
Provides controls for:
Starting new games
Loading AI models
Customizing colors
Choosing to play as X or O
How to Use
First, train an AI model using improved_gomoku2.py or use a pre-trained model
Launch the GUI application (gomoku_gui2.py)
Load the trained model through the GUI
Play the game by clicking on the board to place your pieces
The AI ​​combines strategic rules with neural network evaluation to provide a challenging opponent. The system is designed to be both educational and entertaining, suitable for Gomoku enthusiasts of all skill levels.


