# Numerical ODE Solver with Error Analysis

**Tech Stack:** Python, NumPy, Matplotlib, Tkinter

## Overview

A desktop application for solving first-order Ordinary Differential Equations (ODEs) using classical numerical methods. The tool provides an interactive GUI for inputting custom ODE expressions, selecting a numerical method, and visualizing results with real-time error analysis.

## Features

- **Numerical Methods:** Euler, Heun (Improved Euler), and 4th-order Runge-Kutta (RK4)
- **Error Analysis:** Computes and plots the deviation between the numerical approximation and a user-provided analytical solution
- **Interactive GUI:** Built with Tkinter — accepts custom ODE functions, initial conditions, time range, and step size
- **Data Export:** Save results as CSV, PDF, or PNG for reporting and further analysis
- **Real-time Visualization:** Plots numerical solution, analytical solution, and error curve using Matplotlib

## Usage

1. Install dependencies:
   ```
   pip install numpy matplotlib fpdf
   ```
2. Run the application:
   ```
   python as.py
   ```
3. Enter the ODE function, initial conditions, time range, step size, and (optionally) the analytical solution, then click **Solve**.

## CV / Resume Description

> **Numerical ODE Solver with Error Analysis** | Python, NumPy, Matplotlib, Tkinter  
> Developed a desktop application for solving first-order Ordinary Differential Equations (ODEs) using three classical numerical methods: Euler, Heun, and 4th-order Runge-Kutta (RK4). Built an interactive GUI with Tkinter that accepts custom ODE expressions and initial conditions, and visualizes numerical vs. analytical solutions with real-time error analysis. Implemented data export functionality to save results as CSV, PDF, or PNG.
