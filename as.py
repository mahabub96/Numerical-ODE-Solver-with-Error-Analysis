import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import csv
from fpdf import FPDF

# Numerical Methods
class ODESolver:
    def euler_method(self, func, y0, t0, t_end, h):
        t_values = [t0]
        y_values = [y0]

        while t_values[-1] < t_end:
            t = t_values[-1]
            y = y_values[-1]
            y_next = y + h * func(t, y)
            t_next = t + h

            y_values.append(y_next)
            t_values.append(t_next)

        return np.array(t_values), np.array(y_values)

    def heun_method(self, func, y0, t0, t_end, h):
        t_values = [t0]
        y_values = [y0]

        while t_values[-1] < t_end:
            t = t_values[-1]
            y = y_values[-1]
            y_predict = y + h * func(t, y)
            y_correct = y + (h / 2) * (func(t, y) + func(t + h, y_predict))
            t_next = t + h

            y_values.append(y_correct)
            t_values.append(t_next)

        return np.array(t_values), np.array(y_values)

    def runge_kutta_4(self, func, y0, t0, t_end, h):
        t_values = [t0]
        y_values = [y0]

        while t_values[-1] < t_end:
            t = t_values[-1]
            y = y_values[-1]

            k1 = h * func(t, y)
            k2 = h * func(t + h / 2, y + k1 / 2)
            k3 = h * func(t + h / 2, y + k2 / 2)
            k4 = h * func(t + h, y + k3)

            y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t_next = t + h

            y_values.append(y_next)
            t_values.append(t_next)

        return np.array(t_values), np.array(y_values)

# GUI Application
class ODESolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ODE Solver with Error Analysis")

        self.solver = ODESolver()

        # Input frame
        self.input_frame = ttk.Frame(root, padding="10")
        self.input_frame.grid(row=0, column=0, sticky="NSEW")

        # Inputs
        ttk.Label(self.input_frame, text="ODE Function (e.g., -y + np.sin(t)):").grid(row=0, column=0, sticky="W")
        self.func_entry = ttk.Entry(self.input_frame, width=30)
        self.func_entry.grid(row=0, column=1, sticky="W")
        self.func_entry.insert(0, "-y + np.sin(t)")

        ttk.Label(self.input_frame, text="Initial Condition (y0):").grid(row=1, column=0, sticky="W")
        self.y0_entry = ttk.Entry(self.input_frame)
        self.y0_entry.grid(row=1, column=1, sticky="W")
        self.y0_entry.insert(0, "1")

        ttk.Label(self.input_frame, text="Time Start (t0):").grid(row=2, column=0, sticky="W")
        self.t0_entry = ttk.Entry(self.input_frame)
        self.t0_entry.grid(row=2, column=1, sticky="W")
        self.t0_entry.insert(0, "0")

        ttk.Label(self.input_frame, text="Time End (t_end):").grid(row=3, column=0, sticky="W")
        self.t_end_entry = ttk.Entry(self.input_frame)
        self.t_end_entry.grid(row=3, column=1, sticky="W")
        self.t_end_entry.insert(0, "10")

        ttk.Label(self.input_frame, text="Step Size (h):").grid(row=4, column=0, sticky="W")
        self.h_entry = ttk.Entry(self.input_frame)
        self.h_entry.grid(row=4, column=1, sticky="W")
        self.h_entry.insert(0, "0.1")

        ttk.Label(self.input_frame, text="Method:").grid(row=5, column=0, sticky="W")
        self.method_var = tk.StringVar(value="Euler")
        self.method_menu = ttk.Combobox(self.input_frame, textvariable=self.method_var, state="readonly")
        self.method_menu["values"] = ["Euler", "Heun", "RK4"]
        self.method_menu.grid(row=5, column=1, sticky="W")

        ttk.Label(self.input_frame, text="Analytical Solution (if known):").grid(row=6, column=0, sticky="W")
        self.analytical_func_entry = ttk.Entry(self.input_frame, width=30)
        self.analytical_func_entry.grid(row=6, column=1, sticky="W")
        self.analytical_func_entry.insert(0, "np.exp(-t) + np.sin(t)")

        # Solve Button
        self.solve_button = ttk.Button(self.input_frame, text="Solve", command=self.solve_with_error_analysis)
        self.solve_button.grid(row=7, column=0, columnspan=2)

        # Save Buttons
        self.save_csv_button = ttk.Button(self.input_frame, text="Save as CSV", command=self.save_solution_csv)
        self.save_csv_button.grid(row=8, column=0, columnspan=1)

        self.save_pdf_button = ttk.Button(self.input_frame, text="Save as PDF", command=self.save_solution_pdf)
        self.save_pdf_button.grid(row=8, column=1, columnspan=1)

        self.save_plot_button = ttk.Button(self.input_frame, text="Save Plot as PNG", command=self.save_plot_png)
        self.save_plot_button.grid(row=9, column=0, columnspan=2)

        # Visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0)

        self.solution_label = ttk.Label(root, text="", foreground="blue")
        self.solution_label.grid(row=2, column=0)

        self.solution = None

    def solve_with_error_analysis(self):
        # Clear previous plot and label
        self.plot.clear()
        self.solution_label.config(text="")

        # Inputs
        try:
            y0 = float(self.y0_entry.get())
            t0 = float(self.t0_entry.get())
            t_end = eval(self.t_end_entry.get())  # Allow for expressions like "2*np.pi"
            h = float(self.h_entry.get())
            func_str = self.func_entry.get()
            analytical_func_str = self.analytical_func_entry.get()  # New Input

            # Define ODE function
            def func(t, y):
                return eval(func_str)

            # Define Analytical Solution
            def analytical_solution(t):
                return eval(analytical_func_str)

            # Solve using selected method
            method = self.method_var.get()
            if method == "Euler":
                t_values, y_values = self.solver.euler_method(func, y0, t0, t_end, h)
            elif method == "Heun":
                t_values, y_values = self.solver.heun_method(func, y0, t0, t_end, h)
            elif method == "RK4":
                t_values, y_values = self.solver.runge_kutta_4(func, y0, t0, t_end, h)
            else:
                raise ValueError("Unknown method")

            # Store solution for saving
            self.solution = (t_values, y_values)

            # Calculate Errors
            analytical_values = np.array([analytical_solution(t) for t in t_values])
            errors = np.abs(analytical_values - y_values)

            # Plot results
            self.plot.plot(t_values, y_values, label=f"{method} Numerical Solution", color="blue")
            self.plot.plot(t_values, analytical_values, label="Analytical Solution", linestyle="dashed", color="green")
            self.plot.plot(t_values, errors, label="Error (|Analytical - Numerical|)", color="red")
            self.plot.set_title("ODE Solution with Error Analysis")
            self.plot.set_xlabel("Time (t)")
            self.plot.set_ylabel("Solution (y)")
            self.plot.legend()

            self.canvas.draw()

            # Show final error
            self.solution_label.config(
                text=f"Final Value: y({t_end}) = {y_values[-1]:.4f}, Error = {errors[-1]:.4f}"
            )
        except Exception as e:
            print(f"Error: {e}")

    def save_solution_csv(self):
        if self.solution is None:
            print("No solution to save!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        t_values, y_values = self.solution
       
        # Save solution to CSV
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Time (t)", "Numerical Solution (y)"])
                for t, y in zip(t_values, y_values):
                    writer.writerow([t, y])
            print(f"Solution saved to {file_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def save_solution_pdf(self):
        if self.solution is None:
            print("No solution to save!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return

        t_values, y_values = self.solution
        try:
            # Create a PDF document
            pdf = FPDF()
            pdf.add_page()

            # Set font and add title
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="ODE Solution and Error Analysis", ln=True, align="C")

            # Add table with solution data
            pdf.ln(10)
            pdf.cell(40, 10, "Time (t)", border=1, align="C")
            pdf.cell(60, 10, "Numerical Solution (y)", border=1, align="C")
            for t, y in zip(t_values, y_values):
                pdf.ln(10)
                pdf.cell(40, 10, str(t), border=1, align="C")
                pdf.cell(60, 10, f"{y:.4f}", border=1, align="C")

            # Save the PDF
            pdf.output(file_path)
            print(f"Solution saved to {file_path}")
        except Exception as e:
            print(f"Error saving PDF: {e}")

    def save_plot_png(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            return

        try:
            self.figure.savefig(file_path)
            print(f"Plot saved to {file_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")


# Create the main window and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ODESolverApp(root)
    root.mainloop()
