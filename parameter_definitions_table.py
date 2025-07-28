#!/usr/bin/env python3
"""
Script to generate a LaTeX table with parameter definitions and explanations.
"""

import pandas as pd

def create_parameter_definitions_table():
    """Create a comprehensive table of parameter definitions."""
    
    # Define all parameters and their explanations
    parameters = {
        'best_value': 'Validation accuracy achieved by the best trial (percentage)',
        'n_trials': 'Total number of optimization trials completed',
        'n_units': 'Number of unicycle units in the reservoir network',
        'lr': 'Learning rate for the Adam optimizer',
        'batch_size': 'Number of samples processed in each training batch',
        'n_epochs': 'Number of complete passes through the training dataset',
        'dt': 'Time step size for numerical integration of dynamics',
        'anchor_con_fraction': 'Fraction of connections that are anchor connections',
        'ang_damping_max': 'Maximum angular damping coefficient for unicycle dynamics',
        'ang_damping_min': 'Minimum angular damping coefficient for unicycle dynamics',
        'ang_stiff_max': 'Maximum angular stiffness coefficient for connections',
        'ang_stiff_min': 'Minimum angular stiffness coefficient for connections',
        'eq_dist_max': 'Maximum equilibrium distance for linear connections',
        'eq_dist_max_ang': 'Maximum equilibrium distance for angular connections',
        'eq_dist_min': 'Minimum equilibrium distance for linear connections',
        'eq_dist_min_ang': 'Minimum equilibrium distance for angular connections',
        'inp_bias': 'Bias term added to input signals',
        'lin_damping_max': 'Maximum linear damping coefficient for unicycle dynamics',
        'lin_damping_min': 'Minimum linear damping coefficient for unicycle dynamics',
        'lin_stiff_max': 'Maximum linear stiffness coefficient for connections',
        'lin_stiff_min': 'Minimum linear stiffness coefficient for connections',
        'magnitude_max': 'Maximum magnitude for linear input mapping weights',
        'magnitude_max_ang': 'Maximum magnitude for angular input mapping weights',
        'magnitude_min': 'Minimum magnitude for linear input mapping weights',
        'magnitude_min_ang': 'Minimum magnitude for angular input mapping weights',
        'n_connections': 'Total number of inter-unit connections in the network',
        'non_zero_elements': 'Number of non-zero elements in linear input mapping',
        'non_zero_elements_ang': 'Number of non-zero elements in angular input mapping',
        'steps_readout': 'Number of past time steps used for readout computation',
        'washup_steps': 'Number of initialization steps before processing input data',
        'param_count': 'Total estimated parameter count (n_units × 5 × 10 × steps_readout)'
    }
    
    return parameters

def generate_latex_definitions_table(parameters, filename="parameter_definitions.tex"):
    """Generate a LaTeX table with parameter definitions."""
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Unicycle Network Hyperparameter Definitions}")
    latex_content.append("\\label{tab:parameter_definitions}")
    latex_content.append("\\begin{tabular}{p{4cm}p{8cm}}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Parameter} & \\textbf{Description} \\\\")
    latex_content.append("\\midrule")
    
    # Add each parameter
    for param, description in parameters.items():
        # Escape underscores for LaTeX
        param_latex = param.replace('_', '\\_')
        latex_content.append(f"\\texttt{{{param_latex}}} & {description} \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"Parameter definitions table saved to: {filename}")
    return latex_content

def generate_csv_definitions_table(parameters, filename="parameter_definitions.csv"):
    """Generate a CSV file with parameter definitions."""
    
    # Create DataFrame
    df = pd.DataFrame(list(parameters.items()), columns=['Parameter', 'Description'])
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Parameter definitions CSV saved to: {filename}")
    
    return df

def print_definitions_table(parameters):
    """Print a formatted table of parameter definitions."""
    
    print("=" * 80)
    print("UNICYCLE NETWORK PARAMETER DEFINITIONS")
    print("=" * 80)
    print()
    
    # Calculate column widths
    max_param_len = max(len(param) for param in parameters.keys())
    param_width = max(max_param_len, 20)
    
    # Print header
    print(f"{'Parameter':<{param_width}} | Description")
    print("-" * param_width + "-+-" + "-" * 50)
    
    # Print each parameter
    for param, description in parameters.items():
        # Wrap long descriptions
        if len(description) > 50:
            lines = []
            words = description.split()
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= 50:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Print first line
            print(f"{param:<{param_width}} | {lines[0]}")
            # Print continuation lines
            for line in lines[1:]:
                print(f"{'':<{param_width}} | {line}")
        else:
            print(f"{param:<{param_width}} | {description}")
    
    print()

def generate_grouped_latex_table(parameters, filename="parameter_definitions_grouped.tex"):
    """Generate a grouped LaTeX table organizing parameters by category."""
    
    # Define parameter groups
    groups = {
        'Performance Metrics': [
            'best_value', 'n_trials'
        ],
        'Network Architecture': [
            'n_units', 'n_connections', 'non_zero_elements', 'non_zero_elements_ang', 'param_count'
        ],
        'Training Parameters': [
            'lr', 'batch_size', 'n_epochs', 'steps_readout', 'washup_steps'
        ],
        'Dynamics Parameters': [
            'dt', 'inp_bias'
        ],
        'Linear Dynamics': [
            'lin_stiff_min', 'lin_stiff_max', 'lin_damping_min', 'lin_damping_max',
            'magnitude_min', 'magnitude_max'
        ],
        'Angular Dynamics': [
            'ang_stiff_min', 'ang_stiff_max', 'ang_damping_min', 'ang_damping_max',
            'magnitude_min_ang', 'magnitude_max_ang'
        ],
        'Connection Properties': [
            'anchor_con_fraction', 'eq_dist_min', 'eq_dist_max', 'eq_dist_min_ang', 'eq_dist_max_ang'
        ]
    }
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Unicycle Network Hyperparameter Definitions by Category}")
    latex_content.append("\\label{tab:parameter_definitions_grouped}")
    latex_content.append("\\begin{tabular}{p{4cm}p{8cm}}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Parameter} & \\textbf{Description} \\\\")
    latex_content.append("\\midrule")
    
    # Add each group
    for group_name, param_list in groups.items():
        # Add group header
        latex_content.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{group_name}}}}} \\\\")
        latex_content.append("\\midrule")
        
        # Add parameters in this group
        for param in param_list:
            if param in parameters:
                param_latex = param.replace('_', '\\_')
                description = parameters[param]
                latex_content.append(f"\\texttt{{{param_latex}}} & {description} \\\\")
        
        # Add some spacing between groups (except for the last one)
        if group_name != list(groups.keys())[-1]:
            latex_content.append("\\midrule")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"Grouped parameter definitions table saved to: {filename}")
    return latex_content

if __name__ == "__main__":
    print("Generating parameter definitions tables...")
    
    # Get parameter definitions
    parameters = create_parameter_definitions_table()
    
    # Print to console
    print_definitions_table(parameters)
    
    # Generate files
    generate_csv_definitions_table(parameters, "parameter_definitions.csv")
    generate_latex_definitions_table(parameters, "parameter_definitions.tex")
    generate_grouped_latex_table(parameters, "parameter_definitions_grouped.tex")
    
    print("\nFiles generated:")
    print("- parameter_definitions.csv")
    print("- parameter_definitions.tex")
    print("- parameter_definitions_grouped.tex")
    
    print("\nLaTeX packages needed:")
    print("\\usepackage{booktabs}")
    print("\\usepackage{array}")
