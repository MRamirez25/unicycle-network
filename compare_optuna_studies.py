#!/usr/bin/env python3
"""
Script to compare hyperparameters from multiple Optuna studies.
"""

import optuna
import pandas as pd
import os

# Define the study names
study_names = [
    "unicycle_opt_all_classes_aligned_no_ang_input_no_ang_connections",
    "unicycle_opt_all_classes_not_aligned_no_ang_input_no_ang_connections",
    "unicycle_opt_all_classes_not_aligned_w_ang_input_no_ang_connections"
]

def load_study_best_params(study_name, base_dir="."):
    """Load the best parameters from an Optuna study."""
    try:
        storage_name = f"sqlite:///{base_dir}/optuna_databases/{study_name}.db"
        study = optuna.load_study(storage=storage_name, study_name=study_name)
        
        # Get best trial info
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_params['best_value'] = best_trial.value
        best_params['n_trials'] = len(study.trials)
        
        return best_params
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")
        return None

def compare_studies(study_names, base_dir="."):
    """Compare hyperparameters across multiple studies."""
    
    # Load all studies
    studies_data = {}
    for study_name in study_names:
        params = load_study_best_params(study_name, base_dir)
        if params is not None:
            studies_data[study_name] = params
        else:
            print(f"Skipping study: {study_name}")
    
    if not studies_data:
        print("No studies could be loaded!")
        return None
    
    # Convert to DataFrame for easy comparison
    df = pd.DataFrame(studies_data).T
    
    # Reorder columns to put important metrics first
    important_cols = ['best_value', 'n_trials']
    param_cols = [col for col in df.columns if col not in important_cols]
    df = df[important_cols + sorted(param_cols)]
    
    return df

def print_comparison_table(df):
    """Print a nicely formatted comparison table."""
    print("=" * 100)
    print("OPTUNA STUDIES COMPARISON")
    print("=" * 100)
    
    # Print basic info
    print("\nStudy Overview:")
    print("-" * 50)
    for study_name in df.index:
        best_val = df.loc[study_name, 'best_value']
        n_trials = df.loc[study_name, 'n_trials']
        print(f"{study_name[:50]:<50} | Best: {best_val:.4f} | Trials: {n_trials}")
    
    print("\n" + "=" * 100)
    print("DETAILED HYPERPARAMETERS COMPARISON")
    print("=" * 100)
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    print(df.round(6))

def save_comparison_to_csv(df, filename="optuna_studies_comparison.csv"):
    """Save the comparison table to a CSV file."""
    df.to_csv(filename)
    print(f"\nComparison table saved to: {filename}")

def generate_latex_table(df, filename="optuna_studies_comparison.tex"):
    """Generate a LaTeX table from the comparison DataFrame."""
    
    # Create a copy for LaTeX formatting
    latex_df = df.copy()
    
    # Round numeric columns and handle NaN values
    numeric_cols = latex_df.select_dtypes(include=['float64', 'int64']).columns
    latex_df[numeric_cols] = latex_df[numeric_cols].round(4)
    
    # Replace NaN values with a LaTeX-safe placeholder
    latex_df = latex_df.fillna('--')
    
    # Fix column names for LaTeX (replace underscores)
    latex_df.columns = [col.replace('_', '\\_') for col in latex_df.columns]
    
    # Shorten study names for better LaTeX formatting
    latex_df.index = [
        "Aligned (no ang)",
        "Not aligned (no ang)", 
        "Not aligned (w/ ang)"
    ]
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Comparison of Optuna Study Hyperparameters}")
    latex_content.append("\\label{tab:optuna_comparison}")
    latex_content.append("\\resizebox{\\textwidth}{!}{")
    
    # Generate column specification
    n_cols = len(latex_df.columns) + 1  # +1 for row names
    col_spec = "l" + "c" * (n_cols - 1)
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\toprule")
    
    # Header row
    header = "Study & " + " & ".join([f"\\textbf{{{col}}}" for col in latex_df.columns]) + " \\\\"
    latex_content.append(header)
    latex_content.append("\\midrule")
    
    # Data rows
    for idx, row in latex_df.iterrows():
        row_values = []
        for val in row.values:
            # Convert to string and escape LaTeX special characters
            val_str = str(val)
            val_str = val_str.replace('_', '\\_')
            val_str = val_str.replace('nan', '--')
            row_values.append(val_str)
        row_str = f"\\textbf{{{idx}}} & " + " & ".join(row_values) + " \\\\"
        latex_content.append(row_str)
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"LaTeX table saved to: {filename}")
    
    # Also create a simplified version with fewer parameters
    important_params = ['best_value', 'n_trials', 'n_units', 'lr', 'batch_size', 'n_epochs', 'dt']
    available_params = [param for param in important_params if param in latex_df.columns]
    
    if len(available_params) > 2:  # Only create if we have enough parameters
        simple_df = latex_df[available_params]
        
        # Create simplified LaTeX table
        simple_latex = []
        simple_latex.append("\\begin{table}[htbp]")
        simple_latex.append("\\centering")
        simple_latex.append("\\caption{Key Hyperparameters Comparison}")
        simple_latex.append("\\label{tab:optuna_key_params}")
        
        n_cols_simple = len(simple_df.columns) + 1
        col_spec_simple = "l" + "c" * (n_cols_simple - 1)
        simple_latex.append(f"\\begin{{tabular}}{{{col_spec_simple}}}")
        simple_latex.append("\\toprule")
        
        # Header row
        header_simple = "Study & " + " & ".join([f"\\textbf{{{col}}}" for col in simple_df.columns]) + " \\\\"
        simple_latex.append(header_simple)
        simple_latex.append("\\midrule")
        
        # Data rows
        for idx, row in simple_df.iterrows():
            row_values = []
            for val in row.values:
                # Convert to string and escape LaTeX special characters
                val_str = str(val)
                val_str = val_str.replace('_', '\\_')
                val_str = val_str.replace('nan', '--')
                row_values.append(val_str)
            row_str = f"\\textbf{{{idx}}} & " + " & ".join(row_values) + " \\\\"
            simple_latex.append(row_str)
        
        simple_latex.append("\\bottomrule")
        simple_latex.append("\\end{tabular}")
        simple_latex.append("\\end{table}")
        
        # Write simplified table
        simple_filename = filename.replace('.tex', '_simple.tex')
        with open(simple_filename, 'w') as f:
            f.write("\n".join(simple_latex))
        
        print(f"Simplified LaTeX table saved to: {simple_filename}")
    
    # Create custom table with specific parameters and calculated parameter count
    custom_params = ['best\\_value', 'n\\_units', 'n\\_connections', 'non\\_zero\\_elements', 'non\\_zero\\_elements\\_ang', 'steps\\_readout']
    
    # Get the original dataframe (before escaping) to do calculations
    orig_df = df.copy()
    orig_df = orig_df.fillna(0)  # Replace NaN with 0 for calculations
    
    # Calculate parameter count: n_units * 5 * 10 * steps_readout
    orig_df['param_count'] = orig_df['n_units'] * 5 * 10 * orig_df['steps_readout']
    
    # Now create the custom dataframe with escaped names
    custom_df = pd.DataFrame()
    custom_df['best\\_value'] = orig_df['best_value']
    custom_df['n\\_units'] = orig_df['n_units']
    custom_df['n\\_connections'] = orig_df['n_connections']
    custom_df['non\\_zero\\_elements'] = orig_df['non_zero_elements']
    custom_df['non\\_zero\\_elements\\_ang'] = orig_df['non_zero_elements_ang']
    custom_df['steps\\_readout'] = orig_df['steps_readout']
    custom_df['param\\_count'] = orig_df['param_count']
    
    # Round numeric values and replace 0s that were NaN with --
    for col in custom_df.columns:
        if custom_df[col].dtype in ['float64', 'int64']:
            custom_df[col] = custom_df[col].round(0).astype(int)
            # Replace 0s that were originally NaN with --
            if col in ['non\\_zero\\_elements\\_ang']:
                custom_df.loc[orig_df['non_zero_elements_ang'].isna(), col] = '--'
    
    # Set the same row names as before
    custom_df.index = [
        "Aligned (no ang)",
        "Not aligned (no ang)", 
        "Not aligned (w/ ang)"
    ]
    
    # Create custom LaTeX table
    custom_latex = []
    custom_latex.append("\\begin{table}[htbp]")
    custom_latex.append("\\centering")
    custom_latex.append("\\caption{Custom Hyperparameters Comparison with Parameter Count}")
    custom_latex.append("\\label{tab:optuna_custom}")
    
    n_cols_custom = len(custom_df.columns) + 1
    col_spec_custom = "l" + "c" * (n_cols_custom - 1)
    custom_latex.append(f"\\begin{{tabular}}{{{col_spec_custom}}}")
    custom_latex.append("\\toprule")
    
    # Header row
    header_custom = "Study & " + " & ".join([f"\\textbf{{{col}}}" for col in custom_df.columns]) + " \\\\"
    custom_latex.append(header_custom)
    custom_latex.append("\\midrule")
    
    # Data rows
    for idx, row in custom_df.iterrows():
        row_values = []
        for val in row.values:
            val_str = str(val)
            row_values.append(val_str)
        row_str = f"\\textbf{{{idx}}} & " + " & ".join(row_values) + " \\\\"
        custom_latex.append(row_str)
    
    custom_latex.append("\\bottomrule")
    custom_latex.append("\\end{tabular}")
    custom_latex.append("\\end{table}")
    
    # Write custom table
    custom_filename = filename.replace('.tex', '_custom.tex')
    with open(custom_filename, 'w') as f:
        f.write("\n".join(custom_latex))
    
    print(f"Custom LaTeX table saved to: {custom_filename}")
    
    return latex_content

def generate_custom_latex_table(df, filename="optuna_studies_comparison_custom.tex"):
    """Generate a custom LaTeX table with specific parameters and calculated parameter count (transposed)."""
    
    # Select specific columns
    custom_params = ['best_value', 'n_units', 'n_connections', 'non_zero_elements', 'non_zero_elements_ang', 'steps_readout']
    available_params = [param for param in custom_params if param in df.columns]
    
    if len(available_params) < 3:
        print("Not enough parameters available for custom table")
        return
    
    custom_df = df[available_params].copy()
    
    # Calculate parameter count: n_units * 5 * 10 * steps_readout
    custom_df['param_count'] = (custom_df['n_units'] * 5 * 10 * custom_df['steps_readout']).astype(int)
    
    # Round and format values
    custom_df['best_value'] = custom_df['best_value'].round(0).astype(int)
    custom_df['n_units'] = custom_df['n_units'].astype(int)
    custom_df['n_connections'] = custom_df['n_connections'].astype(int)
    custom_df['non_zero_elements'] = custom_df['non_zero_elements'].astype(int)
    custom_df['steps_readout'] = custom_df['steps_readout'].astype(int)
    
    # Handle NaN values in non_zero_elements_ang
    custom_df['non_zero_elements_ang'] = custom_df['non_zero_elements_ang'].fillna(0).astype(int)
    
    # Shorten study names for columns
    custom_df.index = [
        "Aligned (no ang)",
        "Not aligned (no ang)", 
        "Not aligned (w/ ang)"
    ]
    
    # Transpose the dataframe so parameters are rows
    transposed_df = custom_df.T
    
    # Create parameter labels for rows
    param_labels = {
        'best_value': 'Best Accuracy (\\%)',
        'n_units': 'Number of Units',
        'n_connections': 'Connections', 
        'non_zero_elements': 'Non-zero Elements',
        'non_zero_elements_ang': 'Non-zero Angular Elements',
        'steps_readout': 'Readout Steps',
        'param_count': 'Parameter Count'
    }
    
    # Create LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Custom Hyperparameters Comparison with Parameter Count}")
    latex_content.append("\\label{tab:optuna_custom}")
    
    # Generate column specification (parameter names + 3 study columns)
    n_cols = len(transposed_df.columns) + 1  # +1 for parameter names
    col_spec = "l" + "c" * (n_cols - 1)
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\toprule")
    
    # Header row (study names)
    header = "Parameter & " + " & ".join([f"\\textbf{{{col}}}" for col in transposed_df.columns]) + " \\\\"
    latex_content.append(header)
    latex_content.append("\\midrule")
    
    # Data rows (each parameter)
    for param_name, row in transposed_df.iterrows():
        param_label = param_labels.get(param_name, param_name.replace('_', '\\_'))
        row_values = [str(int(val)) if isinstance(val, (int, float)) else str(val) for val in row.values]
        row_str = f"\\textbf{{{param_label}}} & " + " & ".join(row_values) + " \\\\"
        latex_content.append(row_str)
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(filename, 'w') as f:
        f.write("\n".join(latex_content))
    
    print(f"Custom LaTeX table saved to: {filename}")
    
    return latex_content

if __name__ == "__main__":
    # Get the parent directory (assuming script is in a subdirectory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading Optuna studies...")
    print(f"Looking for databases in: {current_dir}/optuna_databases/")
    
    # Compare the studies
    comparison_df = compare_studies(study_names, current_dir)
    
    if comparison_df is not None:
        # Print the comparison
        print_comparison_table(comparison_df)
        
        # Save to CSV
        save_comparison_to_csv(comparison_df, "optuna_studies_comparison.csv")
        
        # Generate LaTeX tables
        generate_latex_table(comparison_df, "optuna_studies_comparison.tex")
        generate_custom_latex_table(comparison_df, "optuna_studies_comparison_custom.tex")
        
        # Additional analysis
        print("\n" + "=" * 100)
        print("ADDITIONAL ANALYSIS")
        print("=" * 100)
        
        # Find best performing study
        best_study = comparison_df['best_value'].idxmax()
        best_score = comparison_df.loc[best_study, 'best_value']
        print(f"\nBest performing study: {best_study}")
        print(f"Best validation accuracy: {best_score:.4f}")
        
        # Show parameter ranges
        print("\nParameter Ranges Across Studies:")
        print("-" * 40)
        numeric_cols = comparison_df.select_dtypes(include=['float64', 'int64']).columns
        param_cols = [col for col in numeric_cols if col not in ['best_value', 'n_trials']]
        
        for param in sorted(param_cols):
            if param in comparison_df.columns:
                values = comparison_df[param].dropna()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()
                    mean_val = values.mean()
                    print(f"{param:<25}: {min_val:.4f} - {max_val:.4f} (mean: {mean_val:.4f})")
    else:
        print("No studies could be loaded for comparison!")
