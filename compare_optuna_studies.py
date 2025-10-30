#!/usr/bin/env python3
"""
Script to compare hyperparameters from multiple Optuna studies.
"""

import optuna
import pandas as pd
import os

# Configuration for different comparison scenarios
# Scenario 1: Multiple databases, same study name
SCENARIO_1_CONFIG = {
    "db_names": [
        "unicycle_nets_mnist_all_digits_logreg",
        "unicycle_nets_mnist_all_digits_logreg_not_aligned",
        "unicycle_nets_mnist_all_digits_logreg_not_aligned_w_input_w_connections"
    ],
    "study_name": "alignment_ang_input_ang_connections_as_params_true",
    "mode": "multiple_dbs_one_study"
}

# Scenario 2: One database, multiple studies
SCENARIO_2_CONFIG = {
    "db_name": "unicycle_nets_lorenz",  # Replace with your actual database name
    "study_names": [
          # Replace with your actual study names
        "lorenz_prediction_esn_lag25"
    ],
    "mode": "one_db_multiple_studies"
}

# Choose which scenario to use (1 or 2)
ACTIVE_SCENARIO = 2

# Set active configuration based on scenario
if ACTIVE_SCENARIO == 1:
    config = SCENARIO_1_CONFIG
else:
    config = SCENARIO_2_CONFIG
def load_study_best_params_single(db_name, study_name, base_dir="."):
    """Load the best parameters from an Optuna study (single database, single study)."""
    try:
        storage_name = f"sqlite:///{base_dir}/optuna_databases/{db_name}.db"
        study = optuna.load_study(storage=storage_name, study_name=study_name)
        
        # Get best trial info
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_params['best_value'] = best_trial.value
        best_params['n_trials'] = len(study.trials)
        
        return best_params
    except Exception as e:
        print(f"Error loading study {study_name} from {db_name}: {e}")
        return None

def compare_studies_multiple_dbs(db_names, study_name, base_dir="."):
    """Compare hyperparameters across multiple databases with the same study name."""
    
    # Load all studies
    studies_data = {}
    for db_name in db_names:
        params = load_study_best_params_single(db_name, study_name, base_dir)
        if params is not None:
            studies_data[db_name] = params  # Use db_name as key
        else:
            print(f"Skipping study: {db_name}")
    
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

def compare_studies_one_db(db_name, study_names, base_dir="."):
    """Compare hyperparameters across multiple studies in the same database."""
    
    # Load all studies
    studies_data = {}
    for study_name in study_names:
        params = load_study_best_params_single(db_name, study_name, base_dir)
        if params is not None:
            studies_data[study_name] = params  # Use study_name as key
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
    study_name_mapping = {
        "unicycle_nets_mnist_all_digits_logreg": "Aligned (no ang)",
        "unicycle_nets_mnist_all_digits_logreg_not_aligned": "Not aligned (no ang)", 
        "unicycle_nets_mnist_all_digits_logreg_not_aligned_w_input_w_connections": "Not aligned (w/ ang)"
    }
    
    # Only map the studies that actually exist in the DataFrame
    new_index = []
    for original_name in latex_df.index:
        if original_name in study_name_mapping:
            new_index.append(study_name_mapping[original_name])
        else:
            new_index.append(original_name)
    
    latex_df.index = new_index
    
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
    custom_params = ['best\\_value', 'n\\_units', 'n\\_connections', 'non\\_zero\\_fraction', 'non\\_zero\\_fraction\\_ang', 'steps\\_readout']
    
    # Get the original dataframe (before escaping) to do calculations
    orig_df = df.copy()
    orig_df = orig_df.fillna(0)  # Replace NaN with 0 for calculations
    
    # Calculate parameter count: n_units * 5 * 10 * steps_readout
    orig_df['param_count'] = orig_df['n_units'] * 5 * 10 * orig_df['steps_readout']
    
    # Now create the custom dataframe with escaped names
    custom_df = pd.DataFrame()
    custom_df['best\\_value'] = orig_df['best_value']
    custom_df['n\\_units'] = orig_df['n_units']
    custom_df['n\\_connections'] = orig_df['n_connections_fraction']
    custom_df['non\\_zero\\_fraction'] = orig_df['non_zero_fraction']
    custom_df['non\\_zero\\_fraction\\_ang'] = orig_df['non_zero_fraction_ang']
    custom_df['steps\\_readout'] = orig_df['steps_readout']
    custom_df['param\\_count'] = orig_df['param_count']
    
    # Round numeric values and replace 0s that were NaN with --
    for col in custom_df.columns:
        if custom_df[col].dtype in ['float64', 'int64']:
            custom_df[col] = custom_df[col].round(0).astype(int)
            # Replace 0s that were originally NaN with --
            if col in ['non\\_zero\\_fraction\\_ang']:
                custom_df.loc[orig_df['non_zero_fraction_ang'].isna(), col] = '--'
    
    # Set the same row names as before
    study_name_mapping = {
        "unicycle_nets_mnist_all_digits_logreg": "Aligned (no ang)",
        "unicycle_nets_mnist_all_digits_logreg_not_aligned": "Not aligned (no ang)", 
        "unicycle_nets_mnist_all_digits_logreg_not_aligned_w_input_w_connections": "Not aligned (w/ ang)"
    }
    
    # Only map the studies that actually exist in the DataFrame
    new_index = []
    for original_name in custom_df.index:
        if original_name in study_name_mapping:
            new_index.append(study_name_mapping[original_name])
        else:
            new_index.append(original_name)
    
    custom_df.index = new_index
    
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
    custom_params = ['best_value', 'n_units', 'n_connections_fraction', 'non_zero_fraction', 'non_zero_fraction_ang', 'steps_readout']
    available_params = [param for param in custom_params if param in df.columns]
    
    if len(available_params) < 3:
        print("Not enough parameters available for custom table")
        return
    
    custom_df = df[available_params].copy()
    
    # Calculate parameter count: n_units * 5 * 10 * steps_readout
    custom_df['param_count'] = (custom_df['n_units'] * 5 * 10 * custom_df['steps_readout']).astype(int)
    
    # Store original NaN mask for non_zero_fraction_ang before any conversions
    ang_nan_mask = custom_df['non_zero_fraction_ang'].isna() if 'non_zero_fraction_ang' in custom_df.columns else pd.Series([False] * len(custom_df))
    
    # Round and format values, but preserve NaN for proper handling
    # Convert best_value to percentage (multiply by 100) and round to 1 decimal place
    custom_df['best_value'] = (custom_df['best_value'] * 100).round(1)
    custom_df['n_units'] = custom_df['n_units'].astype(int)
    if 'n_connections_fraction' in custom_df.columns:
        custom_df['n_connections_fraction'] = custom_df['n_connections_fraction'].astype(float)
    custom_df['non_zero_fraction'] = custom_df['non_zero_fraction'].astype(int)
    custom_df['steps_readout'] = custom_df['steps_readout'].astype(int)
    
    # Handle NaN values in non_zero_fraction_ang properly
    if 'non_zero_fraction_ang' in custom_df.columns:
        # Fill NaN with a placeholder value first, then convert to int, then replace placeholder with '--'
        custom_df['non_zero_fraction_ang'] = custom_df['non_zero_fraction_ang'].fillna(-999).astype(int)
    
    # Shorten study names for columns
    study_name_mapping = {
        "unicycle_nets_mnist_all_digits_logreg": "Aligned (no ang)",
        "unicycle_nets_mnist_all_digits_logreg_not_aligned": "Not aligned (no ang)", 
        "unicycle_nets_mnist_all_digits_logreg_not_aligned_w_input_w_connections": "Not aligned (w/ ang)"
    }
    
    # Only map the studies that actually exist in the DataFrame
    new_index = []
    for original_name in custom_df.index:
        if original_name in study_name_mapping:
            new_index.append(study_name_mapping[original_name])
        else:
            new_index.append(original_name)
    
    custom_df.index = new_index
    
    # Add a baseline/comparison column with N/A values 
    # We'll add this after transposing to avoid data type conflicts
    baseline_column_data = ['N/A'] * len(custom_df)
    
    # Transpose the dataframe so parameters are rows
    transposed_df = custom_df.T
    
    # Add baseline column after transposing to avoid data type issues
    baseline_data = ['N/A'] * len(transposed_df)
    transposed_df['Baseline/Other'] = baseline_data
    
    # Create parameter labels for rows
    param_labels = {
        'best_value': 'Best Accuracy (\\%)',
        'n_units': 'Number of Units',
        'n_connections_fraction': 'Connection Fraction', 
        'non_zero_fraction': 'Non-zero Fraction',
        'non_zero_fraction_ang': 'Non-zero Angular Fraction',
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
        row_values = []
        for col_name, val in zip(transposed_df.columns, row.values):
            if col_name == 'Baseline/Other':
                # Always show N/A for baseline column except for best_value and param_count
                if param_name in ['best_value', 'param_count']:
                    row_values.append('N/A')
                else:
                    row_values.append('N/A')
            elif param_name == 'non_zero_fraction_ang' and val == -999:
                # Replace placeholder with --
                row_values.append('--')
            elif param_name == 'best_value':
                # Format as percentage with 1 decimal place
                row_values.append(f"{val:.1f}" if isinstance(val, (int, float)) else str(val))
            elif param_name == 'n_connections_fraction':
                # Format as float with proper precision
                row_values.append(f"{val:.3f}" if isinstance(val, (int, float)) else str(val))
            elif isinstance(val, (int, float)) and val != -999:
                # Format as integer for most parameters
                row_values.append(str(int(val)))
            else:
                row_values.append(str(val))
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
    print(f"Using scenario {ACTIVE_SCENARIO}: {config['mode']}")
    
    # Compare the studies based on the active scenario
    if config['mode'] == 'multiple_dbs_one_study':
        comparison_df = compare_studies_multiple_dbs(config['db_names'], config['study_name'], current_dir)
        scenario_name = "multiple_databases"
    else:  # one_db_multiple_studies
        comparison_df = compare_studies_one_db(config['db_name'], config['study_names'], current_dir)
        scenario_name = "multiple_studies"
    
    if comparison_df is not None:
        # Print the comparison
        print_comparison_table(comparison_df)
        
        # Save to CSV with scenario-specific filename
        csv_filename = f"optuna_studies_comparison_{scenario_name}.csv"
        save_comparison_to_csv(comparison_df, csv_filename)
        
        # Generate LaTeX tables with scenario-specific filenames
        tex_filename = f"optuna_studies_comparison_{scenario_name}.tex"
        custom_tex_filename = f"optuna_studies_comparison_{scenario_name}_custom.tex"
        
        generate_latex_table(comparison_df, tex_filename)
        generate_custom_latex_table(comparison_df, custom_tex_filename)
        
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
