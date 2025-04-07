#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.transforms import Affine2D

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            
        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)
            
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
                
        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
                
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            
        def _gen_axes_patch(self):
            if frame == 'circle':
                return plt.Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return plt.RegularPolygon((0.5, 0.5), num_vars,
                                          radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
                
        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)
                
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon produces a polygon of radius 1
                # centered at (0, 0) but we want a polygon
                # of radius 0.5 centered at (0.5, 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    
    # Register the projection with Matplotlib
    register_projection(RadarAxes)
    return theta


def find_metric_files(base_dir='models'):
    """Find all metric files in the models directory"""
    metric_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_test_metrics.pkl'):
                metric_files.append(os.path.join(root, file))
    return metric_files


def load_metrics(file_path):
    """Load metrics from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def extract_model_name(file_path):
    """Extract model name from file path"""
    base_name = os.path.basename(file_path)
    model_name = base_name.replace('_test_metrics.pkl', '')
    return model_name


def clean_model_name(model_name):
    """Convert model name to a nicer format for the paper"""
    if model_name == 'GCNLinkPrediction':
        return 'GCN'
    elif model_name == 'GATLinkPrediction':
        return 'GAT'
    elif model_name == 'GraphSAGELinkPrediction':
        return 'GraphSAGE'
    elif model_name == 'SEALLinkPrediction':
        return 'SEAL'
    elif model_name == 'HeuristicLinkPrediction':
        return 'Heuristic (RF)'
    elif model_name == 'RF':
        return 'Random Forest'
    elif model_name == 'GBDT':
        return 'GBDT'
    elif model_name == 'MatrixFactorization':
        return 'MF'
    else:
        return model_name


def generate_report(metric_files):
    """Generate a comprehensive report of all metrics"""
    all_metrics = []
    
    for file_path in metric_files:
        metrics = load_metrics(file_path)
        if metrics is None:
            continue
        
        model_name = extract_model_name(file_path)
        model_name = clean_model_name(model_name)
        
        # Extract key metrics
        metric_row = {
            'Model': model_name,
            'AUC': metrics.get('auc', 0),
            'AP': metrics.get('ap', 0),
            'F1': metrics.get('f1', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0)
        }
        
        all_metrics.append(metric_row)
    
    # Create a DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by AUC
    df = df.sort_values('AUC', ascending=False)
    
    return df


def create_radar_plot(df, output_file='radar_comparison.png'):
    """Create a radar plot comparing different models"""
    # Select top N models
    N = min(5, len(df))
    top_models = df.head(N)
    
    # Extract metrics for the radar plot
    metrics = ['AUC', 'AP', 'F1', 'Precision', 'Recall']
    
    # Create the radar plot
    theta = radar_factory(len(metrics), frame='polygon')
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    
    # Set up colors for different model types
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    
    # Plot each model
    for i, (idx, row) in enumerate(top_models.iterrows()):
        values = [row[metric] for metric in metrics]
        ax.plot(theta, values, color=colors[i], label=row['Model'])
        ax.fill(theta, values, color=colors[i], alpha=0.25)
    
    # Set the labels
    ax.set_varlabels(metrics)
    
    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Set title
    plt.title('Model Performance Comparison', size=15, y=1.1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Radar plot saved to {output_file}")
    
    # Return the figure
    return fig


def main():
    # Find all metric files
    metric_files = find_metric_files()
    
    if not metric_files:
        print("No metric files found. Please train models first.")
        return
    
    print(f"Found {len(metric_files)} metric files.")
    
    # Generate report
    df = generate_report(metric_files)
    
    # Create radar plot
    create_radar_plot(df, output_file='radar_comparison.png')
    
    # Print the DataFrame for verification
    print("\nMetrics used for radar plot:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main() 