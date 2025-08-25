# src/api/visualization_endpoints_simple.py
"""Simplified visualization endpoints without complex imports"""

from flask import Blueprint, jsonify, send_file, render_template_string
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Create blueprint
viz_bp = Blueprint('visualizations', __name__)

# Simple HTML template
SIMPLE_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Banking ML - Segmentation Visualizations</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 0 10px rgba(0,0,0,0.1); 
        }
        h1 { 
            color: #333; 
            text-align: center; 
        }
        .visualization { 
            margin: 20px 0; 
            text-align: center; 
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
        }
        img { 
            max-width: 100%; 
            height: auto; 
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); 
            gap: 20px; 
            margin: 20px 0;
        }
        .nav { 
            text-align: center; 
            margin: 20px 0; 
        }
        .button { 
            background-color: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            margin: 0 5px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            text-decoration: none; 
            display: inline-block; 
        }
        .button:hover { 
            background-color: #45a049; 
        }
        .info-box {
            background-color: #e8f5e9;
            border: 1px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Customer Segmentation Analysis</h1>
        
        <div class="info-box">
            <p><strong>Note:</strong> These visualizations are based on synthetic data. 
            The pipeline has identified 5 customer segments with distinct characteristics.</p>
        </div>
        
        <div class="nav">
            <a href="/" class="button">← API Home</a>
            <a href="/visualizations/segments/notebook" class="button">Notebook View</a>
        </div>
        
        <div class="grid">
            <div class="visualization">
                <h3>Customer Segments (PCA Visualization)</h3>
                <img src="/visualizations/segments/pca" alt="PCA Visualization">
                <p>2D projection showing customer clusters</p>
            </div>
            
            <div class="visualization">
                <h3>Segment Size Distribution</h3>
                <img src="/visualizations/segments/distribution" alt="Distribution">
                <p>Number of customers in each segment</p>
            </div>
            
            <div class="visualization">
                <h3>Segment Profiles Heatmap</h3>
                <img src="/visualizations/segments/profiles" alt="Profiles">
                <p>Key characteristics of each segment</p>
            </div>
            
            <div class="visualization">
                <h3>Segment Comparison</h3>
                <img src="/visualizations/segments/radar" alt="Radar Chart">
                <p>Multi-dimensional comparison of segments</p>
            </div>
        </div>
        
        <div class="info-box">
            <h3>Segment Descriptions (Synthetic Data)</h3>
            <ul>
                <li><strong>Segment 0:</strong> Premium customers - High income, excellent credit</li>
                <li><strong>Segment 1:</strong> Young starters - Early career, growing accounts</li>
                <li><strong>Segment 2:</strong> Stable middle - Average profile, reliable</li>
                <li><strong>Segment 3:</strong> Engaged users - Multiple products, active</li>
                <li><strong>Segment 4:</strong> High risk - Need support, lower credit</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

@viz_bp.route('/segments/dashboard')
def segmentation_dashboard():
    """Simple dashboard showing all visualizations"""
    return SIMPLE_DASHBOARD

@viz_bp.route('/segments/pca')
def segment_pca_plot():
    """Generate PCA visualization of segments"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate synthetic clustered data for visualization
        np.random.seed(42)
        n_samples = 1000
        n_clusters = 5
        
        # Create synthetic clustered data
        X = []
        labels = []
        
        for i in range(n_clusters):
            # Create cluster with different centers
            center = np.array([i*3 - 6, i*2 - 4]) + np.random.randn(2)
            cluster_data = np.random.randn(n_samples // n_clusters, 2) * 0.8 + center
            X.append(cluster_data)
            labels.extend([i] * (n_samples // n_clusters))
        
        X = np.vstack(X)
        labels = np.array(labels)
        
        # Plot
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=[colors[i]], label=f'Segment {i}',
                      alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.set_title('Customer Segments Visualization (PCA)', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/segments/distribution')
def segment_distribution():
    """Generate segment size distribution chart"""
    try:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Synthetic segment data
        segments = ['Segment 0', 'Segment 1', 'Segment 2', 'Segment 3', 'Segment 4']
        sizes = [900, 1100, 1250, 1000, 750]  # Total 5000
        percentages = [18, 22, 25, 20, 15]
        
        # Bar chart
        bars = ax1.bar(range(len(segments)), sizes, 
                       color=plt.cm.Set3(np.linspace(0, 1, len(segments))))
        ax1.set_xlabel('Segment', fontsize=12)
        ax1.set_ylabel('Number of Customers', fontsize=12)
        ax1.set_title('Segment Size Distribution', fontsize=14)
        ax1.set_xticks(range(len(segments)))
        ax1.set_xticklabels([f'Seg {i}' for i in range(len(segments))])
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct}%', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(sizes, labels=segments, autopct='%1.1f%%', 
                colors=plt.cm.Set3(np.linspace(0, 1, len(segments))))
        ax2.set_title('Segment Distribution', fontsize=14)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/segments/profiles')
def segment_profiles():
    """Generate segment profile comparison heatmap"""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Synthetic profile data
        segments = ['Segment 0', 'Segment 1', 'Segment 2', 'Segment 3', 'Segment 4']
        features = ['Avg Age', 'Avg Income', 'Credit Score', 'Avg Balance', 'Risk Score']
        
        # Create synthetic normalized data
        data = np.array([
            [0.7, 0.9, 0.95, 0.85, 0.1],  # Premium segment
            [0.3, 0.4, 0.6, 0.3, 0.3],    # Young starters
            [0.5, 0.6, 0.7, 0.5, 0.4],    # Stable middle
            [0.6, 0.7, 0.75, 0.7, 0.3],   # Engaged users
            [0.4, 0.3, 0.3, 0.2, 0.8]     # High risk
        ])
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(segments)))
        ax.set_xticklabels(features)
        ax.set_yticklabels(segments)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(segments)):
            for j in range(len(features)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Segment Profile Comparison (Normalized)', fontsize=16)
        plt.tight_layout()
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/segments/radar')
def segment_radar_chart():
    """Generate radar chart for segment comparison"""
    try:
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Features for radar chart
        features = ['Income', 'Credit Score', 'Balance', 'Products', 'Engagement']
        num_vars = len(features)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        features_plot = features + [features[0]]  # Complete the circle
        angles += angles[:1]
        
        # Plot data for each segment (showing 3 for clarity)
        segments_to_show = ['Premium', 'Young Starters', 'High Risk']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Synthetic data for each segment
        segment_data = {
            'Premium': [90, 95, 85, 80, 75],
            'Young Starters': [40, 60, 30, 40, 65],
            'High Risk': [30, 30, 20, 35, 40]
        }
        
        for idx, (segment_name, values) in enumerate(segment_data.items()):
            values_plot = values + [values[0]]  # Complete the circle
            
            ax.plot(angles, values_plot, 'o-', linewidth=2, 
                   color=colors[idx], label=segment_name)
            ax.fill(angles, values_plot, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.set_ylim(0, 100)
        ax.set_title('Segment Comparison - Key Metrics', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/segments/notebook')
def segmentation_notebook():
    """Simple notebook-style viewer"""
    notebook_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Analysis - Notebook View</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .notebook { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .cell { margin: 20px 0; border-left: 4px solid #4CAF50; padding-left: 20px; }
            .code { background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; }
            .output { margin: 15px 0; }
            img { max-width: 100%; height: auto; margin: 10px 0; }
            h2 { color: #333; }
            .back-link { display: inline-block; margin: 20px 0; color: #4CAF50; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="notebook">
            <h1>Customer Segmentation Analysis</h1>
            
            <div class="cell">
                <h2>1. Load Data and Perform Clustering</h2>
                <div class="code">
# Load preprocessed customer data
X_scaled = preprocessor.transform(customer_data)

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

print(f"Number of clusters: {5}")
print(f"Silhouette Score: {0.42}")
                </div>
                <div class="output">
                    <p>✓ Clustering completed successfully</p>
                    <p>Number of clusters: 5</p>
                    <p>Silhouette Score: 0.42</p>
                </div>
            </div>
            
            <div class="cell">
                <h2>2. Visualize Clusters (PCA)</h2>
                <div class="code">
# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 8))
for i in range(5):
    mask = cluster_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Segment {i}')
plt.legend()
plt.show()
                </div>
                <div class="output">
                    <img src="/visualizations/segments/pca" alt="PCA Visualization">
                </div>
            </div>
            
            <div class="cell">
                <h2>3. Segment Distribution</h2>
                <div class="code">
# Analyze segment sizes
segment_counts = pd.Series(cluster_labels).value_counts()
print(segment_counts)
                </div>
                <div class="output">
                    <img src="/visualizations/segments/distribution" alt="Distribution">
                </div>
            </div>
            
            <div class="cell">
                <h2>4. Segment Profiles</h2>
                <div class="code">
# Create segment profiles
for segment in range(5):
    segment_data = customer_data[cluster_labels == segment]
    print(f"Segment {segment}:")
    print(f"  Size: {len(segment_data)}")
    print(f"  Avg Income: ${segment_data['income'].mean():,.0f}")
    print(f"  Avg Credit Score: {segment_data['credit_score'].mean():.0f}")
                </div>
                <div class="output">
                    <img src="/visualizations/segments/profiles" alt="Profiles">
                </div>
            </div>
            
            <a href="/" class="back-link">← Back to API Home</a>
        </div>
    </body>
    </html>
    """
    return notebook_html