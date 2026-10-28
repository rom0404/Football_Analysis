#!/usr/bin/env python3
"""
Player Profile Analysis - Configurable Scout Tool
Analyze any player's statistical profile vs. position peers, find comparables, generate scout card.

Usage:
    python analyze_player.py                    # Run with default CONFIG
    python analyze_player.py --player "Haaland" --position "FW"  # Override via CLI
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
from math import pi
import warnings
import argparse
import sys
import os
import json
import tempfile
from pathlib import Path

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
CONFIG = {
    "csv_path": "players_data-2024_2025.csv",
    "player_name": "Martinez",          # substring match (case-insensitive)
    "position_filter": "DF",            # position substring filter (e.g., "DF", "MF", "FW", "GK")
    "min_minutes": 900,                 # minimum minutes played
    "max_age": 30,                      # maximum age for comparable search
    "n_neighbors": 20,                  # number of similar players to find
    "top_k_anomalies": 12,              # top anomaly metrics for radar/KNN
    "percentile_threshold": 90,         # highlight metrics above this percentile
}


def load_and_filter(csv_path: str, config: dict) -> pd.DataFrame:
    """Load CSV and filter by position + minimum minutes."""
    # Try multiple encodings to handle mojibake
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            # Check if special characters look reasonable
            sample = df['Player'].iloc[0] if len(df) > 0 else ''
            if chr(65533) not in sample:
                break
        except UnicodeDecodeError:
            continue
    df_filtered = df[df['Min'] >= config['min_minutes']].copy()
    df_pos = df_filtered[df_filtered['Pos'].str.contains(config['position_filter'], na=False, case=False)].copy()
    print(f"Loaded {len(df)} players -> {len(df_pos)} {config['position_filter']}s with >={config['min_minutes']} min")
    return df_pos


def find_player(df: pd.DataFrame, player_name: str) -> pd.Series:
    """Find player row by case-insensitive substring match on Player column."""
    matches = df[df['Player'].str.contains(player_name, case=False, na=False)]
    if len(matches) == 0:
        raise ValueError(f"No player found matching '{player_name}'")
    if len(matches) > 1:
        print(f"Multiple matches for '{player_name}': {matches['Player'].tolist()}")
        print(f"Using first match: {matches.iloc[0]['Player']}")
    return matches.iloc[0]


def compute_percentiles(df: pd.DataFrame, player_row: pd.Series, numeric_cols: list) -> dict:
    """Compute percentile rank of player for each numeric metric."""
    percentiles = {}
    for col in numeric_cols:
        if col in df.columns:
            # Only consider non-null values
            valid = df[col].dropna()
            if len(valid) > 0:
                player_val = player_row[col]
                if pd.notna(player_val):
                    pct = (valid < player_val).sum() / len(valid) * 100
                    percentiles[col] = round(pct, 1)
    return percentiles


def find_anomalies(df: pd.DataFrame, player_row: pd.Series, config: dict) -> dict:
    """
    Find metrics where player excels (above percentile threshold).
    Returns dict of metric -> {value, percentile, rank}
    """
    # Select numeric columns for analysis (exclude metadata)
    exclude_cols = {'Rk', 'Age', 'Born', 'MP', 'Starts', 'Min', '90s', 'Player', 'Nation', 'Pos', 'Squad', 'Comp'}
    # Also exclude any column with 'Rk_' or '_stats_' prefix/suffix or Unnamed
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols and not c.startswith('Rk_') and not '_stats_' in c]

    percentiles = compute_percentiles(df, player_row, numeric_cols)

    # Filter to metrics above threshold
    anomalies = {}
    for metric, pct in percentiles.items():
        if pct >= config['percentile_threshold']:
            anomalies[metric] = {
                'value': player_row[metric],
                'percentile': pct,
                'rank': (df[metric].dropna() > player_row[metric]).sum() + 1
            }

    # Sort by percentile descending, take top_k
    sorted_anomalies = dict(sorted(anomalies.items(), key=lambda x: x[1]['percentile'], reverse=True))
    top_anomalies = dict(list(sorted_anomalies.items())[:config['top_k_anomalies']])

    print(f"Found {len(top_anomalies)} metrics above {config['percentile_threshold']}th percentile")
    return top_anomalies


def find_neighbors(df: pd.DataFrame, player_row: pd.Series, anomalies: dict, config: dict) -> pd.DataFrame:
    """
    Find similar players using KNN on anomaly metrics.
    Returns DataFrame with neighbor players and similarity scores.
    """
    anomaly_metrics = list(anomalies.keys())
    if len(anomaly_metrics) < 2:
        print("Warning: Not enough anomaly metrics for KNN, using all numeric columns")
        exclude_cols = {'Rk', 'Age', 'Born', 'MP', 'Starts', 'Min', '90s'}
        anomaly_metrics = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c not in exclude_cols and not c.startswith('Rk_') and not '_stats_' in c]

    # Prepare feature matrix
    df_clean = df[anomaly_metrics].dropna()
    if len(df_clean) < config['n_neighbors'] + 1:
        print(f"Warning: Not enough data for KNN ({len(df_clean)} rows), reducing neighbors")
        config['n_neighbors'] = max(1, len(df_clean) - 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean)

    # Find player index in clean dataframe
    player_idx_in_clean = df_clean.index.get_loc(player_row.name) if player_row.name in df_clean.index else 0

    # Fit KNN
    nn = NearestNeighbors(n_neighbors=config['n_neighbors'] + 1, metric='euclidean')
    nn.fit(X)

    distances, indices = nn.kneighbors([X[player_idx_in_clean]])

    # Get neighbor rows (skip first which is the player themselves)
    neighbor_indices = df_clean.index[indices[0][1:]]
    neighbor_distances = distances[0][1:]

    neighbors = df.loc[neighbor_indices].copy()
    neighbors['similarity'] = 1 / (1 + neighbor_distances)  # Convert distance to similarity
    neighbors = neighbors.sort_values('similarity', ascending=False)

    return neighbors


def run_analysis(config: dict) -> dict:
    """Main analysis pipeline."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {config['player_name']} ({config['position_filter']})")
    print(f"{'='*60}")

    # Load and filter data
    df = load_and_filter(config['csv_path'], config)

    # Find target player
    player_row = find_player(df, config['player_name'])
    player_idx = player_row.name
    player_name_safe = str(player_row['Player']).encode('ascii', 'replace').decode('ascii')
    squad_safe = str(player_row['Squad']).encode('ascii', 'replace').decode('ascii')
    print(f"Target: {player_name_safe} ({squad_safe}, {player_row['Pos']}, Age {player_row['Age']}, {player_row['Min']} min)")

    # Compute anomalies
    anomalies = find_anomalies(df, player_row, config)

    # Find comparable players (filter by max_age for comps)
    df_comps = df[df['Age'] <= config['max_age']].copy()
    neighbors = find_neighbors(df_comps, player_row, anomalies, config)

    # Filter neighbors to exclude target player
    neighbors = neighbors[neighbors.index != player_idx]

    print(f"\nTop {len(neighbors)} Comparables:")
    for _, row in neighbors.head(10).iterrows():
        player = str(row['Player']).encode('ascii', 'replace').decode('ascii')
        squad = str(row['Squad']).encode('ascii', 'replace').decode('ascii')
        print(f"  {player:25s} {squad:20s} Age {row['Age']:4.1f} Sim: {row['similarity']:.3f}")

    return {
        'df': df,
        'player_row': player_row,
        'player_idx': player_idx,
        'anomalies': anomalies,
        'neighbors': neighbors,
        'config': config
    }


def plot_radar_chart(df: pd.DataFrame, player_idx: int, anomalies: dict, config: dict) -> plt.Figure:
    """Create radar chart of top anomaly metrics."""
    if not anomalies:
        print("No anomalies to plot")
        return None

    metrics = list(anomalies.keys())
    values = [anomalies[m]['percentile'] for m in metrics]

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the loop

    # Values for radar (percentiles)
    vals = values + values[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot
    ax.plot(angles, vals, 'o-', linewidth=2, color='#1f77b4', label=config['player_name'])
    ax.fill(angles, vals, alpha=0.25, color='#1f77b4')

    # Fix axis to go clockwise from top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each metric
    plt.xticks(angles[:-1], [m.replace('_', ' ').replace('Stats', '') for m in metrics], size=9)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 90, 95, 99], ["25", "50", "75", "90", "95", "99"], color="grey", size=8)
    plt.ylim(0, 100)

    # Title
    player_name = df.loc[player_idx, 'Player'] if player_idx in df.index else config['player_name']
    plt.title(f"{player_name} - Top {len(metrics)} Percentile Metrics\nvs {config['position_filter']} Peers (>={config['min_minutes']} min)",
              size=14, y=1.08, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{player_name.replace(' ', '_')}_radar.png", dpi=150, bbox_inches='tight')
    print(f"Saved radar chart to {player_name.replace(' ', '_')}_radar.png")
    return fig


def plot_percentile_bars(anomalies: dict, config: dict) -> plt.Figure:
    """Create horizontal bar chart of percentile rankings."""
    if not anomalies:
        print("No anomalies to plot")
        return None

    metrics = list(anomalies.keys())
    percentiles = [anomalies[m]['percentile'] for m in metrics]
    values = [anomalies[m]['value'] for m in metrics]

    # Reverse for horizontal bar (highest at top)
    metrics = metrics[::-1]
    percentiles = percentiles[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(metrics) * 0.4)))

    # Color gradient based on percentile
    colors = plt.cm.RdYlGn([p/100 for p in percentiles])

    bars = ax.barh(range(len(metrics)), percentiles, color=colors, edgecolor='black', height=0.7)

    # Add value labels on bars
    for i, (bar, pct, val) in enumerate(zip(bars, percentiles, values)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% (val: {val:.2f})', va='center', fontsize=9)

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace('_', ' ').replace('Stats', '') for m in metrics], fontsize=10)
    ax.set_xlabel('Percentile Rank vs Position Peers', fontsize=12)
    ax.set_xlim(0, 110)
    ax.axvline(x=config['percentile_threshold'], color='red', linestyle='--', alpha=0.7,
               label=f'Threshold ({config["percentile_threshold"]}%)')
    ax.legend()

    plt.title(f"Percentile Rankings - {config['player_name']} ({config['position_filter']})",
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{config['player_name'].replace(' ', '_')}_percentiles.png", dpi=150, bbox_inches='tight')
    print(f"Saved percentile bars to {config['player_name'].replace(' ', '_')}_percentiles.png")
    return fig


def generate_scout_card(player_row: pd.Series, anomalies: dict, neighbors: pd.DataFrame, config: dict) -> str:
    """Generate markdown scout card."""
    def safe(s):
        """Convert string to ASCII-safe version."""
        return str(s).encode('ascii', 'replace').decode('ascii')

    lines = []
    lines.append(f"# [ENG] Scout Card: {safe(player_row['Player'])}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Header info
    lines.append(f"**Club:** {safe(player_row['Squad'])}  |  **Position:** {safe(player_row['Pos'])}  |  **Age:** {player_row['Age']}  |  **Minutes:** {player_row['Min']:,}")
    lines.append(f"**Nation:** {safe(player_row['Nation'])}  |  **Competition:** {safe(player_row['Comp'])}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Key Strengths (Anomalies)
    lines.append("## [KEY] Key Strengths (Percentile vs Position Peers)")
    lines.append("")
    if anomalies:
        lines.append("| Metric | Value | Percentile | Rank |")
        lines.append("|--------|-------|------------|------|")
        for metric, data in anomalies.items():
            metric_display = metric.replace('_', ' ').replace('Stats', '')
            lines.append(f"| {safe(metric_display)} | {data['value']:.2f} | **{data['percentile']:.1f}%** | #{data['rank']} |")
    else:
        lines.append("*No metrics above threshold*")
    lines.append("")

    # Comparable Players
    lines.append("## [COMP] Comparable Players")
    lines.append("")
    lines.append(f"*Filtered: {config['position_filter']}s, Age <={config['max_age']}, >={config['min_minutes']} min*")
    lines.append("")
    lines.append("| Rank | Player | Club | Age | Min | Similarity |")
    lines.append("|------|--------|------|-----|-----|------------|")
    for i, (_, row) in enumerate(neighbors.head(10).iterrows(), 1):
        lines.append(f"| {i} | {safe(row['Player'])} | {safe(row['Squad'])} | {row['Age']:.1f} | {row['Min']:,} | {row['similarity']:.3f} |")
    lines.append("")

    # Summary
    lines.append("---")
    lines.append("")
    lines.append(f"*Analysis based on {config['position_filter']}s with >={config['min_minutes']} min in 2024-25 season*")
    lines.append(f"*Anomaly threshold: {config['percentile_threshold']}th percentile | Top {config['top_k_anomalies']} metrics shown*")

    return "\n".join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Player Profile Analysis - Scout Tool')
    parser.add_argument('--player', '-p', type=str, default=CONFIG['player_name'],
                        help='Player name (substring match)')
    parser.add_argument('--position', '-pos', type=str, default=CONFIG['position_filter'],
                        choices=['DF', 'MF', 'FW', 'GK'],
                        help='Position filter')
    parser.add_argument('--min-minutes', '-m', type=int, default=CONFIG['min_minutes'],
                        help='Minimum minutes played')
    parser.add_argument('--max-age', '-a', type=int, default=CONFIG['max_age'],
                        help='Maximum age for comparable search')
    parser.add_argument('--neighbors', '-n', type=int, default=CONFIG['n_neighbors'],
                        help='Number of comparable players to find')
    parser.add_argument('--top-k', '-k', type=int, default=CONFIG['top_k_anomalies'],
                        help='Top K anomaly metrics for radar/KNN')
    parser.add_argument('--threshold', '-t', type=int, default=CONFIG['percentile_threshold'],
                        help='Percentile threshold for anomalies')
    parser.add_argument('--csv', type=str, default=CONFIG['csv_path'],
                        help='Path to CSV data file')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plot images')
    # MLflow arguments
    parser.add_argument('--mlflow-uri', type=str, default=None,
                        help='MLflow tracking URI (e.g., http://localhost:5000 or file:./mlruns)')
    parser.add_argument('--mlflow-experiment', type=str, default='football-scout-analysis',
                        help='MLflow experiment name')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    return parser.parse_args()


def setup_mlflow(args):
    """Setup MLflow tracking if available and not disabled."""
    if not HAS_MLFLOW or args.no_mlflow:
        return False

    # Set tracking URI
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    else:
        # Default to SQLite database (MLflow 3.x requires DB backend)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Set or create experiment
    mlflow.set_experiment(args.mlflow_experiment)

    print(f"MLflow tracking: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {args.mlflow_experiment}")
    return True


def log_to_mlflow(results, config, args, scout_card, radar_path=None, bars_path=None, card_path=None):
    """Log parameters, metrics, and artifacts to MLflow."""
    if not HAS_MLFLOW or args.no_mlflow:
        return

    player_row = results['player_row']
    anomalies = results['anomalies']
    neighbors = results['neighbors']

    with mlflow.start_run(run_name=f"{config['player_name']}_{config['position_filter']}") as run:
        # Log parameters
        mlflow.log_params({
            'player_name': config['player_name'],
            'position_filter': config['position_filter'],
            'min_minutes': config['min_minutes'],
            'max_age': config['max_age'],
            'n_neighbors': config['n_neighbors'],
            'top_k_anomalies': config['top_k_anomalies'],
            'percentile_threshold': config['percentile_threshold'],
            'target_player': str(player_row['Player']),
            'target_club': str(player_row['Squad']),
            'target_age': float(player_row['Age']),
            'target_minutes': int(player_row['Min']),
            'target_nation': str(player_row['Nation']),
            'target_comp': str(player_row['Comp']),
        })

        # Log metrics
        mlflow.log_metrics({
            'n_position_peers': int(len(results['df'])),
            'n_anomalies_found': len(anomalies),
            'n_comparables_found': len(neighbors),
            'top_similarity': float(neighbors.iloc[0]['similarity']) if len(neighbors) > 0 else 0.0,
            'avg_top5_similarity': float(neighbors.head(5)['similarity'].mean()) if len(neighbors) >= 5 else 0.0,
        })

        # Log anomaly metrics as individual metrics
        for metric, data in anomalies.items():
            safe_metric = metric.replace(' ', '_').replace('/', '_').replace('+', 'plus').replace('-', '_')[:50]
            mlflow.log_metric(f'anomaly_{safe_metric}_percentile', data['percentile'])
            mlflow.log_metric(f'anomaly_{safe_metric}_value', data['value'])
            mlflow.log_metric(f'anomaly_{safe_metric}_rank', data['rank'])

        # Log top comparables as metrics
        for i, (_, row) in enumerate(neighbors.head(10).iterrows()):
            mlflow.log_metric(f'comp_{i+1}_similarity', float(row['similarity']))
            mlflow.log_metric(f'comp_{i+1}_age', float(row['Age']))
            mlflow.log_metric(f'comp_{i+1}_minutes', int(row['Min']))

        # Log artifacts
        if radar_path and os.path.exists(radar_path):
            mlflow.log_artifact(radar_path, "plots")
        if bars_path and os.path.exists(bars_path):
            mlflow.log_artifact(bars_path, "plots")
        if card_path and os.path.exists(card_path):
            mlflow.log_artifact(card_path, "reports")

        # Log scout card as text artifact
        if scout_card:
            card_temp = os.path.join(tempfile.gettempdir(), f"scout_card_{config['player_name'].replace(' ', '_')}.md")
            with open(card_temp, 'w', encoding='utf-8') as f:
                f.write(scout_card)
            mlflow.log_artifact(card_temp, "reports")
            try:
                os.remove(card_temp)
            except:
                pass

        # Log config as JSON artifact
        config_temp = os.path.join(tempfile.gettempdir(), f"config_{config['player_name'].replace(' ', '_')}.json")
        with open(config_temp, 'w') as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_temp, "config")
        try:
            os.remove(config_temp)
        except:
            pass

        # Log neighbors as CSV artifact
        if len(neighbors) > 0:
            neighbors_temp = os.path.join(tempfile.gettempdir(), f"neighbors_{config['player_name'].replace(' ', '_')}.csv")
            neighbors[['Player', 'Squad', 'Pos', 'Age', 'Min', 'similarity']].head(20).to_csv(neighbors_temp, index=False)
            mlflow.log_artifact(neighbors_temp, "data")
            try:
                os.remove(neighbors_temp)
            except:
                pass

        print(f"MLflow run logged: {run.info.run_id}")


def main():
    """Main entry point."""
    args = parse_args()

    # Build config from CLI args
    config = {
        'csv_path': args.csv,
        'player_name': args.player,
        'position_filter': args.position,
        'min_minutes': args.min_minutes,
        'max_age': args.max_age,
        'n_neighbors': args.neighbors,
        'top_k_anomalies': args.top_k,
        'percentile_threshold': args.threshold,
    }

    # Setup MLflow
    mlflow_enabled = setup_mlflow(args)

    try:
        # Run analysis
        results = run_analysis(config)

        # Generate visualizations
        radar_path = None
        bars_path = None
        if not args.no_plots:
            print("\nGenerating visualizations...")
            radar_fig = plot_radar_chart(results['df'], results['player_idx'], results['anomalies'], config)
            bars_fig = plot_percentile_bars(results['anomalies'], config)
            player_name_safe = config['player_name'].replace(' ', '_')
            radar_path = f"{player_name_safe}_radar.png"
            bars_path = f"{player_name_safe}_percentiles.png"

        # Generate and print scout card
        print("\n" + "="*60)
        print("SCOUT CARD")
        print("="*60)
        scout_card = generate_scout_card(
            results['player_row'],
            results['anomalies'],
            results['neighbors'],
            config
        )
        print(scout_card)

        # Save scout card to file
        player_safe = config['player_name'].replace(' ', '_')
        card_path = f"{player_safe}_scout_card.md"
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(scout_card)
        print(f"\nSaved scout card to {card_path}")

        # Log to MLflow
        if mlflow_enabled:
            log_to_mlflow(results, config, args, scout_card, radar_path, bars_path, card_path)

        return 0

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())