"""
scripts/run_clustering_analysis.py
Run the UMAP + HDBScan loop to find AI clusters.
"""
import pandas as pd
import numpy as np
# Note: Requires 'umap-learn' and 'hdbscan' installed
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from adv_hedging.data.loaders import load_wiki_data
from adv_hedging.constants import AI_MAKERS, AI_USERS, AI_CATEGORIES

def main():
    print("Loading data...")
    df = load_wiki_data()
    
    # Filter for AI companies
    ai_universe = AI_MAKERS + AI_USERS
    df_ai = df[df['ticker'].isin(ai_universe)].copy()
    
    # Map ground truth labels
    df_ai['expert_label'] = df_ai['ticker'].map(AI_CATEGORIES)
    ground_truth = df_ai['expert_label'].values
    
    print(f"Analyzing {len(df_ai)} AI companies...")
    
    # Parse embeddings (assuming they are stored as arrays in parquet)
    # If they are strings/lists, ensure they are converted to numpy matrix
    embeddings = np.stack(df_ai['embedding_mpnet'].values)
    
    best_score = -1
    best_params = {}
    
    # Hyperparameter Tuning Loop
    neighbors_list = [5, 10, 15]
    min_dist_list = [0.0, 0.1]
    
    for n_neighbors in neighbors_list:
        for min_dist in min_dist_list:
            # 1. Run UMAP
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
            
            # 2. Run HDBScan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
            labels = clusterer.fit_predict(embedding_2d)
            
            # 3. Score (Ignoring Noise -1)
            mask = labels != -1
            if mask.sum() > 5: # Need enough points to score
                ari = adjusted_rand_score(ground_truth[mask], labels[mask])
                
                if ari > best_score:
                    best_score = ari
                    best_params = {'n': n_neighbors, 'd': min_dist}
                    print(f"New Best ARI: {ari:.3f} (Neighbors={n_neighbors}, Dist={min_dist})")

    print(f"\nOptimization Complete. Best Params: {best_params}")

if __name__ == "__main__":
    main()