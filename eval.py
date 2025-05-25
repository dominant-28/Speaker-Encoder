import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from speakerendcoder import SpeakerEncoder
import torch.nn.functional 
import torch.nn as nn
import matplotlib.pyplot as plt
import random



def evaluate_embeddings(model, test_loader, device, id_to_speaker):
    model.eval()
    all_embeddings = []
    all_speakers = []
    all_texts = []
    
    # Extract embeddings for test samples
    with torch.no_grad():
        for specs, labels, texts in tqdm(test_loader, desc="Extracting embeddings"):
            specs = specs.to(device)
            embeddings = model.extract_embedding(specs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_speakers.extend([id_to_speaker[label.item()] for label in labels])
            all_texts.extend(texts)
    
    all_embeddings = np.vstack(all_embeddings)
    
    # 1. Visualize embeddings with t-SNE
    visualize_embeddings(all_embeddings, all_speakers)
    
    # 2. Calculate intra/inter speaker similarities
    calculate_similarity_metrics(all_embeddings, all_speakers)
    
    # 3. Speaker verification experiment
    speaker_verification_test(all_embeddings, all_speakers)

def visualize_embeddings(embeddings, speakers, n_samples=1000):
    """Visualize speaker embeddings using t-SNE"""
    # Limit to n_samples for visualization
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        sample_speakers = [speakers[i] for i in indices]
    else:
        sample_embeddings = embeddings
        sample_speakers = speakers
    
    # Apply t-SNE for dimensionality reduction
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sample_embeddings)
    
    # Convert speaker IDs to labels for plotting
    unique_speakers = sorted(set(sample_speakers))
    speaker_to_color = {speaker: i for i, speaker in enumerate(unique_speakers)}
    colors = [speaker_to_color[speaker] for speaker in sample_speakers]
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6, cmap='viridis')
    
    # Add legend with a subset of speakers
    if len(unique_speakers) > 10:
        legend_speakers = unique_speakers[:10]  # Show only first 10 speakers
    else:
        legend_speakers = unique_speakers
    
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=plt.cm.viridis(speaker_to_color[s]/len(unique_speakers)), 
                     markersize=10, label=f"Speaker {s}") 
                     for s in legend_speakers]
    
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title('t-SNE Visualization of Speaker Embeddings')
    plt.savefig('speaker_embeddings_tsne.png')
    plt.close()

def calculate_similarity_metrics(embeddings, speakers):
    """Calculate intra-speaker and inter-speaker similarity metrics"""
    unique_speakers = sorted(set(speakers))
    
    # Group embeddings by speaker
    speaker_embeddings = {s: [] for s in unique_speakers}
    for emb, spk in zip(embeddings, speakers):
        speaker_embeddings[spk].append(emb)
    
    # Calculate average intra-speaker similarity
    intra_similarities = []
    for speaker, embs in speaker_embeddings.items():
        if len(embs) > 1:  # Need at least 2 samples for intra-speaker comparison
            embs_array = np.array(embs)
            sim_matrix = cosine_similarity(embs_array)
            # Get upper triangle values (excluding diagonal)
            upper_triangle = sim_matrix[np.triu_indices(len(embs), k=1)]
            if len(upper_triangle) > 0:
                intra_similarities.extend(upper_triangle)
    
    # Calculate inter-speaker similarity
    inter_similarities = []
    for i, speaker1 in enumerate(unique_speakers[:-1]):
        for speaker2 in unique_speakers[i+1:]:
            embs1 = np.array(speaker_embeddings[speaker1])
            embs2 = np.array(speaker_embeddings[speaker2])
            if len(embs1) > 0 and len(embs2) > 0:
                sim_matrix = cosine_similarity(embs1, embs2)
                inter_similarities.extend(sim_matrix.flatten())
    
    # Print statistics
    print("\nSimilarity Metrics:")
    print(f"Average intra-speaker similarity: {np.mean(intra_similarities):.4f} ± {np.std(intra_similarities):.4f}")
    print(f"Average inter-speaker similarity: {np.mean(inter_similarities):.4f} ± {np.std(inter_similarities):.4f}")
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(intra_similarities, bins=30, alpha=0.5, label='Intra-speaker', density=True)
    plt.hist(inter_similarities, bins=30, alpha=0.5, label='Inter-speaker', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Intra-speaker and Inter-speaker Similarities')
    plt.legend()
    plt.savefig('similarity_distributions.png')
    plt.close()

def speaker_verification_test(embeddings, speakers, num_trials=1000):
    """Run a speaker verification test with equal error rate (EER) calculation"""
    unique_speakers = sorted(set(speakers))
    if len(unique_speakers) < 2:
        print("Need at least 2 speakers for verification test")
        return
    
    # Group embeddings by speaker
    speaker_embeddings = {s: [] for s in unique_speakers}
    for emb, spk in zip(embeddings, speakers):
        speaker_embeddings[spk].append(emb)
    
    # Filter out speakers with less than 2 samples
    speaker_embeddings = {s: embs for s, embs in speaker_embeddings.items() if len(embs) >= 2}
    
    if len(speaker_embeddings) < 2:
        print("Need at least 2 speakers with multiple samples")
        return
    
    # Prepare trials: half same-speaker, half different-speaker
    same_speaker_scores = []
    different_speaker_scores = []
    
    # Same speaker trials
    for _ in range(num_trials // 2):
        speaker = random.choice(list(speaker_embeddings.keys()))
        if len(speaker_embeddings[speaker]) < 2:
            continue
        emb1, emb2 = random.sample(speaker_embeddings[speaker], 2)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        same_speaker_scores.append(similarity)
    
    # Different speaker trials
    for _ in range(num_trials // 2):
        speaker1, speaker2 = random.sample(list(speaker_embeddings.keys()), 2)
        emb1 = random.choice(speaker_embeddings[speaker1])
        emb2 = random.choice(speaker_embeddings[speaker2])
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        different_speaker_scores.append(similarity)
    
    # Calculate Equal Error Rate (EER)
    all_scores = np.concatenate([same_speaker_scores, different_speaker_scores])
    all_labels = np.concatenate([np.ones(len(same_speaker_scores)), np.zeros(len(different_speaker_scores))])
    
    # Try different thresholds
    thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 100)
    min_difference = float('inf')
    eer = 0
    best_threshold = 0
    
    for threshold in thresholds:
        # False accept rate: different speakers classified as same
        far = np.sum((np.array(different_speaker_scores) >= threshold)) / len(different_speaker_scores)
        
        # False reject rate: same speaker classified as different
        frr = np.sum((np.array(same_speaker_scores) < threshold)) / len(same_speaker_scores)
        
        difference = abs(far - frr)
        if difference < min_difference:
            min_difference = difference
            eer = (far + frr) / 2
            best_threshold = threshold
    
    print(f"\nSpeaker Verification Results:")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")
    
    # Plot ROC curve
    fprs, tprs, _ = calculate_roc(all_scores, all_labels)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fprs, tprs)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Speaker Verification')
    plt.savefig('speaker_verification_roc.png')
    plt.close()

def calculate_roc(scores, labels):
    thresholds = np.linspace(np.min(scores), np.max(scores), 100)
    fprs = []
    tprs = []
    
    for threshold in thresholds:
        predictions = scores >= threshold
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    return fprs, tprs, thresholds