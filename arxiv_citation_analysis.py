# Script to extract Hugging Face models with linked papers on arXiv and sort them by subject classification
# Outputted dataset on Hugging Face linked here: https://huggingface.co/datasets/modelbiome/publication_data

import os
import re
import time
import ast
import yaml
import json
import feedparser
import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
from collections import Counter, defaultdict
from pathlib import Path
from datasets import load_dataset
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio
from bertopic import BERTopic
from upsetplot import UpSet, from_memberships
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- Load DataFrame from Hugging Face Hub ---
print("Loading dataset from Hugging Face Hub (modelbiome/model_ecosystem_with_cards)...")
hf_dataset = load_dataset("modelbiome/model_ecosystem_with_cards", split="train")
df = pd.DataFrame(list(hf_dataset))
print(f"Loaded {len(df):,} rows from Hugging Face Hub")
print("Columns:", df.columns.tolist())

# --- Research Domain Mapping ---
def fetch_arxiv_subject_classifications():
    # Always use the static fallback mapping to avoid server/API errors
    return {
        'cs.AI': 'Computer Science, Artificial Intelligence',
        'cs.AR': 'Computer Science, Hardware Architecture',
        'cs.CL': 'Computer Science, Computation and Language',
        'cs.CV': 'Computer Science, Computer Vision and Pattern Recognition',
        'cs.LG': 'Computer Science, Machine Learning',
        'stat.ML': 'Statistics, Machine Learning',
        'eess.SP': 'Electrical Engineering and Systems Science, Signal Processing',
        'eess.IV': 'Electrical Engineering and Systems Science, Image and Video Processing',
        'math.OC': 'Mathematics, Optimization and Control',
        'physics.comp-ph': 'Physics, Computational Physics',
        # Add more mappings as needed
    }

# Language mapping for better analysis
LANGUAGE_MAPPING = {
    'en': 'English',
    'zh': 'Chinese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'pl': 'Polish',
    'tr': 'Turkish',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'fa': 'Persian',
    'ur': 'Urdu',
    'bn': 'Bengali',
    'te': 'Telugu',
    'ta': 'Tamil',
    'ml': 'Malayalam',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'my': 'Burmese',
    'km': 'Khmer',
    'lo': 'Lao',
    'ka': 'Georgian',
    'am': 'Amharic',
    'sw': 'Swahili',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'zu': 'Zulu',
    'af': 'Afrikaans',
    'xh': 'Xhosa',
    'st': 'Southern Sotho',
    'tn': 'Tswana',
    'ts': 'Tsonga',
    've': 'Venda',
    'nr': 'Southern Ndebele',
    'ss': 'Swati',
    'nd': 'Northern Ndebele',
    'unknown': 'Unknown',
    'und': 'Undetermined'
}

RESEARCH_DOMAIN_MAP = fetch_arxiv_subject_classifications()

# --- Helper: universal gradient bar plot ---
def gradient_bar_horizontal(output_path, title, labels, values, cmap_name="Blues", xlabel="Count"):
    wrapped_labels = []
    for lbl in labels:
        if len(lbl) > 50:
            parts = lbl.split(", ")
            if len(parts) > 1:
                wrapped_labels.append("\\n".join(parts))
            else:
                words = lbl.split()
                mid = len(words)//2
                wrapped_labels.append("\\n".join([" ".join(words[:mid]), " ".join(words[mid:])]))
        else:
            wrapped_labels.append(lbl)

    fig, ax = plt.subplots(figsize=(16, max(8, len(labels)*0.4)))
    
    # Use solid colors instead of gradients
    solid_colors = {
        "Blues": "#2E86AB",
        "Oranges": "#F18F01", 
        "Purples": "#A23B72",
        "Greens": "#6A994E",
        "Reds": "#C73E1D"
    }
    
    # Get the base color for this colormap
    base_color = solid_colors.get(cmap_name, "#2E86AB")
    colors = [base_color] * len(values)  # Same solid color for all bars

    bars = ax.barh(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.8, alpha=1.0)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(wrapped_labels, fontsize=10, fontweight="medium")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y()+bar.get_height()/2,
                f"{val:,}", va="center", ha="left",
                fontsize=9, fontweight="bold",
                color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def analyze_arxiv_citations_from_hf(output_dir=None):
    if output_dir is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/arxiv_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    merged = df.copy()

    print("Columns in DataFrame:", merged.columns.tolist())

    # --- Top Research Domains with full mapping ---
    if 'categories' in merged.columns:
        try:
            all_domains_raw = sum(
                [c if isinstance(c, list) else [c] for c in merged['categories'].dropna().tolist()],
                []
            )
            all_domains_full = [RESEARCH_DOMAIN_MAP.get(dom, dom) for dom in all_domains_raw]
            domain_counts = Counter(all_domains_full)
            top_domains = domain_counts.most_common(20)
            labels = [k for k,_ in top_domains]
            values = [v for _,v in top_domains]
            
            # Group by subject category and assign consistent colors
            subject_categories = {
                'Computer Science': ['Computer Science, Artificial Intelligence', 'Computer Science, Hardware Architecture', 
                                   'Computer Science, Computation and Language', 'Computer Science, Computer Vision and Pattern Recognition',
                                   'Computer Science, Machine Learning'],
                'Statistics': ['Statistics, Machine Learning'],
                'Electrical Engineering': ['Electrical Engineering and Systems Science, Signal Processing', 
                                        'Electrical Engineering and Systems Science, Image and Video Processing'],
                'Mathematics': ['Mathematics, Optimization and Control'],
                'Physics': ['Physics, Computational Physics']
            }
            
            # Create color mapping for each category
            category_colors = {
                'Computer Science': '#2E86AB',      # Blue
                'Statistics': '#A23B72',            # Purple
                'Electrical Engineering': '#F18F01', # Orange
                'Mathematics': '#C73E1D',           # Red
                'Physics': '#6A994E'                # Green
            }
            
            # Assign colors based on subject category
            colors = []
            for label in labels:
                assigned_color = '#95A5A6'  # Default gray for unknown categories
                if label is not None:  # Add null check
                    for category, color in category_colors.items():
                        if any(subject in label for subject in subject_categories[category]):
                            assigned_color = color
                            break
                colors.append(assigned_color)
            
            fig, ax = plt.subplots(figsize=(20, 14))
            y_pos = np.arange(len(values))
            bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5, alpha=1.0)

            # Set y-axis labels (not bold, readable font)
            ax.set_yticks(y_pos)
            full_labels = []
            for label in labels:
                if label in LANGUAGE_MAPPING:
                    full_labels.append(LANGUAGE_MAPPING[label])
                else:
                    full_labels.append(str(label))
            ax.set_yticklabels(full_labels, fontname='DejaVu Sans', fontsize=10)

            # --- Create legend for subject categories ---
            legend_lines = []
            legend_labels = []
            
            # Create legend entries for each category that appears in the data
            used_categories = set()
            for i, label in enumerate(labels):
                if label is not None:  # Add null check
                    for category, color in category_colors.items():
                        if any(subject in label for subject in subject_categories[category]):
                            if category not in used_categories:
                                line, = ax.plot([], [], color=color, linewidth=6)
                                line.set_label(category)
                                legend_lines.append(line)
                                legend_labels.append(category)
                                used_categories.add(category)
                            break
            
            # Add legend for unknown categories if any
            if any(color == '#95A5A6' for color in colors):
                line, = ax.plot([], [], color='#95A5A6', linewidth=6)
                line.set_label('Other')
                legend_lines.append(line)
                legend_labels.append('Other')

            legend = ax.legend(
                handles=legend_lines,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                fontsize=10,
                title="Subject Categories",
                title_fontsize=12,
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9
            )
            legend.get_title().set_color('#2c3e50')
            plt.subplots_adjust(right=0.75)

            ax.set_xlabel("Citation Count", fontsize=14, color='#2c3e50')
            ax.set_title("Top Cited Research Domains", fontsize=18, pad=25, color='#2c3e50')

            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(bar.get_width() + max(values) * 0.015, bar.get_y() + bar.get_height()/2, 
                       f"{value:,}", ha='left', va='center', fontsize=10, fontweight='normal', 
                       color='#2c3e50', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color('#bdc3c7')
                spine.set_linewidth(1.5)
            ax.invert_yaxis()
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_domains.png", dpi=300, bbox_inches='tight')
            plt.close()
            pd.DataFrame({"Domain_Name": labels, "Citation_Count": values}).to_csv(f"{output_dir}/top_domains_mapped.csv", index=False)
            print("[Top Domains] Done.")
        except Exception as e:
            print(f"[Top Domains] Skipped: {e}")
    else:
        print("[Top Domains] Skipped: 'categories' column not found.")

    # --- Top Authors ---
    if 'authors' in merged.columns:
        try:
            all_authors = sum([a if isinstance(a,list) else [a] for a in merged["authors"].dropna().tolist()], [])
            author_counts = Counter(all_authors)
            top_authors = author_counts.most_common(20)
            labels = [a for a,_ in top_authors]
            values = [c for _,c in top_authors]
            gradient_bar_horizontal(f"{output_dir}/top_authors.png", "Top Cited Authors", labels, values, cmap_name="Oranges", xlabel="Citation Count")
        except Exception as e:
            print(f"[Top Authors] Skipped: {e}")
    else:
        print("[Top Authors] Skipped: 'authors' column not found.")

    # --- Top Venues ---
    if 'ss_venue' in merged.columns:
        try:
            venues = [v for v in merged["ss_venue"].dropna().tolist() if "arxiv.org" not in str(v).lower()]
            venue_counts = Counter(venues)
            top_venues = venue_counts.most_common(20)
            labels = [v for v,_ in top_venues]
            values = [c for _,c in top_venues]
            gradient_bar_horizontal(f"{output_dir}/top_venues.png", "Top Venues", labels, values, cmap_name="Purples", xlabel="Citation Count")
        except Exception as e:
            print(f"[Top Venues] Skipped: {e}")
    else:
        print("[Top Venues] Skipped: 'ss_venue' column not found.")

    # --- Most Cited Papers ---
    if 'ss_citationCount' in merged.columns:
        try:
            merged["ss_citationCount"] = pd.to_numeric(merged["ss_citationCount"], errors="coerce")
            top_papers = merged.sort_values("ss_citationCount", ascending=False).head(20)
            labels = top_papers["title"].fillna("N/A").tolist()
            values = top_papers["ss_citationCount"].fillna(0).astype(int).tolist()
            gradient_bar_horizontal(f"{output_dir}/top_cited_papers.png", "Most Cited Papers", labels, values, cmap_name="Blues", xlabel="Citation Count")
        except Exception as e:
            print(f"[Top Papers] Skipped: {e}")
    else:
        print("[Top Papers] Skipped: 'ss_citationCount' column not found.")

    # --- Timeline ---
    if 'ss_year' in merged.columns:
        try:
            merged["ss_year"] = pd.to_numeric(merged["ss_year"], errors="coerce")
            year_counts = merged.groupby("ss_year")["arxiv_id"].count().sort_index()
            labels = [str(int(y)) for y in year_counts.index.tolist()]
            values = year_counts.values.tolist()
            gradient_bar_horizontal(f"{output_dir}/citation_timeline.png", "Cited Papers by Year", labels, values, cmap_name="Greens", xlabel="Number of Papers")
        except Exception as e:
            print(f"[Timeline] Skipped: {e}")
    else:
        print("[Timeline] Skipped: 'ss_year' column not found.")

    print(f"All main plots saved to {output_dir}")

if __name__ == "__main__":
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"outputs/arxiv_analysis_{timestamp}"
    analyze_arxiv_citations_from_hf(outdir)
