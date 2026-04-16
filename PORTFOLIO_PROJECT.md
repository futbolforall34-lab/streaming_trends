# Portfolio Showcase: Movie Character Analysis 🎬

## Executive Summary
Can we quantify the "soul" of a story? This project transforms raw screenplays into a multi-dimensional map of character behavior. By merging **Deep Learning (Transformers)**, **Graph Theory**, and **Unsupervised Learning**, I developed a framework that identifies character archetypes and maps emotional journeys across a movie's narrative.

## The Challenge
Screenplays are semi-structured text. The challenge was to move from "lines of text" to "meaningful roles," bridging the gap between raw dialogue and high-level narrative functions.

## My Approach
I built a three-layered analytical engine:
1. **Structural Layer:** Built interaction networks to see who "controls" the story.
2. **Semantic Layer:** Used a DistilRoBERTa transformer to extract a spectrum of 7 emotions from every line of dialogue.
3. **Pattern Layer:** Applied KMeans clustering to let the data define "archetypes" (e.g., the "Emotional Catalyst" vs. the "Narrative Anchor").

## Key Skills Demonstrated
- **NLP & Deep Learning:** Implementation of HuggingFace Transformers for sequence classification and emotion detection.
- **Network Analysis:** Using `NetworkX` to calculate centrality and model social dynamics.
- **Machine Learning:** Feature engineering and unsupervised clustering (KMeans) with PCA visualization.
- **Data Engineering:** Building a robust pipeline from raw `.txt` files to processed `.parquet` datasets.

## Impactful Results
- **Archetype Discovery:** Automatically grouped characters into 5 distinct roles based on behavior and importance.
- **Emotional Arcs:** Mapped the psychological evolution of characters, revealing shifts from stability to vulnerability.
- **Narrative Validation:** Proved that structural centrality (who talks to whom) and narrative importance (who speaks the most) are related but distinct dimensions of a character's role.

## Next Steps
- **Genre Benchmarking:** Comparing "Hero" archetypes across Action vs. Drama.
- **Real-time Dashboard:** Creating an interactive tool for writers to visualize the emotional balance of their scripts.
- **Plot Point Detection:** Using emotion spikes to automatically detect the "climax" of a story.
