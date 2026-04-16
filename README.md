# Movie Character Analysis: Unveiling Storytelling Dynamics

## Business Problem
Understanding the underlying structure of a narrative—how characters interact, their importance to the plot, and their emotional evolution—is a complex task that usually requires manual literary analysis. For studios, writers, and analysts, there is a need to quantify these dynamics to ensure pacing, character consistency, and emotional impact.

## Methodology
This project implements a multi-dimensional pipeline to analyze character roles and dynamics using Natural Language Processing (NLP) and Graph Theory.

### 1. Data Pipeline
- **Parsing:** Raw screenplay text was parsed into a structured format, separating dialogue from narrative descriptions.
- **Normalization:** Text cleaning and normalization were applied to ensure consistency for downstream NLP tasks.
- **Aggregation:** Data was aggregated at the scene level (using `segment_id` as an operational unit) to capture co-occurrence and context.

### 2. Interaction Modeling (Graph Theory)
- **Network Construction:** A character interaction graph was built where nodes are characters and edges represent co-occurrence in a scene.
- **Centrality Metrics:** We computed **Degree Centrality**, **Weighted Degree**, and **Betweenness** to identify structural hubs and bridge characters.

### 3. Emotional Profiling (Deep Learning)
- **Emotion Detection:** A pre-trained DistilRoBERTa transformer model (`j-hartmann/emotion-english-distilroberta-base`) was used to classify dialogue into 7 emotions: anger, disgust, fear, sadness, neutral, surprise, and joy.
- **Temporal Analysis:** Emotional arcs were constructed by tracking the mean emotion intensity across the narrative progression.

### 4. Pattern Discovery (Unsupervised Learning)
- **Feature Engineering:** A unified feature matrix was created combining narrative importance, network metrics, and emotional profiles.
- **Clustering:** KMeans clustering (k=5) was applied to identify character archetypes based on multi-dimensional similarity.

## Results
- **Quantified Importance:** The "Importance Score" successfully isolated protagonists from secondary characters.
- **Archetype Discovery:** Unsupervised clustering identified 5 distinct roles:
    - **Core Drivers:** High centrality and narrative importance.
    - **Emotional Outliers:** Characters with high anger/intensity (e.g., authority figures).
    - **Supportive/Peripheral:** Lower centrality, stable emotional profiles.
- **Emotional Trajectories:** The analysis revealed distinct "emotional signatures" (e.g., Kat Stratford's transition from emotional control to vulnerability).

## Technical Decisions
- **Scene Definition:** Initially used `(segment, scene)`, but pivoted to `segment_id` only, as the original `scene_id` was too granular, which would have skewed the interaction network.
- **Smoothing:** Applied a rolling mean window to emotional arcs to remove noise and highlight narrative trends.
- **Scaling:** Used `StandardScaler` before KMeans to prevent magnitude-heavy features (like total words) from dominating emotional scores.

## Recommendations
- **Cross-Movie Analysis:** Scale the pipeline to compare archetypes across different genres (e.g., Horror vs. Comedy).
- **Dynamic Network Analysis:** Implement sliding-window graphs to see how the interaction network evolves in real-time.
- **Sentiment Correlation:** Correlate emotional peaks with specific plot points (climax, resolution) to analyze narrative pacing.
