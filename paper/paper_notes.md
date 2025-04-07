# Notes for Paper Improvement

## Data and Results to Add

1. **Specific Dataset Information**
   - Add exact counts of genes and diseases in the dataset
   - Include statistics on the degree distribution
   - Mention specific disease categories most represented
   - Provide a reference to the original DG-Miner dataset paper

2. **Detailed Results**
   - Include confidence intervals for all performance metrics
   - Add a learning curve showing model performance vs. training set size
   - Include ROC and Precision-Recall curves for all models
   - Provide specific hyperparameter values used for each model

3. **Real Gene-Disease Predictions**
   - Replace the placeholder GENE1/DISEASE1 entries with actual top predictions
   - For each prediction, add a brief note on biological relevance
   - Check if any predictions are supported by recent literature not in training data

## Figures to Create

1. **Framework Overview (Figure 1)**
   - Create professional version based on framework_diagram.txt
   - Use consistent color scheme and styling

2. **Model Architecture Details (Figure 2)**
   - Detailed illustration of each GNN architecture
   - Show internal structure with dimensions and layer connections

3. **Performance Comparison (Figure 3)**
   - Bar chart comparing all models across multiple metrics
   - Include error bars for statistical significance

4. **Scalability Analysis (Figure 4)**
   - Plot showing training time vs. graph size
   - Plot showing memory usage vs. graph size
   - Include all models in the comparison

5. **Case Study Visualization (Figure 5)**
   - Network visualization of top predicted gene-disease associations
   - Highlight the subgraph structure that led to these predictions

## Writing Improvements

1. **Introduction**
   - Strengthen motivation with specific examples of successful GDA discoveries
   - Add more background on computational approaches in biomedicine
   - Clarify the novelty of our framework compared to existing approaches

2. **Related Work**
   - Expand the literature review to include more recent papers (2020-2023)
   - Organize related work into clearer categories
   - Add a brief review of biological databases used for GDA research

3. **Methodology**
   - Provide more details on the negative sampling strategy
   - Explain the rationale behind the chosen GNN architectures
   - Add pseudocode for the training algorithm

4. **Results**
   - Perform statistical significance tests between different models
   - Add more ablation studies on different components
   - Include results on transfer learning between different disease categories

5. **Discussion**
   - Add more insights on why certain models perform better
   - Discuss limitations of the current approach
   - Suggest specific biological applications for the framework

## Revisions Needed

1. **Abstract**
   - Make more concise, focus on key contributions
   - Add 1-2 specific findings with numbers

2. **Equations**
   - Check all equations for correctness and consistency in notation
   - Add explanations for all variables and symbols

3. **References**
   - Update placeholder references with actual papers
   - Add more recent work (2020-2023)
   - Ensure consistent formatting

4. **Overall Structure**
   - Consider adding a "Background" section before "Related Work"
   - Expand the "Discussion" section to better interpret results
   - Consider adding an "Ethics" section discussing implications

## Timeline for Completion

1. **First Draft Revisions** (1 week)
   - Update all placeholder content
   - Add real data and results
   - Complete initial figures

2. **Second Draft Revisions** (1 week)
   - Address writing improvements
   - Enhance figures with feedback
   - Complete all tables with detailed results

3. **Final Review** (3 days)
   - Check formatting and references
   - Ensure consistent terminology
   - Proofread for clarity and accuracy 