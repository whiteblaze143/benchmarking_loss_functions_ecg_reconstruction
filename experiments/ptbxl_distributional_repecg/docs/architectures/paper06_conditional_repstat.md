# Paper 06: Conditional RepStat

## The Core Question
Does the later part of the heartbeat (the T-wave) contain genuinely *new* biological information, or could a doctor have diagnosed the patient entirely just by looking at the earlier part (the QRS complex)?

## The Mathematical Framework: Conditional MMD
We upgrade our statistics to use **Conditional Maximum Mean Discrepancy (Conditional MMD)**. 
Instead of measuring the raw distribution of the T-wave, we measure the distribution of the T-wave *given* prior information (the QRS complex). 

## The Architecture
The neural network computes conditional KMEs. Mathematically, it projects the current phase state (e.g., the T-wave representation) so that it is strictly orthogonal to prior states $Z_{<g}$. This can also be used to condition out clinical metadata, ensuring the representation of the ECG is completely orthogonal to the patient's Age and Sex.

## The Biological Context
This is a rigorous mathematical test of conditional dependence. It prevents deep learning models from "double-counting" information. If the shape of the QRS complex perfectly and deterministically predicts the shape of the T-wave, the model is forced (via orthogonality) to recognize that the T-wave provides absolutely zero *new* conditional information for the diagnosis.

## Rigorous Falsification & Controls
- **Matched Control**: Marginal MMD (Paper 01) which does not condition out prior information.
- **Falsification (The Kill Test)**: We inject highly correlated, redundant synthetic signals into the dataset. The conditional model must completely ignore the redundant signal (because it contains no *new* information), while the standard marginal model will over-index on it and fail the test.
