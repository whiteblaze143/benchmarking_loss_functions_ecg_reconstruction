# Paper 15: Causal Mechanism Factorization of the Cardiac Cycle

## The Core Question
The heart goes through distinct physiological phases (e.g., depolarization transitioning to repolarization). Does the fundamental *mechanism* that causes this transition operate independently of the heart's current *state*?

## The Mathematical Framework: ICM
We build upon the **Independent Causal Mechanisms (ICM)** principle from causal inference.

## The Architecture
A sequence-to-sequence model that learns independent transition mechanisms $M_g : Z_g \rightarrow Z_{g+1}$. A severe MMD-based statistical penalty is applied to force the model to mathematically disentangle the transition operator $M$ (the rules of the heart) from the state vector $Z$ (the current voltage of the heart).

## The Biological Context
If the ICM principle holds true, it means a disease might alter the *state* of the patient's heart, but leave the fundamental biological transition *mechanisms* intact. This would allow for highly robust transfer learning between entirely different cardiac diseases, because the underlying physical "rules" of the heart remain exactly the same.

## Rigorous Falsification & Controls
- **Matched Control**: Entangled representation learning (a standard RNN).
- **Falsification (The Kill Test)**: We force mechanism entanglement by randomly mixing the basis before downstream queries. The model's ability to factorize the cycle must fail under forced entanglement.
