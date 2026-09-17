# Paper 15: Causal Mechanism Factorization of the Cardiac Cycle

## The Core Question
The heart goes through distinct physiological phases (e.g., depolarization transitioning to repolarization). Does the *mechanism* that causes this transition operate independently of the heart's current *state*?

## The Math
The Independent Causal Mechanisms (ICM) principle.

## The Architecture
A sequence-to-sequence model that learns independent transition mechanisms $M_g : Z_g \rightarrow Z_{g+1}$. An MMD-based penalty is applied to force the model to mathematically disentangle the transition operator $M$ from the state vector $Z$.

## The Mechanism (Biological Context)
If this principle holds true, it means a disease might alter the *state* of the heart, but leave the fundamental biological transition *mechanisms* intact. This would allow for highly robust transfer learning between entirely different cardiac diseases, because the underlying "rules" of the heart remain the same.

## The Falsification & Control
- **Matched Control**: Entangled representation learning (standard RNN).
- **Falsification**: Force mechanism entanglement by randomly mixing the basis before downstream queries. The model's ability to factorize the cycle must fail.
