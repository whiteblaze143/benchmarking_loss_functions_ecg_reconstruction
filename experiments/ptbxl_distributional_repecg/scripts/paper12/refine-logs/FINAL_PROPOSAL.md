# Final Proposal: Paper 12

## Problem Anchor
Causal feature disentanglement

## Core Thesis
Phase-ordered conditional location-scale residuals may isolate diagnostically useful unpredictable variation beyond predictable phase state.

## Inductive Biases and Assumptions Embedded
The representation assumes that phase history predicts a conditional mean and scale, and that standardized residuals are approximately history-independent within the tested probe family. This is predictive residualization, not identified causal-mechanism disentanglement.

## Rejected Complexity
Conditional normalizing flows, invertibility claims, and causal-direction claims are removed until separately identified. The smallest adequate mechanism is a causal location-scale predictor with an explicit likelihood objective.

**Verdict: RETHINK.** The legacy BCE-only grid does not execute this thesis and must not run.
