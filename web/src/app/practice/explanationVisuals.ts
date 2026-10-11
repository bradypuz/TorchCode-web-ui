import type { VISUAL_KINDS } from './ExplanationVisual';

type VisualKind = typeof VISUAL_KINDS[number];

// Supplemental examples belong to the UI, independently of the model's answer.
export const EXPLANATION_VISUALS: Readonly<Partial<Record<string, VisualKind>>> = {
  layernorm: 'normalization',
  batchnorm: 'normalization',
  rmsnorm: 'normalization',
  attention: 'attention',
  causal_attention: 'attention-mask',
  sliding_window: 'attention-mask',
  dropout: 'dropout',
  embedding: 'embedding',
  rope: 'rope',
  topk_sampling: 'sampling',
  ppo_loss: 'ppo',
};
