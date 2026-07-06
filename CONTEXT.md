# Learning Rate Matters LoRA Experiments

This glossary fixes project-specific language for LoRA experiment design and curvature-based adapter variants in this repository.

## Language

**Curvature anchor set**:
An external raw-text corpus used to estimate curvature statistics for CurvatureLoRA before supervised fine-tuning. It is independent of the math/code task training dataset.
_Avoid_: retain set, retain dataset, curvature dataset

**CurvatureLoRA**:
A LoRA training method whose adapter update is constrained by curvature-derived projection buffers during training. It is not an initialization-only LoRA method.
_Avoid_: CurvatureLora init, CurvatureLoRA initialization

**Base teacher**:
The frozen pretrained base model distribution used to sample Fisher targets for CurvatureLoRA. In PEFT training this is the active model with adapters disabled, not a separately loaded reference model.
_Avoid_: retain model, reference model copy, student teacher

**Folded CurvatureLoRA adapter**:
A saved adapter whose LoRA weights already include the curvature projection, so it behaves like a plain LoRA adapter during loading and merging.
_Avoid_: custom CurvatureLoRA checkpoint, projection-buffer adapter
