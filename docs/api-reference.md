# API Overview

The public API is centered on a small set of user-facing classes.

## Main Entry Points

### `ProjectionPipeline`

Use this for supervised reconstruction training data generation:

```text
CT -> motion -> projections -> reconstructed CBCT target
```

### `SegmentationPipeline`

Use this when you need simulated CBCTs together with segmentation labels.

### `RegressionPipeline`

Use this when you need aligned CT/CBCT pairs for regression workflows.

### `Patient`

Use `Patient` when you want to prepare image-space patient data directly before calling the simulation components.

## Practical Order

For most users the best order is:

1. Start with the example for the workflow you need.
2. Move to the matching pipeline class.
3. Only look at `Patient` if you need custom image loading or preloaded-image setup.

Continue with:

- [Pipelines](api/pipelines.md)
- [Advanced Configuration](api/advanced-configuration.md)
- [Patient](api/patient.md)
