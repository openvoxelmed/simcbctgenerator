# Pipelines

These are the main framework entry points. Most users only need one of these three.

## Reconstruction: `ProjectionPipeline`

Use this pipeline when your primary goal is to generate supervised reconstruction training data. It produces simulated projections from the planning CT and the corresponding reconstructed CBCT target under controlled acquisition and motion settings.

```python
from simcbctgenerator import CBCTSystemConfig, ProjectionPipeline, Vendor

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
pipeline = ProjectionPipeline(vendor=Vendor.ELEKTA, gpu=True)
cbct = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    output_dir="output/reconstruction",
)
```

CLI:

```bash
simcbct-pipeline-reconstruction --help
```

## Segmentation: `SegmentationPipeline`

Use this pipeline when you want a simulated CBCT and organ labels for segmentation training.

```python
from simcbctgenerator import CBCTSystemConfig, Vendor
from simcbctgenerator.api.segmentation import SegmentationPipeline

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
pipeline = SegmentationPipeline(
    method="standard",
    vendor=Vendor.ELEKTA,
    organ_list=["bowel", "bladder", "rectum"],
    priority=[1, 2, 3],
)
results = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    output_dir="output/segmentation",
)
```

CLI:

```bash
simcbct-pipeline-segmentation --help
```

## Regression: `RegressionPipeline`

Use this pipeline when you need aligned CT/CBCT pairs for regression or synthesis workflows.

```python
from simcbctgenerator import CBCTSystemConfig, Vendor
from simcbctgenerator.api.regression import RegressionPipeline

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
pipeline = RegressionPipeline(vendor=Vendor.ELEKTA, gpu=True)
results = pipeline.run(
    ct_image=ct_image,
    simulate_cbct=True,
    system_config=system_config,
    output_dir="output/regression",
)
```

CLI:

```bash
simcbct-pipeline-regression --help
```

## API Reference

For INI presets, custom INI files, metadata YAML, and geometry XML overrides, see [Advanced Configuration](advanced-configuration.md).

`geometry_xml` refers to an RTK geometry XML file from the Reconstruction Toolkit:
<https://www.openrtk.org/>

### Projection Pipeline

::: simcbctgenerator.api.reconstruction
    options:
      show_root_heading: true
      show_source: false

### Segmentation Pipeline

::: simcbctgenerator.api.segmentation
    options:
      show_root_heading: true
      show_source: false

### Regression Pipeline

::: simcbctgenerator.api.regression
    options:
      show_root_heading: true
      show_source: false
