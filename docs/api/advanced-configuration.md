# Advanced Configuration

This page is for API users who want to control the simulation setup more directly.

For most users the recommended starting point is:

```python
from simcbctgenerator import CBCTSystemConfig, Vendor

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
```

You can then pass that `system_config` into the pipelines and only use YAML or XML when you want to override parts of the default setup.

## Configuration Priority

The effective configuration order is:

1. `system_config`
2. vendor defaults from `CBCTSystemConfig.for_vendor(...)`
3. optional `metadata_yaml` overrides
4. optional `geometry_xml` override

That means:

- `system_config` is the main API path
- `metadata_yaml` is an optional metadata import path
- `geometry_xml` is an optional trajectory override path

## Recommended API Usage

### Vendor Defaults

Use vendor defaults when you want a stable, reusable scanner setup without reading any external metadata file.

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

### Packaged INI Presets

Use packaged INI presets when you want to reuse framework-defined settings.

```python
from simcbctgenerator import CBCTSystemConfig

available = CBCTSystemConfig.list_presets()
system_config = CBCTSystemConfig.from_preset(available[0])
```

### Custom INI Files

Use custom INI files when you want your own reusable framework configuration outside the code.

```python
from simcbctgenerator import CBCTSystemConfig

system_config = CBCTSystemConfig.from_ini("my_config.ini")
```

### Metadata YAML Overrides

Use `metadata_yaml` when you want to import patient- or dataset-specific scanner metadata such as exposure, detector sampling, or reconstruction volume settings.

```python
cbct = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    metadata_yaml="metadata.yaml",
    output_dir="output/reconstruction",
)
```

### Geometry XML Overrides

Use `geometry_xml` when you want to override the projection trajectory geometry.

The file must be an RTK geometry XML file, meaning the XML format used by the Reconstruction Toolkit (RTK):

- RTK homepage: <https://www.openrtk.org/>
- RTK documentation: <https://docs.openrtk.org/>

```python
cbct = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    geometry_xml="geometry.xml",
    output_dir="output/reconstruction",
)
```

You can also combine both:

```python
cbct = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    metadata_yaml="metadata.yaml",
    geometry_xml="geometry.xml",
    output_dir="output/reconstruction",
)
```

## INI Structure

`CBCTSystemConfig.from_ini(...)` expects these sections:

- `[PhysicsConfig]`
- `[GeometryConfig]`
- `[ReconstructionVolumeConfig]`

### Minimal INI Example

```ini
[PhysicsConfig]
spr = 1.6
mAs = 1.6
kv = 120
saturation_factor = 2.0
bp_amplitude = 1.07
bp_std = 522.0
threads = 8
max_block_index = 200

[GeometryConfig]
source_origin_distance = 1000.0
source_detector_distance = 1536.0
detector_offset = -115.0
detector_size_h = 410.0
detector_size_w = 410.0
detector_pixels_h = 1024
detector_pixels_w = 1024
start_angle = 0.0
end_angle = 360.0
angle_increments = 1.0

[ReconstructionVolumeConfig]
recon_size = 512,512,512
recon_origin = -255.0,-255.0,-255.0
recon_spacing = 1.0,1.0,1.0
```

### What INI Is For

INI is the framework-native configuration format. Use it when you want to define:

- scanner physics defaults
- detector geometry defaults
- reconstruction volume defaults
- reusable presets for API or CLI use

## YAML Structure

`metadata_yaml` is optional. It is mainly an import format for external datasets, not the primary API configuration format.

The current code reads these values from the YAML `cbct` section:

- `Manufacturer`
- `TubeVoltage`
- `TubeCurrent`
- `PulseLength`
- `ImagerSizeX`
- `ImagerSizeY`
- `ImagerResX`
- `ImagerResY`
- `DetectorOffsetX`
- `DetectorOffsetY`
- `ReconstructionSizeX`
- `ReconstructionSizeY`
- `ReconstructionSizeZ`
- `ReconstructionSpacingX`
- `ReconstructionSpacingY`
- `ReconstructionSpacingZ`
- `Frames`

### Minimal YAML Example

```yaml
cbct:
  Manufacturer: Elekta
  TubeVoltage: 120
  TubeCurrent: 40
  PulseLength: 40
  ImagerSizeX: 1024
  ImagerSizeY: 1024
  ImagerResX: 0.4
  ImagerResY: 0.4
  DetectorOffsetX: -115.0
  ReconstructionSizeX: 512
  ReconstructionSizeY: 512
  ReconstructionSizeZ: 512
  ReconstructionSpacingX: 1.0
  ReconstructionSpacingY: 1.0
  ReconstructionSpacingZ: 1.0
  Frames: 656
```

### What YAML Is For

YAML is useful when:

- you import scanner metadata from a challenge or dataset
- you want per-case detector sampling or exposure values
- you want to reuse externally provided metadata rather than define a framework preset

## Which Path To Choose

- Use `CBCTSystemConfig.for_vendor(...)` for the normal API path.
- Use `from_preset(...)` or `from_ini(...)` when you want reusable framework configs.
- Use `metadata_yaml` only when you want metadata import behavior.
- Use `geometry_xml` only when you want to override the actual acquisition geometry with an RTK geometry XML file.
