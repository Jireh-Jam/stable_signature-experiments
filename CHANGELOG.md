# Change Log

- common/__init__.py: New shared utilities package for logging, seeding, image I/O, and transforms registry.
- common/logging_utils.py: Added minimal logging setup helper.
- common/seeding.py: Added reproducible seeding utility.
- common/image_io.py: Added PIL-based image I/O helpers.
- common/transformations.py: Centralized transformations pipeline and registry.
- advanced_attacks/__init__.py: Expose `AdvancedWatermarkAttacks`.
- advanced_attacks/attack_class.py: Add type hints, structured logging, and docstrings; no breaking public flags.
- advanced_attacks/run.py: New orchestrator CLI for attacks.
- advanced_attacks/Readme.md: Overhauled README documenting architecture and usage.
- detector/__init__.py: Package init.
- detector/run.py: New orchestrator CLI for batch/single detection.
- detector/Readme.md: New README with pipeline and usage.
- docs/README_generate_watermarked_images.md: New focused README for the watermark generation script.
- docs/README_transformations_pipeline.md: New focused README for the transformations pipeline.
