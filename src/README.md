# README

## Structure

```
src/
├── data.py                    # Data loading and preprocessing
├── we_cs.py                   # Word embedding and analogy experiments
├── README.md                  # Project documentation (this file)

scripts/
├── train.py                   # Script to launch training
├── evaluate.py                # Script to run evaluation
├── preprocess.py              # Data preprocessing script

configs/
├── default.yaml               # Default experiment configuration
├── experiment1.yaml           # Example experiment configuration

tests/
├── test_model.py              # Unit tests for model components
├── test_data.py               # Unit tests for data pipeline
├── test_utils.py              # Unit tests for utility functions

logs/                          # Training logs and outputs

pyproject.toml                 # Project configuration and dependencies
uv.lock                        # Locked dependency versions
LICENSE                        # Project license
```
