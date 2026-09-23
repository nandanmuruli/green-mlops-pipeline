import yaml
from datasets import disable_caching as _disable_hf_datasets_caching

# See src/pipeline/evaluate.py for why: HF `datasets` writes cache-*.arrow
# byproduct files into any dataset directory it transforms, which drifts
# the DVC-tracked hash of data/raw/imdb with no real change to the data.
_disable_hf_datasets_caching()

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
