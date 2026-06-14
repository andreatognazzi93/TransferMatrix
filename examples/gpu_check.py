from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multilayer_tmm import print_jax_devices


print_jax_devices()
