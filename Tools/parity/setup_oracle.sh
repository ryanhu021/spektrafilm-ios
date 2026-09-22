#!/usr/bin/env bash
# Build the pinned Python oracle used to generate parity goldens.
#
# Clones upstream spektrafilm at the commit in upstream_pin.json and installs it into a
# Python 3.13 venv. Everything lands under Tools/parity/oracle/, which is gitignored:
# goldens are committed, the generator that produced them is reproducible on demand.
#
# Only the runtime dependencies are installed. Upstream's default install also pulls
# napari/PySide6 for its desktop GUI, which the engine never imports and which would add
# minutes to a cold setup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORACLE="$ROOT/Tools/parity/oracle"
PIN="$ROOT/Tools/parity/upstream_pin.json"

commit=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['upstream']['oracle_commit'])" "$PIN")
url=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['upstream']['url'])" "$PIN")

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

if [[ ! -d "$ORACLE/.git" ]]; then
  echo "==> cloning $url"
  git clone --quiet "$url" "$ORACLE"
fi

echo "==> checking out $commit"
git -C "$ORACLE" fetch --quiet origin "$commit" 2>/dev/null || git -C "$ORACLE" fetch --quiet origin
git -C "$ORACLE" checkout --quiet "$commit"

echo "==> creating Python 3.13 venv"
uv python install 3.13 >/dev/null
uv venv --quiet --python 3.13 "$ORACLE/.venv"

# The four libraries whose output lands in a committed fixture are pinned exactly; see
# oracle_environment in upstream_pin.json. The rest only need to satisfy upstream's imports.
pinned=$(python3 - "$PIN" <<'EOF'
import json, sys
packages = json.load(open(sys.argv[1]))["oracle_environment"]["packages"]
print(" ".join(f"{name}=={want}" for name, want in packages.items()))
EOF
)

echo "==> installing runtime dependencies"
echo "    pinned: $pinned"
# shellcheck disable=SC2086
VIRTUAL_ENV="$ORACLE/.venv" uv pip install --quiet $pinned \
  "scikit-image~=0.26" "opt-einsum~=3.4.0" "pyfftw~=0.15.0" "matplotlib~=3.10" \
  "rawpy~=0.26.1" "exiv2~=0.18.1" "lensfunpy~=1.18.0" "OpenImageIO~=3.1.11"
VIRTUAL_ENV="$ORACLE/.venv" uv pip install --quiet --no-deps -e "$ORACLE"

echo "==> verifying"
"$ORACLE/.venv/bin/python" - <<'PY'
import numpy as np
from spektrafilm import init_params, simulate

params = init_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura")
params.camera.auto_exposure = False
params.debug.lut_mode = True
out = simulate(np.full((4, 4, 3), 0.184), params)
print("18% gray ->", np.round(out[0, 0], 8))
PY

echo "==> oracle ready at Tools/parity/oracle"
