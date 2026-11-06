"""Snippets for docs/getting_started/8_ome_zarr_zip.md.

Each section between `--8<-- [start:name]` / `--8<-- [end:name]` markers is included
into the page by `pymdownx.snippets` and executed by `markdown-exec`. The whole file
is also runnable on its own:

    python docs/snippets/getting_started/ome_zarr_zip.py
"""

# --8<-- [start:create]
from pathlib import Path

import numpy as np

from ngio import create_ome_zarr_from_array

data_dir = Path("./data").absolute()
ozx_path = data_dir / "zip_demo.ozx"

rng = np.random.default_rng(0)
array = rng.integers(0, 2**12, size=(2, 4, 256, 256), dtype="uint16")
ome_zarr = create_ome_zarr_from_array(
    store=ozx_path,
    array=array,
    pixelsize=0.325,
    z_spacing=1.0,
    axes_names="czyx",
    channels_meta=["DAPI", "GFP"],
    levels=3,
    ngff_version="0.5",
    overwrite=True,
)
print(ome_zarr)
# --8<-- [end:create]

# --8<-- [start:edit]
from ngio import open_ome_zarr_container

ome_zarr = open_ome_zarr_container(ozx_path)
ome_zarr.set_channel_labels(["nuclei", "membrane"])
label = ome_zarr.derive_label("nuclei_mask")
label.set_array(np.zeros(label.shape, dtype="uint8"))
label.consolidate()
print(ome_zarr.list_labels())
# --8<-- [end:edit]

# --8<-- [start:check]
from ngio import check_ozx

print(check_ozx(ozx_path))
# --8<-- [end:check]

# --8<-- [start:session]
from ngio import OzxStore

session_path = data_dir / "zip_session.ozx"
with OzxStore(session_path, mode="w") as store:
    ome_zarr = create_ome_zarr_from_array(
        store, array, pixelsize=0.325, axes_names="czyx", ngff_version="0.5"
    )
    ome_zarr.derive_label("cells").set_array(np.zeros(array.shape[1:], "uint8"))
    print(store)
# --8<-- [end:session]

# --8<-- [start:repack]
size_before = session_path.stat().st_size
with OzxStore(session_path, mode="r+") as store:
    store.repack()
print(f"{size_before} B -> {session_path.stat().st_size} B")
# --8<-- [end:repack]

# --8<-- [start:transfer]
from ngio import from_memory, from_ozx, to_memory, to_ozx

in_memory = to_memory(ozx_path)
ome_zarr = open_ome_zarr_container(in_memory)
ome_zarr.derive_label("membrane_mask").set_array(np.ones(array.shape[1:], "uint8"))
from_memory(in_memory, ozx_path, overwrite=True)

from_ozx(ozx_path, data_dir / "zip_demo.zarr", overwrite=True)
to_ozx(data_dir / "zip_demo.zarr", data_dir / "zip_demo_copy.ozx", overwrite=True)
print(open_ome_zarr_container(data_dir / "zip_demo_copy.ozx").list_labels())
# --8<-- [end:transfer]
