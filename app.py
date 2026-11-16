"""
NE111 Term Project - Plot Digitizer (Streamlit web app)

Extracts x-y data from an image of a graph.

Workflow
  1. Upload an image (png / jpg / gif / tiff / bmp, colour or greyscale).
  2. Calibrate: click the axis origin, a point on the x-axis and a point on the
     y-axis, then type in the data values at those points.  Because the mapping
     is built from the two axis *vectors* (not from the image's pixel rows and
     columns), rotated or skewed scans are handled automatically.
  3. Digitize: click on data points.  Several named datasets are supported,
     each drawn in its own colour.
  4. Edit the live table (the marks on the image follow the edits), then
     download the CSV.

Points are stored in ORIGINAL-image pixel coordinates, so changing the
calibration, axis type (linear/log) or zoom level never loses any clicks; the
data values are always recomputed from the pixels.
"""

import csv
import hashlib
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# One distinct colour per dataset (cycled if there are more datasets).
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#00a3a3", "#f032e6", "#9a6324", "#000075", "#808000",
]
# Calibration markers: name -> (label, colour)
CALIB = {
    "O": ("Origin", "#ff0000"),
    "X": ("X-axis point", "#0066ff"),
    "Y": ("Y-axis point", "#00aa00"),
}
MAX_VIEW_W, MAX_VIEW_H = 1000, 800  # size of the clickable picture in pixels
ZOOM_LEVELS = [1, 1.5, 2, 3, 4, 6, 8, 12]
MODES = ["Calibrate axes", "Digitize points", "Pan (click to re-centre zoom)"]

ss = st.session_state


# --------------------------------------------------------------------------
# Session-state helpers
# --------------------------------------------------------------------------
def new_dataset():
    """Create a dataset with a unique id, default name and the next colour."""
    i = ss.ds_counter
    ss.ds_counter += 1
    return {"id": i, "name": f"Dataset {i + 1}", "color": PALETTE[i % len(PALETTE)], "points": []}


def reset_digitizing_state():
    """Clear calibration and datasets (called when a new image is loaded)."""
    ss.calib = {"O": None, "X": None, "Y": None}
    ss.ds_counter = 0
    ss.datasets = [new_dataset()]
    ss.editor_ver = 0
    set_pending(active_id=ss.datasets[0]["id"], mode=MODES[0], calib_target="O",
                zoom=1, pan_x=50, pan_y=50)


def set_pending(**kwargs):
    """Queue widget-state changes.

    Streamlit forbids changing a widget's value after that widget has been drawn
    in the current run, so changes are queued here and applied at the top of the
    next run (before any widget exists).
    """
    ss.setdefault("_pending", {}).update(kwargs)


def apply_pending():
    for k, v in ss.pop("_pending", {}).items():
        ss[k] = v


def points_changed():
    """Force the data-editor table to rebuild from the current points."""
    ss.editor_ver += 1


def active_dataset():
    for ds in ss.datasets:
        if ds["id"] == ss.get("active_id"):
            return ds
    ss.active_id = ss.datasets[0]["id"]
    return ss.datasets[0]


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_image(data: bytes) -> Image.Image:
    """Open any common image and normalise it to 8-bit RGB.

    Handles greyscale (L), 16-bit / float greyscale TIFFs, palette images,
    transparency (flattened onto white), animated GIFs (first frame) and EXIF
    orientation tags.
    """
    img = Image.open(io.BytesIO(data))
    img.seek(0)  # first frame of an animated GIF / multi-page TIFF
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), "L")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


@st.cache_data(show_spinner=False)
def make_sample_image() -> bytes:
    """Build a demo graph (log-y axis, two curves, scanned ~4 degrees crooked)."""
    W, H = 900, 650
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x0p, y0p, x1p, y1p = 110, 560, 840, 60  # plot area corners (pixels)

    def to_px(x, y):  # x in [0, 10] linear, y in [1, 1000] log
        return (x0p + x / 10 * (x1p - x0p), y0p - math.log10(y) / 3 * (y0p - y1p))

    font = _font(18)
    for i in range(11):  # x grid + tick labels
        px, _ = to_px(i, 1)
        d.line([(px, y1p), (px, y0p)], fill="#dddddd")
        d.text((px, y0p + 18), str(i), fill="black", font=font, anchor="mm")
    for dec in range(4):  # y decades, with minor log grid lines
        for m in range(1, 10):
            if dec == 3 and m > 1:
                break
            _, py = to_px(0, m * 10**dec)
            d.line([(x0p, py), (x1p, py)], fill="#bbbbbb" if m == 1 else "#eeeeee")
        _, py = to_px(0, 10**dec)
        d.text((x0p - 12, py), f"{10**dec}", fill="black", font=font, anchor="rm")
    d.line([(x0p, y0p), (x1p, y0p)], fill="black", width=2)
    d.line([(x0p, y0p), (x0p, y1p)], fill="black", width=2)
    d.text(((x0p + x1p) / 2, y0p + 50), "Time (s)", fill="black", font=font, anchor="mm")
    d.text((20, 40), "Count", fill="black", font=font)

    xs = np.linspace(0, 10, 11)
    for ys, col in ((2 * 2 ** (xs * 0.8), "#c00000"), (500 * np.exp(-xs / 3) + 1, "#0040c0")):
        pts = [to_px(x, y) for x, y in zip(xs, ys)]
        d.line(pts, fill=col, width=2)
        for px, py in pts:
            d.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
    img = img.rotate(4, resample=Image.BICUBIC, expand=True, fillcolor="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# Calibration maths
# --------------------------------------------------------------------------
def build_calibration():
    """Return (calibration dict, None) or (None, reason it is not ready).

    With origin O, x-axis point X and y-axis point Y (pixels), any pixel P is
    written as  P = O + a*(X-O) + b*(Y-O).  Then
        x = x0 + a*(x1-x0)   and   y = y0 + b*(y1-y0)
    where, for a log axis, the values are replaced by their log10.  Solving
    for (a, b) uses the 2x2 matrix [X-O, Y-O], which automatically accounts for
    image rotation (and even shear).
    """
    c = ss.calib
    missing = [CALIB[k][0] for k in "OXY" if c[k] is None]
    if missing:
        return None, "Still to place: " + ", ".join(missing) + "."
    x0, x1, y0, y1 = ss.x0, ss.x1, ss.y0, ss.y1
    if ss.xlog and (x0 <= 0 or x1 <= 0):
        return None, "Log x-axis values must be > 0."
    if ss.ylog and (y0 <= 0 or y1 <= 0):
        return None, "Log y-axis values must be > 0."
    if x0 == x1:
        return None, "Origin x-value and X-axis point value must differ."
    if y0 == y1:
        return None, "Origin y-value and Y-axis point value must differ."
    O = np.array(c["O"], float)
    u, v = np.array(c["X"], float) - O, np.array(c["Y"], float) - O
    M = np.column_stack([u, v])
    if abs(np.linalg.det(M)) < 1e-3 * np.linalg.norm(u) * np.linalg.norm(v):
        return None, "The three calibration points are (nearly) in a straight line."
    f = np.log10
    return {
        "O": O, "M": M, "Minv": np.linalg.inv(M),
        "xlog": ss.xlog, "ylog": ss.ylog,
        "xa": f(x0) if ss.xlog else x0, "xb": f(x1) if ss.xlog else x1,
        "ya": f(y0) if ss.ylog else y0, "yb": f(y1) if ss.ylog else y1,
    }, None


def px_to_data(cal, pts):
    """Pixel coordinates (N x 2) -> data values (N x 2)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    ab = (pts - cal["O"]) @ cal["Minv"].T
    x = cal["xa"] + ab[:, 0] * (cal["xb"] - cal["xa"])
    y = cal["ya"] + ab[:, 1] * (cal["yb"] - cal["ya"])
    if cal["xlog"]:
        x = 10**x
    if cal["ylog"]:
        y = 10**y
    return np.column_stack([x, y])


def data_to_px(cal, vals):
    """Data values (N x 2) -> pixel coordinates (N x 2).  Inverse of px_to_data."""
    vals = np.asarray(vals, float).reshape(-1, 2)
    x, y = vals[:, 0], vals[:, 1]
    if cal["xlog"]:
        x = np.log10(x)
    if cal["ylog"]:
        y = np.log10(y)
    a = (x - cal["xa"]) / (cal["xb"] - cal["xa"])
    b = (y - cal["ya"]) / (cal["yb"] - cal["ya"])
    return cal["O"] + np.column_stack([a, b]) @ cal["M"].T


# --------------------------------------------------------------------------
# Drawing the clickable view
# --------------------------------------------------------------------------
def view_window(img):
    """Crop box of the original image for the current zoom/pan, and the scale
    factor from original-image pixels to displayed pixels."""
    W, H = img.size
    z = ss.zoom
    cw, ch = W / z, H / z
    left = min(max(ss.pan_x / 100 * W - cw / 2, 0), W - cw)
    top = min(max(ss.pan_y / 100 * H - ch / 2, 0), H - ch)
    scale = min(MAX_VIEW_W / cw, MAX_VIEW_H / ch)
    return (left, top, cw, ch), scale


def render_view(img, box, scale, cal):
    left, top, cw, ch = box
    out_w, out_h = max(1, round(cw * scale)), max(1, round(ch * scale))
    view = img.resize((out_w, out_h), Image.LANCZOS, box=(left, top, left + cw, top + ch))
    d = ImageDraw.Draw(view)
    font = _font(14)

    def tv(p):  # original pixel -> view pixel
        return ((p[0] - left) * scale, (p[1] - top) * scale)

    # Axis guide lines through the origin (show rotation & help check calibration)
    c = ss.calib
    if c["O"] is not None:
        O = np.array(c["O"], float)
        reach = 3 * max(img.size)
        for k in "XY":
            if c[k] is not None:
                vec = np.array(c[k], float) - O
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n * reach
                    d.line([tv(O - vec), tv(O + vec)], fill=CALIB[k][1], width=1)

    # Calibration crosshairs
    for k, (label, col) in CALIB.items():
        if c[k] is not None:
            x, y = tv(c[k])
            r = 10
            d.line([(x - r, y), (x + r, y)], fill=col, width=2)
            d.line([(x, y - r), (x, y + r)], fill=col, width=2)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=col, width=2)
            d.text((x + 8, y - 20), k, fill=col, font=font, stroke_width=2, stroke_fill="white")

    # Data points: same colour within a dataset, active dataset drawn larger
    act = ss.get("active_id")
    for ds in ss.datasets:
        r = 6 if ds["id"] == act else 4
        for p in ds["points"]:
            x, y = tv(p)
            if -r <= x <= out_w + r and -r <= y <= out_h + r:
                d.ellipse([x - r, y - r, x + r, y + r], fill=ds["color"], outline="white", width=1)
    return view


def handle_click(px, py, img):
    """Act on a click at original-image pixel (px, py) according to the mode."""
    mode = ss.mode
    if mode == MODES[0]:  # calibrate
        target = ss.calib_target
        ss.calib[target] = (px, py)
        remaining = [k for k in "OXY" if ss.calib[k] is None]
        if remaining:
            set_pending(calib_target=remaining[0])
        elif build_calibration()[0] is not None:  # all placed -> move on to digitizing
            set_pending(mode=MODES[1])
        points_changed()
    elif mode == MODES[1]:  # digitize
        active_dataset()["points"].append([px, py])
        points_changed()
    else:  # pan: re-centre the zoom window on the click
        W, H = img.size
        set_pending(pan_x=round(100 * px / W, 1), pan_y=round(100 * py / H, 1))


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def fmt(v, sig):
    return f"{v:.{sig}g}"


def build_csv_rows(cal, sig):
    """Rows: [dataset names], [axis labels], then x, y values side by side."""
    xl, yl = ss.xlabel or "x", ss.ylabel or "y"
    names, labels, cols = [], [], []
    for ds in ss.datasets:
        names += [ds["name"], ""]
        labels += [xl, yl]
        cols.append(px_to_data(cal, ds["points"]) if ds["points"] else np.empty((0, 2)))
    n = max((len(c) for c in cols), default=0)
    rows = [names, labels]
    for i in range(n):
        row = []
        for c in cols:
            row += [fmt(c[i, 0], sig), fmt(c[i, 1], sig)] if i < len(c) else ["", ""]
        rows.append(row)
    return rows


def rows_to_csv(rows):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


# ==========================================================================
# APP
# ==========================================================================
ss.setdefault("ds_counter", 0)
if "datasets" not in ss:
    reset_digitizing_state()
ss.setdefault("img_hash", None)
ss.setdefault("last_click", None)
apply_pending()

st.title("📈 Plot Digitizer")
st.caption("Extract numbers from a picture of a graph: upload → calibrate the axes → click the data → download CSV.")

# ---------------- Sidebar: image, axes, calibration ----------------
with st.sidebar:
    st.header("1 · Image")
    up = st.file_uploader(
        "Upload a graph image",
        type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "webp"],
    )
    use_sample = st.checkbox("Use built-in sample graph (rotated, semi-log)", value=False,
                             disabled=up is not None)
    data = up.getvalue() if up is not None else (make_sample_image() if use_sample else None)

    st.header("2 · Axes")
    st.text_input("X-axis label", value="x", key="xlabel")
    st.text_input("Y-axis label", value="y", key="ylabel")
    c1, c2 = st.columns(2)
    c1.checkbox("Log x-axis", key="xlog")
    c2.checkbox("Log y-axis", key="ylog")

    st.header("3 · Calibration values")
    st.caption("Data values at the three reference points you click on the image. "
               "The origin is where the axes cross; it need not be (0, 0).")
    c1, c2 = st.columns(2)
    c1.number_input("Origin x", value=0.0, key="x0", format="%g")
    c2.number_input("Origin y", value=0.0, key="y0", format="%g")
    c1.number_input("X-axis point x", value=1.0, key="x1", format="%g")
    c2.number_input("Y-axis point y", value=1.0, key="y1", format="%g")
    if st.button("Reset calibration points", width="stretch"):
        ss.calib = {"O": None, "X": None, "Y": None}
        set_pending(calib_target="O", mode=MODES[0])
        points_changed()
        st.rerun()

    st.header("Export")
    sig = st.slider("Significant figures in CSV", 3, 12, 6)

if data is None:
    st.info("⬅️ Upload an image of a graph in the sidebar (or tick **Use built-in sample graph**) to begin.")
    with st.expander("How it works", expanded=True):
        st.markdown(
            """
1. **Upload** a PNG, JPG, GIF, TIFF, BMP or WEBP image (colour or greyscale).
2. **Calibrate** – in *Calibrate axes* mode click the **origin** (where the axes cross),
   then a point far along the **x-axis**, then a point far up the **y-axis**.
   Enter the data values of those points in the sidebar. Tick *Log* for logarithmic axes.
   Rotated scans are fine: the calibration follows the clicked axes, not the image edges.
3. **Digitize** – switch to *Digitize points* and click on the curve. Add more datasets
   with **➕ New dataset**; each gets its own colour and name.
4. Use **zoom** + *Pan* mode for precise clicks, edit values directly in the table,
   and **download the CSV**.
"""
        )
    st.stop()

h = hashlib.md5(data).hexdigest()
if h != ss.img_hash:  # new image -> start fresh
    ss.img_hash = h
    reset_digitizing_state()
    apply_pending()
img = load_image(data)
cal, cal_msg = build_calibration()

left_col, right_col = st.columns([3, 2], gap="large")

# ---------------- Right column: mode, datasets, table ----------------
with right_col:
    st.radio("Click mode", MODES, key="mode", horizontal=False)
    if ss.mode == MODES[0]:
        st.radio("Next click places", list(CALIB), key="calib_target", horizontal=True,
                 format_func=lambda k: CALIB[k][0])
        placed = ", ".join(f"{CALIB[k][0]} ✅" if ss.calib[k] else f"{CALIB[k][0]} ⬜" for k in "OXY")
        st.caption(placed)
    if cal is None:
        st.warning("Calibration incomplete – " + cal_msg)
    else:
        st.success("Axes calibrated ✔")

    st.subheader("Datasets")
    ids = [ds["id"] for ds in ss.datasets]
    names = {ds["id"]: ds["name"] for ds in ss.datasets}
    st.selectbox("Active dataset (new clicks go here)", ids, key="active_id",
                 format_func=lambda i: names.get(i, "?"))
    ds = active_dataset()
    new_name = st.text_input("Dataset name", value=ds["name"], key=f"ds_name_{ds['id']}")
    if new_name != ds["name"]:
        ds["name"] = new_name
        st.rerun()  # refresh the selectbox label
    st.markdown(f"Colour: <span style='color:{ds['color']};font-size:1.4em'>●</span> "
                f"&nbsp; {len(ds['points'])} point(s)", unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("↶ Undo", help="Remove the last point of the active dataset", width="stretch",
                 disabled=not ds["points"]):
        ds["points"].pop()
        points_changed()
        st.rerun()
    if b2.button("🧹 Clear", help="Remove all points of the active dataset", width="stretch",
                 disabled=not ds["points"]):
        ds["points"].clear()
        points_changed()
        st.rerun()
    if b3.button("➕ New", help="Start a new dataset", width="stretch"):
        nd = new_dataset()
        ss.datasets.append(nd)
        set_pending(active_id=nd["id"], mode=MODES[1] if cal else ss.mode)
        points_changed()
        st.rerun()
    if b4.button("🗑 Delete", help="Delete the active dataset", width="stretch"):
        idx = ids.index(ds["id"])
        ss.datasets.pop(idx)
        if not ss.datasets:
            ss.datasets.append(new_dataset())
        set_pending(active_id=ss.datasets[max(0, idx - 1)]["id"])
        points_changed()
        st.rerun()

    # Editable table for the active dataset -> edits move the marks on the image
    st.subheader("Points (editable)")
    if cal is None:
        st.caption("Calibrate the axes to see and edit data values.")
    else:
        vals = px_to_data(cal, ds["points"]) if ds["points"] else np.empty((0, 2))
        df = pd.DataFrame(vals, columns=["x", "y"])
        edited = st.data_editor(
            df,
            key=f"editor_{ds['id']}_{ss.editor_ver}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "x": st.column_config.NumberColumn(ss.xlabel or "x", required=True),
                "y": st.column_config.NumberColumn(ss.ylabel or "y", required=True),
            },
        )
        e = edited[["x", "y"]].to_numpy(dtype=float) if len(edited) else np.empty((0, 2))
        changed = e.shape != vals.shape or not np.allclose(e, vals, rtol=1e-12, atol=0, equal_nan=True)
        if changed:
            if np.isnan(e).any():
                st.info("Finish filling in the new row (both x and y) to place it.")
            elif (cal["xlog"] and (e[:, 0] <= 0).any()) or (cal["ylog"] and (e[:, 1] <= 0).any()):
                st.error("Values on a log axis must be > 0.")
            else:
                ds["points"] = data_to_px(cal, e).tolist()
                points_changed()
                st.rerun()
        st.caption("Edit a cell to move a point, use the row checkbox + 🗑 to delete, "
                   "or add a row to type in a point.")

# ---------------- Left column: zoom controls + clickable image ----------------
with left_col:
    z1, z2, z3, z4 = st.columns([2, 2, 2, 1])
    z1.select_slider("Zoom", ZOOM_LEVELS, key="zoom", format_func=lambda z: f"{z}×")
    z2.slider("Pan ↔ (%)", 0.0, 100.0, key="pan_x", step=0.5, disabled=ss.zoom == 1)
    z3.slider("Pan ↕ (%)", 0.0, 100.0, key="pan_y", step=0.5, disabled=ss.zoom == 1)
    z4.write("")
    if z4.button("Reset", help="Reset zoom"):
        set_pending(zoom=1, pan_x=50, pan_y=50)
        st.rerun()

    box, scale = view_window(img)
    view = render_view(img, box, scale, cal)
    hint = {
        MODES[0]: f"Click to place the **{CALIB[ss.calib_target][0]}**.",
        MODES[1]: (f"Click to add points to **{ds['name']}**." if cal
                   else "⚠️ Calibrate the axes first – clicks are ignored until then."),
        MODES[2]: "Click to centre the zoom window there (choose a zoom > 1×).",
    }[ss.mode]
    st.markdown(hint)
    click = streamlit_image_coordinates(view, key="canvas", width="content", cursor="crosshair")

    # Each click arrives once with a unique timestamp; ignore stale repeats.
    if click and click.get("unix_time") != ss.last_click:
        ss.last_click = click.get("unix_time")
        if ss.mode != MODES[1] or cal is not None:
            left, top, cw, ch = box
            px = left + click["x"] / click["width"] * cw
            py = top + click["y"] / click["height"] * ch
            handle_click(px, py, img)
            st.rerun()

# ---------------- Combined CSV preview + download ----------------
st.divider()
st.subheader("CSV preview (all datasets)")
if cal is None:
    st.caption("Available once the axes are calibrated.")
else:
    rows = build_csv_rows(cal, sig)
    # Unique column headers "Dataset · label" for display (CSV keeps the exact 2-row header)
    seen, headers = {}, []
    for i, lab in enumerate(rows[1]):
        base = f"{rows[0][i - i % 2]} · {lab}"
        seen[base] = seen.get(base, 0) + 1
        headers.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    st.dataframe(pd.DataFrame(rows[2:], columns=headers), hide_index=True, width="stretch")
    st.download_button(
        "⬇️ Download CSV",
        rows_to_csv(rows),
        file_name="digitized_data.csv",
        mime="text/csv",
        type="primary",
    )
