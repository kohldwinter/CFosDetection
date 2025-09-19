# main.py  — your working baseline + fast overlays + readable ROI counts + filename
# Flow:
# 1) Pick folder
# 2) Draw 4 ROIs (PolygonSelector, ENTER to accept)
# 3) QC editor: left-click add, right-click remove, ENTER to save
# 4) Saves per-image state JSON, QC PNGs, CSV (csv module)
#
# Additions:
# - FAST QC redraws: single-pass vectorized blending (no per-cell addWeighted loop)
# - Optional downscaled QC preview for instant interaction; full-res data & saved QC preserved
# - Per-ROI counts overlaid (large) + filename top-left (large)
# - Slightly better sensitivity to small cells (MIN_AREA 6, gentler morphology)

import os, json, sys, csv, cv2, numpy as np, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

import tkinter as tk
from tkinter import filedialog, messagebox

# ----------------- USER CONFIG -----------------
LOWER_YELLOW = np.array([20, 90, 90], dtype=np.uint8)
UPPER_YELLOW = np.array([45, 255, 255], dtype=np.uint8)

MIN_AREA = 6        # more sensitive to tiny cells; try 4 if needed
MAX_AREA = 5000

REMOVE_RADIUS = 8   # px (full-res) for right-click removal
DOT_RADIUS   = 4
DOT_ALPHA    = 0.6  # keep your original overlay look

ROI_NAMES = ["PL Left", "PL Right", "IL Left", "IL Right"]

# Label sizes (smaller)
ROI_LABEL_FONT_SCALE = 0.9   # was 1.6
FILENAME_FONT_SCALE  = 0.8   # was 1.3
LABEL_THICKNESS      = 2     # was 3

# QC display performance control (preview only; data remain full-res)
QC_MAX_DISPLAY_W = 1600   # shrink wide images to this width for QC for snappy clicks
QC_MAX_DISPLAY_H = 1200   # and/or this height
# ------------------------------------------------

def pick_image_folder():
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select image folder (PNG/JPG/TIF)")
    root.update(); root.destroy()
    if not folder:
        print("No folder selected. Enter a path or press Enter to exit:")
        path = input("> ").strip()
        if not path or not os.path.isdir(path): return None
        return path
    return folder

# ---------- Visual helpers ----------
def draw_text_with_outline(img_rgb, text, org, font_scale=1.0, color=(255,255,255), thickness=2):
    x, y = org
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0: continue
            cv2.putText(img_rgb, text, (x+dx, y+dy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)
    cv2.putText(img_rgb, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def vector_overlay_dots(base_rgb, cells, radius=4, alpha=0.6, scale=1.0):
    """
    Fast: draw all dots to a single mask, then blend once.
    base_rgb: RGB np.uint8
    cells: list of (x,y) in FULL-RES coordinates
    scale: draw on a downscaled preview (cell coords will be scaled inside)
    """
    h, w = base_rgb.shape[:2]
    out = base_rgb.copy()
    if not cells:
        return out

    # Prepare mask at display scale
    if scale != 1.0:
        disp_w, disp_h = int(round(w*scale)), int(round(h*scale))
        mask = np.zeros((disp_h, disp_w), dtype=np.uint8)
        for (x, y) in cells:
            cx, cy = int(round(x*scale)), int(round(y*scale))
            cv2.circle(mask, (cx, cy), int(round(radius*scale)), 255, -1)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y) in cells:
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Blend once with "red" color
    # NOTE: We keep using (255,0,0) like your original. Even though OpenCV is BGR,
    # your pipeline has been using these values consistently; we'll preserve behavior.
    red = np.zeros_like(out)
    red[...] = (255, 0, 0)  # "red" in your convention

    idx = mask > 0
    out[idx] = (alpha * red[idx] + (1.0 - alpha) * out[idx]).astype(np.uint8)
    return out

# ---------- ROI drawing ----------
class ROISelector:
    def __init__(self, ax):
        self.ax = ax
        self.verts = None
        self.selector = PolygonSelector(
            ax, self._onselect, useblit=True,
            props=dict(color="yellow", linewidth=2, alpha=0.9),
            handle_props=dict(markeredgecolor="black", markerfacecolor="white")
        )
        self.cid = ax.figure.canvas.mpl_connect("key_press_event", self._on_key)
    def _onselect(self, verts):
        self.verts = np.array(verts, dtype=np.float32)
    def _on_key(self, event):
        if event.key == "enter":
            plt.close(self.ax.figure)
    def get_vertices(self):
        return self.verts

def select_polygon(img_rgb, roi_name, filename_for_title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb); ax.axis("off")
    ax.set_title(f"{filename_for_title}  |  Draw ROI: {roi_name}\n(Left-click to add points; ENTER to accept)")
    selector = ROISelector(ax)
    plt.show()
    return selector.get_vertices()

# ---------- Detection ----------
def mask_yellow(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    # gentler to keep tiny cells
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def detect_components(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centroids_list = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_AREA or area > MAX_AREA:
            continue
        cx, cy = centroids[i]
        centroids_list.append((float(cx), float(cy)))
    return centroids_list

# ---------- ROI mask / counts ----------
def roi_mask_from_vertices(image_shape, verts):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if verts is None or len(verts) < 3:
        return mask
    poly = verts.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 1)
    return mask

def counts_by_roi(cells, roi_vertices_dict, image_shape):
    h, w = image_shape[:2]
    counts = {}
    roi_masks = {}
    for roi_name, verts_list in roi_vertices_dict.items():
        verts = np.array(verts_list, dtype=np.float32) if verts_list and len(verts_list) >= 3 else None
        roi_masks[roi_name] = roi_mask_from_vertices(image_shape, verts) if verts is not None else np.zeros((h, w), dtype=np.uint8)
    for roi_name in ROI_NAMES:
        roi_mask = roi_masks.get(roi_name, np.zeros((h, w), dtype=np.uint8))
        cnt = 0
        for (x, y) in cells:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w and roi_mask[yi, xi] == 1:
                cnt += 1
        counts[roi_name] = cnt
    return counts

# ---------- Interactive QC editor (fast) ----------
class QCCellEditor:
    """
    Shows current cells (red dots) with per-dot overlay and ROI outlines + counts.
    - Left click: add cell at cursor
    - Right click: remove nearest cell within REMOVE_RADIUS (full-res)
    - ENTER: accept/save
    Uses downscaled preview for speed; maps clicks back to full-res coords.
    """
    def __init__(self, img_rgb, initial_cells, roi_polys=None, roi_vertices_dict=None, filename_basename=""):
        self.img_full = img_rgb                      # full-res RGB
        self.h, self.w = img_rgb.shape[:2]
        # scale preview (only for display)
        sx = QC_MAX_DISPLAY_W / self.w
        sy = QC_MAX_DISPLAY_H / self.h
        self.scale = min(1.0, sx, sy)
        if self.scale != 1.0:
            self.img_disp = cv2.resize(self.img_full, (int(self.w*self.scale), int(self.h*self.scale)), interpolation=cv2.INTER_AREA)
        else:
            self.img_disp = self.img_full

        self.cells = [(float(x), float(y)) for (x, y) in initial_cells]    # full-res coordinates
        self.roi_vertices = roi_vertices_dict or {}
        # scale ROI polys for preview drawing
        self.roi_polys_disp = []
        for verts in (roi_polys or []):
            if verts is None or len(verts) < 3:
                continue
            if self.scale != 1.0:
                v = verts.copy()
                v[:, 0] = v[:, 0] * self.scale
                v[:, 1] = v[:, 1] * self.scale
                self.roi_polys_disp.append(v)
            else:
                self.roi_polys_disp.append(verts)
        self.filename = filename_basename
        self.im_artist = None

    def _display_overlay(self):
        # build overlay on the DISPLAY image (fast), using scaled cells
        overlay = self.img_disp.copy()
        if self.scale != 1.0:
            scaled_cells = [(x*self.scale, y*self.scale) for (x, y) in self.cells]
        else:
            scaled_cells = self.cells
        overlay = vector_overlay_dots(overlay, scaled_cells, radius=DOT_RADIUS, alpha=DOT_ALPHA, scale=1.0)

        # ROI outlines (display polys)
        for verts in self.roi_polys_disp:
            cv2.polylines(overlay, [verts.astype(np.int32)], True, (0, 255, 0), 2)

        # Per-ROI counts computed in FULL-RES
        counts = counts_by_roi(self.cells, self.roi_vertices, self.img_full.shape)
        for roi_name in ROI_NAMES:
            verts_list = self.roi_vertices.get(roi_name, [])
            if verts_list and len(verts_list) >= 3:
                verts = np.array(verts_list, dtype=np.float32)
                if self.scale != 1.0:
                    v0x, v0y = int(verts[0,0]*self.scale), int(verts[0,1]*self.scale)
                else:
                    v0x, v0y = int(verts[0,0]), int(verts[0,1])
                label = f"{roi_name}: {counts.get(roi_name, 0)}"
                draw_text_with_outline(
                    overlay, label, (v0x, max(30, v0y - 10)),
                    font_scale=ROI_LABEL_FONT_SCALE, color=(0,255,0), thickness=LABEL_THICKNESS
                )

        # Filename top-left
        if self.filename:
            draw_text_with_outline(
                overlay, self.filename, (10, 35),
                font_scale=FILENAME_FONT_SCALE, color=(255,255,255), thickness=LABEL_THICKNESS
            )
        return overlay

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        xd, yd = float(event.xdata), float(event.ydata)
        # map display coords back to full-res
        x = xd / self.scale
        y = yd / self.scale

        if event.button == 1:  # add
            self.cells.append((x, y))
            self._redraw_fast()
        elif event.button == 3:  # remove nearest (compare in full-res)
            if not self.cells:
                return
            d2 = [(x - cx)**2 + (y - cy)**2 for (cx, cy) in self.cells]
            idx = int(np.argmin(d2))
            if np.sqrt(d2[idx]) <= REMOVE_RADIUS:
                self.cells.pop(idx)
                self._redraw_fast()

    def _on_key(self, event):
        if event.key == "enter":
            plt.close(self.fig)

    def _redraw_fast(self):
        overlay = self._display_overlay()
        if self.im_artist is None:
            self.im_artist = self.ax.imshow(overlay, interpolation="nearest")
        else:
            self.im_artist.set_data(overlay)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.axis("off")
        self._redraw_fast()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show()
        return self.cells

# ---------- Paths / state ----------
def state_path_for(base_dir, stem): return os.path.join(base_dir, "output", f"{stem}_state.json")
def roi_json_path_for(base_dir, stem): return os.path.join(base_dir, "output", f"{stem}_rois.json")
def qc_path_for(base_dir, stem): return os.path.join(base_dir, "output", "QC", f"{stem}_QC.png")

def load_state(base_dir, stem):
    p = state_path_for(base_dir, stem)
    if os.path.exists(p):
        with open(p, "r") as f: return json.load(f)
    return None

def save_state(base_dir, stem, state_dict):
    p = state_path_for(base_dir, stem)
    with open(p, "w") as f: json.dump(state_dict, f, indent=2)

# ---------- Save QC image (full-res) ----------
def save_qc_image(base_dir, img_rgb, cells, roi_vertices_dict, stem, filename_for_overlay):
    overlay = vector_overlay_dots(img_rgb, cells, radius=DOT_RADIUS, alpha=DOT_ALPHA, scale=1.0)

    counts = counts_by_roi(cells, roi_vertices_dict, img_rgb.shape)
    for roi_name in ROI_NAMES:
        verts_list = roi_vertices_dict.get(roi_name, [])
        if verts_list and len(verts_list) >= 3:
            verts = np.array(verts_list, dtype=np.float32)
            cv2.polylines(overlay, [verts.astype(np.int32)], True, (0, 255, 0), 2)
            label = f"{roi_name}: {counts.get(roi_name,0)}"
            vx, vy = int(verts[0, 0]), int(verts[0, 1])
            draw_text_with_outline(
                overlay, label, (vx, max(30, vy - 10)),
                font_scale=ROI_LABEL_FONT_SCALE, color=(0,255,0), thickness=LABEL_THICKNESS
            )

    draw_text_with_outline(
        overlay, filename_for_overlay, (10, 35),
        font_scale=FILENAME_FONT_SCALE, color=(255,255,255), thickness=LABEL_THICKNESS
    )
    cv2.imwrite(qc_path_for(base_dir, stem), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# ---------- One image ----------
def process_image(base_dir, image_path):
    basename = os.path.basename(image_path)
    stem, _ = os.path.splitext(basename)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Warning: could not read {image_path}")
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    state = load_state(base_dir, stem)

    # ROIs
    if state and "roi_vertices" in state and state["roi_vertices"]:
        roi_vertices = state["roi_vertices"]
        print("  Loaded existing ROIs.")
    else:
        roi_vertices = {}
        for roi_name in ROI_NAMES:
            verts = select_polygon(img_rgb, roi_name, basename)
            roi_vertices[roi_name] = verts.tolist() if (verts is not None and len(verts) >= 3) else []
        with open(roi_json_path_for(base_dir, stem), "w") as f:
            json.dump(roi_vertices, f, indent=2)

    # Cells
    if state and "final_cells" in state and len(state["final_cells"]) > 0:
        final_cells = [(float(x), float(y)) for x, y in state["final_cells"]]
        print("  Loaded existing final cells. Skipping interactive QC.")
    else:
        binary_mask = mask_yellow(img_bgr)
        detected_cells = detect_components(binary_mask)
        roi_polys = []
        for roi_name in ROI_NAMES:
            verts_list = roi_vertices.get(roi_name, [])
            if verts_list and len(verts_list) >= 3:
                roi_polys.append(np.array(verts_list, dtype=np.float32))
        editor = QCCellEditor(img_rgb, detected_cells, roi_polys=roi_polys,
                              roi_vertices_dict=roi_vertices, filename_basename=basename)
        final_cells = editor.run()

    # Save state & QC image
    state_to_save = {"roi_vertices": roi_vertices, "final_cells": [(float(x), float(y)) for (x, y) in final_cells]}
    save_state(base_dir, stem, state_to_save)
    save_qc_image(base_dir, img_rgb, final_cells, roi_vertices, stem, filename_for_overlay=basename)

    # Counts per ROI
    counts = counts_by_roi(final_cells, roi_vertices, img_rgb.shape)
    row = {"Image": basename, "NumCells_All": len(final_cells)}
    for roi_name in ROI_NAMES:
        row[f"Count_{roi_name}"] = counts.get(roi_name, 0)
    return row

# ---------- CSV ----------
def write_csv_from_states(base_dir):
    rows = []
    for f in os.listdir(base_dir):
        if not f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")): continue
        stem, _ = os.path.splitext(f)
        p = state_path_for(base_dir, stem)
        if not os.path.exists(p): continue
        with open(p, "r") as fh:
            st = json.load(fh)
        if "final_cells" not in st or "roi_vertices" not in st: continue

        img_path = os.path.join(base_dir, f)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        final_cells = [(float(x), float(y)) for (x, y) in st["final_cells"]]
        counts = counts_by_roi(final_cells, st["roi_vertices"], img_rgb.shape)

        row = {"Image": f, "NumCells_All": len(final_cells)}
        for roi_name in ROI_NAMES:
            row[f"Count_{roi_name}"] = counts.get(roi_name, 0)
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["Image"])
    csv_path = os.path.join(base_dir, "output", "cell_counts.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["Image", "NumCells_All"] + [f"Count_{name}" for name in ROI_NAMES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  CSV updated: {csv_path}")

# ---------- Main ----------
def main():
    base_dir = pick_image_folder()
    if not base_dir:
        try: messagebox.showinfo("Cell Counter", "No folder selected. Exiting.")
        except Exception: pass
        return

    output_dir = os.path.join(base_dir, "output")
    qc_dir = os.path.join(output_dir, "QC")
    os.makedirs(qc_dir, exist_ok=True)

    files = [f for f in os.listdir(base_dir) if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
    files.sort()
    if not files:
        try: messagebox.showwarning("Cell Counter", "No images found in that folder.")
        except Exception: print("No images found in that folder.")
        return

    for f in files:
        print(f"Processing {f}...")
        stem, _ = os.path.splitext(f)
        st = load_state(base_dir, stem)
        if st and "final_cells" in st and len(st["final_cells"]) > 0:
            print("  Already processed. Skipping.")
            continue
        _ = process_image(base_dir, os.path.join(base_dir, f))
        write_csv_from_states(base_dir)

    write_csv_from_states(base_dir)
    try:
        messagebox.showinfo("Cell Counter", f"Done.\nQC: {os.path.join(base_dir, 'output', 'QC')}\nCSV/States: {os.path.join(base_dir, 'output')}")
    except Exception:
        print(f"Done. QC in: {os.path.join(base_dir, 'output', 'QC')}")
        print(f"CSV/States in: {os.path.join(base_dir, 'output')}")

if __name__ == "__main__":
    main()
