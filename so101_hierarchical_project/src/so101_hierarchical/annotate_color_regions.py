from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Sequence


COLORS = ("red", "green", "blue")
DRAW = {"red": (40, 40, 255), "green": (40, 210, 40), "blue": (255, 100, 20)}


def build_html_annotator(image_path: Path, output_path: Path, *, force: bool = False) -> None:
    """Create a self-contained browser annotator that works with headless OpenCV."""
    if output_path.exists() and not force:
        raise RuntimeError(f"Refusing to overwrite {output_path}; pass --force deliberately")
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"png", "jpg", "jpeg"}:
        raise RuntimeError("The browser annotator supports PNG and JPEG images")
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    source_json = json.dumps(str(image_path.resolve()))
    download_json = json.dumps(image_path.stem + "_annotations.json")
    html = f"""<!doctype html>
<meta charset="utf-8"><title>SO-101 box polygon annotation</title>
<style>
body {{ margin:0; background:#151515; color:white; font:16px sans-serif }}
#bar {{ padding:10px; position:sticky; top:0; background:#222; z-index:2 }}
button {{ margin-right:8px; padding:8px 12px }}
canvas {{ display:block; max-width:100vw; height:auto; cursor:crosshair }}
</style>
<div id="bar"><span id="status">Draw red: click polygon vertices.</span>
<button onclick="commit()">Finish color</button><button onclick="undo()">Undo point</button>
<button onclick="resetCurrent()">Reset current</button><button onclick="downloadJSON()">Download JSON</button></div>
<canvas id="canvas"></canvas>
<script>
const labels=['red','green','blue'], colors={{red:'#ff3030',green:'#30e030',blue:'#3090ff'}};
const img=new Image(); img.src='data:{mime};base64,{encoded}';
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let index=0,points=[],regions=[];
img.onload=()=>{{canvas.width=img.width;canvas.height=img.height;draw()}};
canvas.onclick=e=>{{if(index>=labels.length)return;const r=canvas.getBoundingClientRect();points.push([Math.round((e.clientX-r.left)*canvas.width/r.width),Math.round((e.clientY-r.top)*canvas.height/r.height)]);draw()}};
function polygon(p,color,close){{if(!p.length)return;ctx.strokeStyle=color;ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(...p[0]);p.slice(1).forEach(q=>ctx.lineTo(...q));if(close)ctx.closePath();ctx.stroke();p.forEach(q=>{{ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],5,0,Math.PI*2);ctx.fill()}})}}
function draw(){{ctx.drawImage(img,0,0);regions.forEach(r=>polygon(r.polygon,colors[r.color],true));if(index<labels.length)polygon(points,colors[labels[index]],false)}}
function commit(){{if(points.length<3){{alert('Use at least three points.');return}}regions.push({{color:labels[index],polygon:points}});points=[];index++;document.getElementById('status').textContent=index<labels.length?'Draw '+labels[index]+': click polygon vertices.':'All colors complete. Download JSON.';draw()}}
function undo(){{points.pop();draw()}} function resetCurrent(){{points=[];draw()}}
function downloadJSON(){{if(regions.length!==3){{alert('Finish all three colors first.');return}}const payload={{schema_version:'1.0',camera:'wrist',source_image:{source_json},image_size:{{width:canvas.width,height:canvas.height}},regions}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download={download_json};a.click();URL.revokeObjectURL(a.href)}}
</script>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def annotate(image_path: Path, output_path: Path, *, force: bool = False) -> None:
    import cv2
    import numpy as np

    if output_path.exists() and not force:
        raise RuntimeError(f"Refusing to overwrite {output_path}; pass --force deliberately")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    points: list[tuple[int, int]] = []
    regions: list[dict[str, Any]] = []
    color_index = 0
    window = "Box polygon annotation"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while color_index < len(COLORS):
        color = COLORS[color_index]
        canvas = image.copy()
        for region in regions:
            polygon = np.asarray(region["polygon"], dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, DRAW[region["color"]], 3)
        if points:
            polygon = np.asarray(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], False, DRAW[color], 2)
            for point in points:
                cv2.circle(canvas, point, 4, DRAW[color], -1)
        cv2.putText(
            canvas,
            f"Draw {color}: click vertices; Enter=save polygon, Backspace=undo, Esc=abort",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            raise RuntimeError("Annotation aborted; nothing was written")
        if key in (8, 127) and points:
            points.pop()
        if key in (10, 13) and len(points) >= 3:
            regions.append({"color": color, "polygon": [list(point) for point in points]})
            points.clear()
            color_index += 1
    cv2.destroyAllWindows()
    payload = {
        "schema_version": "1.0",
        "camera": "wrist",
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "source_image": str(image_path.resolve()),
        "regions": regions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Click polygons around red, green, and blue boxes")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("html", "opencv"), default="html")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.mode == "html":
            build_html_annotator(args.image, args.output, force=args.force)
        else:
            annotate(args.image, args.output, force=args.force)
    except Exception as exc:
        print(f"ANNOTATION ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
