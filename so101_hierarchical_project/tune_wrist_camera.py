#!/usr/bin/env python3
"""Interactive V4L2 camera tuning with OpenCV trackbars.

Usage:
  python tune_wrist_camera.py --device /dev/video3 --width 1280 --height 720

Keys: s = save JSON + image; r = restore startup controls; q/Esc = quit.
The program deliberately does not move or connect to any robot.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2

# (V4L2 control name, minimum, maximum, startup fallback)
CONTROLS = {
    'brightness': (-64, 64, 0),
    'contrast': (0, 64, 42),
    'saturation': (0, 128, 64),
    'gamma': (72, 500, 100),
    'gain': (0, 100, 0),
    'sharpness': (0, 6, 3),
    'white_balance_temperature': (2800, 6500, 4600),
    'exposure_time_absolute': (1, 5000, 157),
}


def v4l2(args):
    proc = subprocess.run(['v4l2-ctl', *args], capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def get_ctrl(device, name, fallback=None):
    ok, result = v4l2(['-d', device, f'--get-ctrl={name}'])
    if ok:
        try:
            return int(result.rsplit(':', 1)[-1].strip())
        except (ValueError, IndexError):
            pass
    return fallback


def set_ctrl(device, name, value):
    return v4l2(['-d', device, f'--set-ctrl={name}={int(value)}'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='/dev/video3')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--output', default='camera_tuning')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ok, listing = v4l2(['-d', args.device, '--list-ctrls'])
    if not ok:
        raise SystemExit('Could not query V4L2 camera controls: ' + listing)

    supported = {name: limits for name, limits in CONTROLS.items() if name in listing}
    supported_auto = [name for name in ('white_balance_automatic', 'auto_exposure') if name in listing]
    initial = {name: get_ctrl(args.device, name, limits[2]) for name, limits in supported.items()}
    for name in supported_auto:
        initial[name] = get_ctrl(args.device, name, 1 if name == 'white_balance_automatic' else 3)

    print('Original settings:', json.dumps(initial, indent=2))
    print('Keys: S save image/settings | R restore original controls | Q or Esc quit')
    print('Tip: turn Auto WB OFF to adjust temperature; Auto Exposure OFF to adjust exposure time.')

    cam = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cam.isOpened():
        raise SystemExit('Cannot open camera. Close ffplay/other programs holding this device.')
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    name = 'Wrist Camera - live tuning'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1150, 780)
    positions = {}

    def noop(_):
        pass

    # Auto controls first. auto_exposure: 3 auto, 1 manual on common UVC cameras.
    if 'white_balance_automatic' in supported_auto:
        positions['auto_wb'] = ('white_balance_automatic', 0, 1)
        cv2.createTrackbar('Auto white balance', name, int(initial['white_balance_automatic']), 1, noop)
    if 'auto_exposure' in supported_auto:
        positions['auto_exp'] = ('auto_exposure', 0, 1)
        cv2.createTrackbar('Auto exposure', name, int(initial['auto_exposure'] == 3), 1, noop)

    for ctrl, (minimum, maximum, _) in supported.items():
        label = ctrl.replace('_', ' ').title()
        positions[label] = (ctrl, minimum, maximum)
        cv2.createTrackbar(label, name, max(0, min(maximum-minimum, int(initial[ctrl]) - minimum)), maximum-minimum, noop)

    last_sent = dict(initial)
    last_frame = None
    last_print = 0

    def current_intended():
        values = {}
        if 'white_balance_automatic' in supported_auto:
            values['white_balance_automatic'] = cv2.getTrackbarPos('Auto white balance', name)
        if 'auto_exposure' in supported_auto:
            values['auto_exposure'] = 3 if cv2.getTrackbarPos('Auto exposure', name) else 1
        for label, (ctrl, minimum, maximum) in positions.items():
            if label in ('auto_wb', 'auto_exp'):
                continue
            values[ctrl] = minimum + cv2.getTrackbarPos(label, name)
        return values

    def apply(values):
        # Switch auto controls first; exposure and WB manual settings work only when auto disabled.
        order = ['white_balance_automatic', 'auto_exposure'] + list(supported)
        for ctrl in order:
            if ctrl not in values:
                continue
            target = int(values[ctrl])
            if last_sent.get(ctrl) == target:
                continue
            if ctrl == 'white_balance_temperature' and values.get('white_balance_automatic', 1):
                continue
            if ctrl == 'exposure_time_absolute' and values.get('auto_exposure', 3) != 1:
                continue
            worked, error = set_ctrl(args.device, ctrl, target)
            if not worked:
                print(f'Warning: failed to set {ctrl}={target}: {error}')
            else:
                last_sent[ctrl] = target

    try:
        while True:
            values = current_intended()
            apply(values)
            good, frame = cam.read()
            if not good:
                print('Warning: camera frame read failed')
                time.sleep(0.1)
                continue
            last_frame = frame.copy()
            preview = frame.copy()
            h, w = preview.shape[:2]
            text = f"WB {'auto' if values.get('white_balance_automatic', 1) else 'manual'} | EXP {'auto' if values.get('auto_exposure', 3) == 3 else 'manual'} | S=save R=reset Q=quit"
            cv2.rectangle(preview, (0, 0), (min(w, 920), 40), (0, 0, 0), -1)
            cv2.putText(preview, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(name, preview)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                stem = f'wrist_{timestamp}'
                path = out / f'{stem}.png'
                cv2.imwrite(str(path), last_frame)
                actual = {k: get_ctrl(args.device, k, v) for k, v in values.items()}
                settings = {'camera': args.device, 'requested': values, 'readback': actual,
                            'timestamp': timestamp, 'image': str(path)}
                (out / f'{stem}.json').write_text(json.dumps(settings, indent=2))
                print('Saved', path, 'and settings JSON')
            if key == ord('r'):
                # Put initial values on trackbars; the following loop writes them.
                if 'white_balance_automatic' in supported_auto:
                    cv2.setTrackbarPos('Auto white balance', name, int(initial['white_balance_automatic']))
                if 'auto_exposure' in supported_auto:
                    cv2.setTrackbarPos('Auto exposure', name, int(initial['auto_exposure'] == 3))
                for ctrl, (minimum, maximum, _) in supported.items():
                    cv2.setTrackbarPos(ctrl.replace('_', ' ').title(), name, int(initial[ctrl]) - minimum)
                print('Restoring original settings...')
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print('Camera closed. Settings remain at their latest values.')
        print('Press R before Q if you want to restore original settings.')


if __name__ == '__main__':
    main()
