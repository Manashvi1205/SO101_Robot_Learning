from so101_hierarchical.collect_images import (
    _close_preview_windows,
    _render_previews,
    opencv_preview_available,
)


class FakeCv2:
    def __init__(self, gui_line: str, *, fail_render: bool = False):
        self.gui_line = gui_line
        self.fail_render = fail_render

    def getBuildInformation(self):
        return f"OpenCV test build\n  GUI: {self.gui_line}\n"

    def imshow(self, _name, _image):
        if self.fail_render:
            raise RuntimeError("preview backend unavailable")

    def waitKey(self, _delay):
        return -1

    def destroyAllWindows(self):
        if self.fail_render:
            raise RuntimeError("cleanup backend unavailable")


class FakeStream:
    def latest(self, copy_image=False):
        assert copy_image is False
        return type("Frame", (), {"image": object()})()


def test_headless_opencv_is_detected():
    assert opencv_preview_available(FakeCv2("NONE")) is False
    assert opencv_preview_available(FakeCv2("QT5")) is True


def test_preview_runtime_failure_disables_preview_without_raising(capsys):
    cv2_module = FakeCv2("QT5", fail_render=True)
    specs = {"wrist": type("Spec", (), {"preview_name": "Wrist"})()}
    available, quit_requested = _render_previews(
        cv2_module,
        specs,
        {"wrist": FakeStream()},
    )
    assert available is False
    assert quit_requested is False
    assert "continuing with terminal-only capture" in capsys.readouterr().err
    _close_preview_windows(cv2_module)
