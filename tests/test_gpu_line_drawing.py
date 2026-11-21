"""Regression for Windows OpenGL core-profile GL_INVALID_VALUE (1281).

Some OpenGL 3.3 implementations only support 1px lines. The renderer must
not call glLineWidth(2), and must restore depth testing after a draw error.
"""
from types import SimpleNamespace

import pytest

from uv3d import gpu_viewport


def test_draw_cut_lines_does_not_request_unsupported_line_width(monkeypatch):
    calls = []

    def record(name):
        def fn(*args):
            calls.append((name, args))
        return fn

    gl = gpu_viewport.GL
    monkeypatch.setattr(gl, "glLineWidth",
                        lambda width: pytest.fail("glLineWidth must not be called"))
    monkeypatch.setattr(gl, "glBindVertexArray", record("vao"))
    monkeypatch.setattr(gl, "glUniform1i", record("uniform"))
    monkeypatch.setattr(gl, "glGetUniformLocation", lambda *args: 2)
    monkeypatch.setattr(gl, "glDisable", record("disable"))
    monkeypatch.setattr(gl, "glDrawArrays", record("draw"))
    monkeypatch.setattr(gl, "glEnable", record("enable"))
    preview = SimpleNamespace(
        _line_dirty=False,
        _line_count=4,
        _line_vao=3,
        _program=1,
    )
    gpu_viewport.GpuModelViewport._draw_cut_lines(preview)
    assert any(name == "draw" and args[-1] == 4 for name, args in calls)
    assert calls[-1] == ("enable", (gl.GL_DEPTH_TEST,))


def test_draw_restores_depth_test_even_on_failure(monkeypatch):
    gl = gpu_viewport.GL
    monkeypatch.setattr(gl, "glUniform1i", lambda *args: None)
    monkeypatch.setattr(gl, "glGetUniformLocation", lambda *args: 2)
    monkeypatch.setattr(gl, "glBindVertexArray", lambda *args: None)
    monkeypatch.setattr(gl, "glDisable", lambda *args: None)
    restored = []
    monkeypatch.setattr(gl, "glEnable", lambda value: restored.append(value))

    def fail(*args):
        raise RuntimeError("Injected GPU failure")

    monkeypatch.setattr(gl, "glDrawArrays", fail)
    preview = SimpleNamespace(
        _line_dirty=False,
        _line_count=2,
        _line_vao=3,
        _program=1,
    )
    with pytest.raises(RuntimeError, match="Injected GPU failure"):
        gpu_viewport.GpuModelViewport._draw_cut_lines(preview)
    assert restored == [gl.GL_DEPTH_TEST]
