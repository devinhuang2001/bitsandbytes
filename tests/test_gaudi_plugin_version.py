"""Regression tests for optional Gaudi package-version detection."""

from packaging.version import Version

from bitsandbytes.backends import utils


def test_gaudi_missing_plugin_version(monkeypatch):
    monkeypatch.setattr(utils, "metadata", lambda name: {"Name": name})
    assert utils.get_gaudi_sw_version() is None


def test_gaudi_valid_plugin_version(monkeypatch):
    monkeypatch.setattr(utils, "metadata", lambda name: {"Version": "1.21.0"})
    assert utils.get_gaudi_sw_version() == Version("1.21.0")


def test_gaudi_missing_plugin(monkeypatch):
    def unavailable(name):
        raise LookupError(name)

    monkeypatch.setattr(utils, "metadata", unavailable)
    assert utils.get_gaudi_sw_version() is None
