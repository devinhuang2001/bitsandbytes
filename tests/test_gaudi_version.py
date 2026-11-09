"""Regression coverage for optional Gaudi plugin metadata lookup."""

from importlib.metadata import PackageNotFoundError

import pytest
from packaging import version

from bitsandbytes.backends import utils


def test_gaudi_version_missing_package(monkeypatch):
    def missing(_name):
        raise PackageNotFoundError("habana-torch-plugin")

    monkeypatch.setattr(utils, "metadata", missing)
    assert utils.get_gaudi_sw_version() is None


@pytest.mark.parametrize("package_metadata", [{}, {"Name": "habana-torch-plugin"}])
def test_gaudi_version_absent_from_metadata(monkeypatch, package_metadata):
    monkeypatch.setattr(utils, "metadata", lambda _name: package_metadata)
    assert utils.get_gaudi_sw_version() is None


def test_gaudi_version_present(monkeypatch):
    monkeypatch.setattr(
        utils, "metadata", lambda _name: {"Version": "1.20.0"}
    )
    assert utils.get_gaudi_sw_version() == version.parse("1.20.0")
