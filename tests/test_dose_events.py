"""GET /jobs/{id}/instances/{sop_uid}/dose-events：CT 照射事件剂量解析与冲突明细。"""
from __future__ import annotations

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    XRayRadiationDoseSRStorage,
)

from app.crypto import derive_uid
from tests.sample_data import _write_bytes

SECRET = "dose-secret-123456"

STUDY = "4.1"
SERIES = "4.2"

# DCM 编码（PS3.16）
CODE_EVENT = "113811"        # CT Irradiation Event
CODE_CTDIVOL = "113818"      # CTDIvol
CODE_DLP = "113814"          # CT Dose Length Product
CODE_DLP_TOTAL = "113813"    # CT Dose Length Product Total（报告级累计，应忽略）
CODE_MEAN_CTDIVOL = "113820"  # Mean CTDIvol（报告级累计，应忽略）
UNIT_MGY = "113846"
UNIT_MGY_CM = "113847"


# --------------------------------------------------------------------------- #
# 合成 DICOM
# --------------------------------------------------------------------------- #
def _code_item(value: str, meaning: str, scheme: str = "DCM") -> Dataset:
    item = Dataset()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def _num_item(
    concept_code: str,
    meaning: str,
    value,
    *,
    units: str | None = UNIT_MGY,
    units_scheme: str = "DCM",
    units_meaning: str = "mGy",
    drop_measured: bool = False,
    drop_numeric: bool = False,
) -> Dataset:
    item = Dataset()
    item.ValueType = "NUM"
    item.RelationshipType = "CONTAINS"
    item.ConceptNameCodeSequence = [_code_item(concept_code, meaning)]
    if not drop_measured:
        measured = Dataset()
        if not drop_numeric:
            measured.NumericValue = value
        if units is not None:
            measured.MeasurementUnitsCodeSequence = [
                _code_item(units, units_meaning, units_scheme)
            ]
        item.MeasuredValueSequence = [measured]
    return item


def _container(code: str, meaning: str, children: list) -> Dataset:
    item = Dataset()
    item.ValueType = "CONTAINER"
    item.RelationshipType = "CONTAINS"
    item.ContinuityOfContent = "SEPARATE"
    item.ConceptNameCodeSequence = [_code_item(code, meaning)]
    if children:
        item.ContentSequence = children
    return item


def _event(children: list) -> Dataset:
    return _container(CODE_EVENT, "CT Irradiation Event", children)


def _ctdi(value="5.0", **kwargs) -> Dataset:
    kwargs.setdefault("units", UNIT_MGY)
    kwargs.setdefault("units_meaning", "mGy")
    return _num_item(CODE_CTDIVOL, "CTDIvol", value, **kwargs)


def _dlp(value="100.0", **kwargs) -> Dataset:
    kwargs.setdefault("units", UNIT_MGY_CM)
    kwargs.setdefault("units_meaning", "mGy-cm")
    return _num_item(CODE_DLP, "CT Dose Length Product", value, **kwargs)


def _accumulated_container(children: list) -> Dataset:
    return _container("113702", "CT Accumulated Dose Data", children)


def _dose_sr_bytes(sop_uid: str, content_items: list, *,
                   series_uid: str = SERIES, study_uid: str = STUDY) -> bytes:
    """构造 X-Ray Radiation Dose SR：根容器下挂给定内容项。"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = XRayRadiationDoseSRStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = XRayRadiationDoseSRStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = "P-1"
    ds.PatientName = "Zhang^San"
    ds.Modality = "SR"
    ds.ContentDate = "20240301"
    ds.ContentTime = "101500"
    ds.InstanceNumber = 1
    root = Dataset()
    root.ValueType = "CONTAINER"
    root.ContinuityOfContent = "SEPARATE"
    root.ConceptNameCodeSequence = [
        _code_item("113701", "X-Ray Radiation Dose Report")
    ]
    root.ContentSequence = content_items
    ds.ContentSequence = [root]
    return _write_bytes(ds)


def _ct_image_bytes(sop_uid: str, *, series_uid: str = SERIES,
                    study_uid: str = STUDY) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = "P-1"
    ds.PatientName = "Zhang^San"
    ds.Modality = "CT"
    return _write_bytes(ds)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    import importlib
    import app.config
    import app.jobs
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _upload(client, files: dict[str, bytes]):
    r = client.post(
        "/jobs",
        files={"file": ("studies.zip", _zip_bytes(files), "application/zip")},
        data={"secret": SECRET, "shift_dates": "false"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _anon(original: str) -> str:
    return derive_uid(SECRET, original)


def _dose_events(client, job_id: str, sop_uid: str):
    return client.get(f"/jobs/{job_id}/instances/{_anon(sop_uid)}/dose-events")


# --------------------------------------------------------------------------- #
# 成功路径
# --------------------------------------------------------------------------- #
def test_dose_events_two_events_and_report_totals_ignored(client):
    """两个事件按内容顺序返回；报告级累计值（DLP Total / Mean CTDIvol）被忽略。"""
    sr = _dose_sr_bytes("7.1.1", [
        _accumulated_container([
            _num_item(CODE_DLP_TOTAL, "CT Dose Length Product Total", "9999.9",
                      units=UNIT_MGY_CM, units_meaning="mGy-cm"),
            _num_item(CODE_MEAN_CTDIVOL, "Mean CTDIvol", "88.8"),
        ]),
        _event([_ctdi("5.3"), _dlp("165.1")]),
        _event([_ctdi("12.1"), _dlp("480.2")]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    r = _dose_events(client, job["id"], "7.1.1")
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["sop_instance_uid"] == _anon("7.1.1")
    assert data["output_file"] == f"{_anon('7.1.1')}.dcm"
    assert data["event_count"] == 2
    assert data["events"] == [
        {"index": 0, "ctdi_vol_mgy": 5.3, "dlp_mgy_cm": 165.1},
        {"index": 1, "ctdi_vol_mgy": 12.1, "dlp_mgy_cm": 480.2},
    ]
    # 合计只含事件 DLP；报告级 9999.9 不得进入合计
    assert data["total_dlp_mgy_cm"] == pytest.approx(645.3)


def test_dose_events_same_coded_values_outside_events_ignored(client):
    """事件容器之外、即使编码与 CTDIvol/DLP 相同的 NUM 项也被忽略。"""
    sr = _dose_sr_bytes("7.1.1", [
        _accumulated_container([
            _num_item(CODE_CTDIVOL, "CTDIvol", "123.4"),
            _num_item(CODE_DLP, "CT Dose Length Product", "555.5",
                      units=UNIT_MGY_CM, units_meaning="mGy-cm"),
        ]),
        _event([_ctdi("5.0"), _dlp("100.0")]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    data = _dose_events(client, job["id"], "7.1.1").json()
    assert data["event_count"] == 1
    assert data["events"] == [
        {"index": 0, "ctdi_vol_mgy": 5.0, "dlp_mgy_cm": 100.0}
    ]
    assert data["total_dlp_mgy_cm"] == pytest.approx(100.0)


def test_dose_events_found_inside_wrapper_containers(client):
    """事件容器包在普通容器（如 CT Irradiation Event Data）内也能递归找到。"""
    wrapper = _container("113703", "CT Irradiation Event Data", [
        _event([_ctdi("3.1"), _dlp("42.5")]),
    ])
    job = _upload(client, {"sr.dcm": _dose_sr_bytes("7.1.1", [wrapper])})
    data = _dose_events(client, job["id"], "7.1.1").json()
    assert data["event_count"] == 1
    assert data["events"][0]["ctdi_vol_mgy"] == 3.1
    assert data["total_dlp_mgy_cm"] == pytest.approx(42.5)


def test_dose_events_values_in_event_subcontainers_attributed(client):
    """事件内更深子容器（如采集参数容器）中的数值仍归属该事件。"""
    sr = _dose_sr_bytes("7.1.1", [
        _event([
            _container("113815", "CT Acquisition Parameters", [
                _ctdi("6.6"),
            ]),
            _dlp("210.0"),
        ]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    data = _dose_events(client, job["id"], "7.1.1").json()
    assert data["events"] == [
        {"index": 0, "ctdi_vol_mgy": 6.6, "dlp_mgy_cm": 210.0}
    ]


def test_dose_events_nested_event_containers_nearest_wins(client):
    """事件容器嵌套时，内层事件优先：数值归属最近（内层）事件。"""
    sr = _dose_sr_bytes("7.1.1", [
        _event([
            _ctdi("5.0"),
            _dlp("100.0"),
            _event([_ctdi("7.0"), _dlp("200.0")]),
        ]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    data = _dose_events(client, job["id"], "7.1.1").json()
    assert data["event_count"] == 2
    assert data["events"] == [
        {"index": 0, "ctdi_vol_mgy": 5.0, "dlp_mgy_cm": 100.0},
        {"index": 1, "ctdi_vol_mgy": 7.0, "dlp_mgy_cm": 200.0},
    ]
    assert data["total_dlp_mgy_cm"] == pytest.approx(300.0)


def test_dose_events_no_ct_events_returns_empty(client):
    """剂量 SR 中没有 CT 照射事件容器：200，空事件列表，合计 0。"""
    sr = _dose_sr_bytes("7.1.1", [
        _accumulated_container([
            _num_item(CODE_DLP_TOTAL, "CT Dose Length Product Total", "300.0",
                      units=UNIT_MGY_CM, units_meaning="mGy-cm"),
        ]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    r = _dose_events(client, job["id"], "7.1.1")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["event_count"] == 0
    assert data["events"] == []
    assert data["total_dlp_mgy_cm"] == 0.0


def test_dose_events_ucum_units_accepted(client):
    """单位接受 UCUM 编码 mGy / mGy.cm。"""
    sr = _dose_sr_bytes("7.1.1", [
        _event([
            _ctdi("4.2", units="mGy", units_scheme="UCUM",
                  units_meaning="milligray"),
            _dlp("150.5", units="mGy.cm", units_scheme="UCUM",
                 units_meaning="milligray centimeter"),
        ]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    data = _dose_events(client, job["id"], "7.1.1").json()
    assert data["events"] == [
        {"index": 0, "ctdi_vol_mgy": 4.2, "dlp_mgy_cm": 150.5}
    ]


def test_dose_events_integer_and_scientific_values(client):
    """DS 整数与科学计数法取值正常解析。"""
    sr = _dose_sr_bytes("7.1.1", [
        _event([_ctdi("8"), _dlp("1.25e2")]),
    ])
    job = _upload(client, {"sr.dcm": sr})
    data = _dose_events(client, job["id"], "7.1.1").json()
    assert data["events"] == [
        {"index": 0, "ctdi_vol_mgy": 8.0, "dlp_mgy_cm": 125.0}
    ]


# --------------------------------------------------------------------------- #
# 404 / 409（任务与实例定位）
# --------------------------------------------------------------------------- #
def test_dose_events_job_not_found_404(client):
    r = client.get(f"/jobs/deadbeef/instances/{_anon('7.1.1')}/dose-events")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "JOB_NOT_FOUND"
    assert detail["sop_instance_uid"] == _anon("7.1.1")


def test_dose_events_instance_not_found_404(client):
    job = _upload(client, {"sr.dcm": _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0"), _dlp("100.0")]),
    ])})
    r = _dose_events(client, job["id"], "7.9.9")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "INSTANCE_NOT_FOUND"
    assert detail["sop_instance_uid"] == _anon("7.9.9")


def test_dose_events_failed_job_409(client):
    r = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": SECRET},
    )
    job = r.json()
    assert job["status"] == "failed"
    resp = client.get(f"/jobs/{job['id']}/instances/x/dose-events")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "JOB_NOT_READY"
    assert detail["sop_instance_uid"] == "x"


def test_dose_events_not_dose_sr_409(client):
    """匿名实例存在但不是剂量 SR（如 CT 影像）：409 not_dose_sr。"""
    job = _upload(client, {"ct.dcm": _ct_image_bytes("7.1.1")})
    r = _dose_events(client, job["id"], "7.1.1")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DOSE_SR_CONFLICT"
    problems = [p for p in detail["problems"] if p["reason"] == "not_dose_sr"]
    assert len(problems) == 1
    assert problems[0]["sop_instance_uid"] == _anon("7.1.1")
    assert problems[0]["sop_class_uid"] == str(CTImageStorage)


# --------------------------------------------------------------------------- #
# 409（事件内容问题）：缺项 / 重复项 / 非有限数值 / 单位不符
# --------------------------------------------------------------------------- #
def _conflict(client, files):
    job = _upload(client, files)
    r = _dose_events(client, job["id"], "7.1.1")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "DOSE_SR_CONFLICT"
    assert detail["sop_instance_uid"] == _anon("7.1.1")
    # 不输出部分合计：409 响应中不得出现事件列表或合计字段
    assert "events" not in detail
    assert "total_dlp_mgy_cm" not in detail
    return detail


def test_conflict_missing_dlp(client):
    sr = _dose_sr_bytes("7.1.1", [_event([_ctdi("5.0")])])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"] if p["reason"] == "missing_value"]
    assert len(problems) == 1
    p = problems[0]
    assert p["sop_instance_uid"] == _anon("7.1.1")
    assert p["event_index"] == 0
    assert p["concept"] == "DLP"
    assert detail["conflict_count"] == len(detail["problems"])


def test_conflict_missing_both_values_listed_together(client):
    sr = _dose_sr_bytes("7.1.1", [_event([])])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"] if p["reason"] == "missing_value"]
    assert {p["concept"] for p in problems} == {"CTDIvol", "DLP"}


def test_conflict_duplicate_ctdivol(client):
    sr = _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0"), _ctdi("6.0"), _dlp("100.0")]),
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"] if p["reason"] == "duplicate_value"]
    assert len(problems) == 1
    p = problems[0]
    assert p["concept"] == "CTDIvol"
    assert p["count"] == 2
    assert p["event_index"] == 0


def test_conflict_non_finite_value(client):
    sr = _dose_sr_bytes("7.1.1", [
        _event([_ctdi("nan"), _dlp("100.0")]),
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"]
                if p["reason"] == "non_finite_value"]
    assert len(problems) == 1
    p = problems[0]
    assert p["concept"] == "CTDIvol"
    assert p["raw_value"] == "nan"
    assert p["event_index"] == 0


def test_conflict_infinite_value(client):
    sr = _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0"), _dlp("inf")]),
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"]
                if p["reason"] == "non_finite_value"]
    assert len(problems) == 1
    assert problems[0]["concept"] == "DLP"
    assert problems[0]["raw_value"] == "inf"


def test_conflict_unit_mismatch(client):
    # CTDIvol 误用 mGy·cm 单位编码
    sr = _dose_sr_bytes("7.1.1", [
        _event([
            _ctdi("5.0", units=UNIT_MGY_CM, units_meaning="mGy-cm"),
            _dlp("100.0"),
        ]),
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"] if p["reason"] == "unit_mismatch"]
    assert len(problems) == 1
    p = problems[0]
    assert p["concept"] == "CTDIvol"
    assert p["expected_units"] == "mGy"
    assert p["actual_units"] == f"DCM:{UNIT_MGY_CM}"


def test_conflict_missing_units_code(client):
    # 缺少 MeasurementUnitsCodeSequence：无法确认单位 -> unit_mismatch
    sr = _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0", units=None), _dlp("100.0")]),
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"] if p["reason"] == "unit_mismatch"]
    assert len(problems) == 1
    assert problems[0]["concept"] == "CTDIvol"
    assert problems[0]["actual_units"] is None


def test_conflict_malformed_measurement(client):
    # NUM 项缺少 MeasuredValueSequence
    item = _num_item(CODE_CTDIVOL, "CTDIvol", None, drop_measured=True)
    sr = _dose_sr_bytes("7.1.1", [_event([item, _dlp("100.0")])])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"]
                if p["reason"] == "malformed_measurement"]
    assert len(problems) == 1
    assert problems[0]["concept"] == "CTDIvol"


def test_conflict_missing_numeric_value(client):
    sr = _dose_sr_bytes("7.1.1", [
        _event([_num_item(CODE_CTDIVOL, "CTDIvol", None, drop_numeric=True),
                _dlp("100.0")]),
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    problems = [p for p in detail["problems"]
                if p["reason"] == "non_finite_value"]
    assert len(problems) == 1
    assert problems[0]["raw_value"] == ""


def test_conflict_problems_across_events_collected_at_once(client):
    """多个事件的问题在同一次 409 中全部给出，各自带事件序号。"""
    sr = _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0")]),                      # 事件 0：缺 DLP
        _event([_ctdi("6.0"), _ctdi("6.5"), _dlp("1.0")]),  # 事件 1：CTDIvol 重复
        _event([_ctdi("7.0"), _dlp("2.0")]),          # 事件 2：正常
    ])
    detail = _conflict(client, {"sr.dcm": sr})
    reasons = {(p["event_index"], p["reason"]) for p in detail["problems"]}
    assert (0, "missing_value") in reasons
    assert (1, "duplicate_value") in reasons
    assert detail["conflict_count"] == len(detail["problems"]) == 2


# --------------------------------------------------------------------------- #
# 409（输出定位问题）
# --------------------------------------------------------------------------- #
def test_conflict_ambiguous_sop_output(client):
    # 输入两个同 SOP UID 的剂量 SR -> 输出 <uid>.dcm / <uid>-2.dcm
    files = {
        "dup_a.dcm": _dose_sr_bytes("7.1.77", [_event([_ctdi("5.0"), _dlp("1.0")])]),
        "dup_b.dcm": _dose_sr_bytes("7.1.77", [_event([_ctdi("6.0"), _dlp("2.0")])]),
    }
    job = _upload(client, files)
    r = _dose_events(client, job["id"], "7.1.77")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DOSE_SR_CONFLICT"
    problems = [p for p in detail["problems"] if p["reason"] == "ambiguous_output"]
    assert len(problems) == 1
    p = problems[0]
    assert p["sop_instance_uid"] == _anon("7.1.77")
    assert p["output_files"] == sorted(
        [f"{_anon('7.1.77')}.dcm", f"{_anon('7.1.77')}-2.dcm"]
    )


def test_dose_events_output_file_removed_conflict(client):
    job = _upload(client, {"sr.dcm": _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0"), _dlp("100.0")]),
    ])})
    from app.main import registry
    job_obj = registry.get(job["id"])
    target = os.path.join(job_obj.result.output_dir, f"{_anon('7.1.1')}.dcm")
    os.remove(target)
    r = _dose_events(client, job["id"], "7.1.1")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DOSE_SR_CONFLICT"
    problems = [p for p in detail["problems"] if p["reason"] == "output_missing"]
    assert len(problems) == 1
    assert problems[0]["sop_instance_uid"] == _anon("7.1.1")


def test_dose_events_unreadable_output_conflict(client):
    job = _upload(client, {"sr.dcm": _dose_sr_bytes("7.1.1", [
        _event([_ctdi("5.0"), _dlp("100.0")]),
    ])})
    from app.main import registry
    job_obj = registry.get(job["id"])
    target = os.path.join(job_obj.result.output_dir, f"{_anon('7.1.1')}.dcm")
    with open(target, "wb") as fh:
        fh.write(b"\x00\x01not dicom anymore")
    r = _dose_events(client, job["id"], "7.1.1")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DOSE_SR_CONFLICT"
    problems = [p for p in detail["problems"] if p["reason"] == "unreadable"]
    assert len(problems) == 1
    assert problems[0]["sop_instance_uid"] == _anon("7.1.1")


def test_conflict_response_contains_only_anonymized_uids(client):
    """冲突明细不得泄露原始 SOP/Study/Series UID。"""
    sr = _dose_sr_bytes("7.1.1", [_event([_ctdi("5.0")])])
    detail = _conflict(client, {"sr.dcm": sr})
    raw = str(detail)
    assert "7.1.1" not in raw
    assert STUDY not in raw
    assert SERIES not in raw
    assert _anon("7.1.1") in raw


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #
def test_dose_events_endpoint_in_openapi(client):
    r = client.get("/openapi.json")
    paths = r.json()["paths"]
    key = "/jobs/{job_id}/instances/{sop_instance_uid}/dose-events"
    assert key in paths
    assert "get" in paths[key]
    schema = paths[key]["get"]
    assert "DoseEventsOut" in str(schema["responses"]["200"])
    # 根路径端点清单也包含新接口
    endpoints = client.get("/").json()["endpoints"]
    assert "GET /jobs/{job_id}/instances/{sop_instance_uid}/dose-events" in endpoints
