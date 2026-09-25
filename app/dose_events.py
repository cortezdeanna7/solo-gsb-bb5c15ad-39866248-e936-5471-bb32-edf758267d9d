"""匿名 X-Ray Radiation Dose SR 的 CT 照射事件剂量解析。

数据来源严格限定为“已完成任务 output 目录中的匿名 DICOM 文件”，
**不重新解压或处理原始 ZIP**，只纳入成功写出的实例。

按匿名 SOPInstanceUID 定位唯一成功输出后，要求其 SOP Class 为
X-Ray Radiation Dose SR（1.2.840.10008.5.1.4.1.1.88.67），随后递归遍历
ContentSequence（任意嵌套深度）：

* 以 ConceptNameCodeSequence = (113811, DCM, "CT Irradiation Event") 的
  CONTAINER 内容项作为**事件边界**：容器子树内的剂量数值归属该事件；
  事件容器嵌套时内层事件优先（数值归属**最近**事件）；
* 事件内收集 ConceptNameCodeSequence = (113818, DCM, "CTDIvol") 与
  (113814, DCM, "CT Dose Length Product") 的 NUM 内容项，分别作为该事件的
  CTDIvol 与 DLP；
* 事件容器之外的 NUM 项**一律忽略**——报告级累计值（如
  (113813, DCM, "CT Dose Length Product Total")、
  (113820, DCM, "Mean CTDIvol")）不参与事件取值与合计。

单位只接受 mGy（CTDIvol）与 mGy·cm（DLP）：DCM 编码 113846 / 113847，
或 UCUM 编码 "mGy" / "mGy.cm"（CodingSchemeDesignator 不区分大小写，
UCUM CodeValue 按标准精确匹配）。

成功时返回按内容顺序排列的事件列表与全部事件 DLP 之和（math.fsum 精确求和）。

存在以下任一情况时抛 :class:`DoseEventsError`（409 DOSE_SR_CONFLICT），
在 ``problems[]`` 中一次性给出**含匿名 SOP UID** 的全部问题明细，
**不输出部分事件或部分合计**：

* 实例不是 X-Ray Radiation Dose SR（``not_dose_sr``）；
* 同一匿名 SOPInstanceUID 对应多份物理输出（``ambiguous_output``）；
* 匿名输出不可读 / 磁盘文件缺失（``unreadable`` / ``output_missing``）；
* 文件内 SOPInstanceUID 与请求不符（``sop_instance_uid_mismatch``）；
* 事件内缺少 CTDIvol 或 DLP（``missing_value``）；
* 事件内同一概念出现多次（``duplicate_value``）；
* 数值缺失 / 无法解析 / 非有限值（``non_finite_value``），或 NUM 项缺少
  MeasuredValueSequence（``malformed_measurement``）；
* 单位不是 mGy / mGy·cm（``unit_mismatch``）。
"""
from __future__ import annotations

import math
import os
from typing import Optional

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import XRayRadiationDoseSRStorage

from .anonymizer import FileReport, JobResult

# --------------------------------------------------------------------------- #
# DCM 编码概念（PS3.16）与单位
# --------------------------------------------------------------------------- #
# (CodingSchemeDesignator, CodeValue)，方案名比较时统一大写
CONCEPT_CT_IRRADIATION_EVENT = ("DCM", "113811")  # CT Irradiation Event
CONCEPT_CTDIVOL = ("DCM", "113818")               # CTDIvol
CONCEPT_DLP = ("DCM", "113814")                   # CT Dose Length Product

# 可接受的单位编码：DCM 113846/113847，或 UCUM "mGy"/"mGy.cm"
UNITS_MGY = frozenset({("DCM", "113846"), ("UCUM", "mGy")})
UNITS_MGY_CM = frozenset({("DCM", "113847"), ("UCUM", "mGy.cm")})

CONCEPT_LABEL_CTDIVOL = "CTDIvol"
CONCEPT_LABEL_DLP = "DLP"

# detail.problems[].reason 取值（机器可读）
REASON_NOT_DOSE_SR = "not_dose_sr"
REASON_AMBIGUOUS_OUTPUT = "ambiguous_output"
REASON_OUTPUT_MISSING = "output_missing"
REASON_UNREADABLE = "unreadable"
REASON_SOP_UID_MISMATCH = "sop_instance_uid_mismatch"
REASON_MISSING_VALUE = "missing_value"
REASON_DUPLICATE_VALUE = "duplicate_value"
REASON_NON_FINITE_VALUE = "non_finite_value"
REASON_UNIT_MISMATCH = "unit_mismatch"
REASON_MALFORMED_MEASUREMENT = "malformed_measurement"


class DoseEventsError(Exception):
    """剂量事件解析失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        sop_uid: Optional[str] = None,
        problems: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        detail: dict = {"code": code, "message": message}
        if sop_uid is not None:
            detail["sop_instance_uid"] = sop_uid
        if problems:
            detail["conflict_count"] = len(problems)
            detail["problems"] = problems
        self.detail = detail


# --------------------------------------------------------------------------- #
# 输出定位
# --------------------------------------------------------------------------- #
def _successful_reports(result: JobResult) -> list[FileReport]:
    """任务内真正写出匿名文件、且记录了匿名 SOP UID 的报告。"""
    return [
        report
        for report in result.files
        if report.status != "error" and report.output_name and report.sop_instance_uid
    ]


def _problem(
    sop_uid: str,
    output_file: Optional[str],
    reason: str,
    message: str,
    **extra,
) -> dict:
    problem = {
        "sop_instance_uid": sop_uid,
        "reason": reason,
        "message": message,
    }
    if output_file is not None:
        problem["output_file"] = output_file
    problem.update(extra)
    return problem


def _locate_unique_output(result: JobResult, sop_uid: str) -> FileReport:
    """按匿名 SOPInstanceUID 定位唯一成功输出；找不到 / 多份输出均明确报错。"""
    matches = [
        report
        for report in _successful_reports(result)
        if report.sop_instance_uid == sop_uid
    ]
    if not matches:
        raise DoseEventsError(
            "INSTANCE_NOT_FOUND",
            404,
            "该匿名 SOPInstanceUID 在本任务的成功输出中不存在",
            sop_uid=sop_uid,
        )
    if len(matches) > 1:
        output_files = sorted(r.output_name for r in matches)
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            "同一匿名 SOPInstanceUID 对应多份输出文件，无法确定要解析的剂量 SR",
            sop_uid=sop_uid,
            problems=[
                _problem(
                    sop_uid,
                    None,
                    REASON_AMBIGUOUS_OUTPUT,
                    "同一匿名 SOPInstanceUID 对应多份输出文件，"
                    "无法确定用于剂量解析的唯一实例",
                    output_files=output_files,
                )
            ],
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# 编码概念与 NUM 项解析
# --------------------------------------------------------------------------- #
def _code_tuple(code_item) -> tuple[str, str]:
    """编码项 -> (大写方案名, CodeValue 原文)。"""
    scheme = str(
        getattr(code_item, "CodingSchemeDesignator", "") or ""
    ).strip().upper()
    value = str(getattr(code_item, "CodeValue", "") or "").strip()
    return (scheme, value)


def _concept_code(item) -> Optional[tuple[str, str]]:
    seq = getattr(item, "ConceptNameCodeSequence", None)
    if not seq:
        return None
    return _code_tuple(seq[0])


def _format_units(units: Optional[tuple[str, str]]) -> Optional[str]:
    if units is None:
        return None
    scheme, value = units
    return f"{scheme}:{value}" if scheme else value


def _parse_num_item(
    item,
    *,
    concept_label: str,
    accepted_units: frozenset,
    expected_units: str,
    event_index: int,
    sop_uid: str,
    output_file: str,
    problems: list[dict],
) -> Optional[float]:
    """解析事件内单个 CTDIvol/DLP NUM 项的数值与单位。

    数值与单位分别校验，问题全部记入 ``problems``；任一不通过返回 None。
    """
    measured_seq = getattr(item, "MeasuredValueSequence", None)
    measured = measured_seq[0] if measured_seq else None
    if measured is None:
        problems.append(_problem(
            sop_uid,
            output_file,
            REASON_MALFORMED_MEASUREMENT,
            f"事件 #{event_index} 的 {concept_label} NUM 项缺少 "
            "MeasuredValueSequence，无法读取数值与单位",
            event_index=event_index,
            concept=concept_label,
        ))
        return None

    ok = True

    raw = getattr(measured, "NumericValue", None)
    text = "" if raw is None else str(raw).strip()
    value: Optional[float] = None
    if text:
        try:
            value = float(text)
        except (TypeError, ValueError):
            value = None
    if value is None or not math.isfinite(value):
        problems.append(_problem(
            sop_uid,
            output_file,
            REASON_NON_FINITE_VALUE,
            f"事件 #{event_index} 的 {concept_label} 数值缺失或不是有限数值："
            f"{text or '<空>'}",
            event_index=event_index,
            concept=concept_label,
            raw_value=text,
        ))
        ok = False

    units_seq = getattr(measured, "MeasurementUnitsCodeSequence", None)
    units = _code_tuple(units_seq[0]) if units_seq else None
    if units not in accepted_units:
        problems.append(_problem(
            sop_uid,
            output_file,
            REASON_UNIT_MISMATCH,
            f"事件 #{event_index} 的 {concept_label} 单位必须为 "
            f"{expected_units}，实际为 "
            f"{_format_units(units) or '<缺失 MeasurementUnitsCodeSequence>'}",
            event_index=event_index,
            concept=concept_label,
            expected_units=expected_units,
            actual_units=_format_units(units),
        ))
        ok = False

    return value if ok else None


# --------------------------------------------------------------------------- #
# ContentSequence 递归遍历与事件归集
# --------------------------------------------------------------------------- #
class _EventAccumulator:
    """单个 CT 照射事件容器子树内收集到的 CTDIvol / DLP NUM 项。"""

    __slots__ = ("index", "ctdivol_items", "dlp_items")

    def __init__(self, index: int) -> None:
        self.index = index
        self.ctdivol_items: list = []
        self.dlp_items: list = []


def _collect_events(ds) -> list[_EventAccumulator]:
    """递归遍历 ContentSequence，按事件容器边界归集剂量 NUM 项（内容顺序）。"""
    events: list[_EventAccumulator] = []

    def walk(items, current_event: Optional[_EventAccumulator]) -> None:
        for item in items or []:
            value_type = str(
                getattr(item, "ValueType", "") or ""
            ).strip().upper()
            concept = _concept_code(item)
            if value_type == "CONTAINER" and concept == CONCEPT_CT_IRRADIATION_EVENT:
                # 事件边界：子树数值归属该事件；嵌套事件向内覆盖（最近事件）
                event = _EventAccumulator(len(events))
                events.append(event)
                walk(getattr(item, "ContentSequence", None), event)
                continue
            if value_type == "NUM" and current_event is not None:
                if concept == CONCEPT_CTDIVOL:
                    current_event.ctdivol_items.append(item)
                elif concept == CONCEPT_DLP:
                    current_event.dlp_items.append(item)
            # 非事件容器与其他内容项：当前事件不变，继续向其子内容递归；
            # current_event 为 None 的 NUM（报告级累计值）在此被自然忽略
            walk(getattr(item, "ContentSequence", None), current_event)

    walk(getattr(ds, "ContentSequence", None), None)
    return events


def _single_event_value(
    items: list,
    *,
    concept_label: str,
    accepted_units: frozenset,
    expected_units: str,
    event_index: int,
    sop_uid: str,
    output_file: str,
    problems: list[dict],
) -> Optional[float]:
    """事件内某一概念应恰好出现一次；缺项/重复项记入 problems。"""
    if not items:
        problems.append(_problem(
            sop_uid,
            output_file,
            REASON_MISSING_VALUE,
            f"事件 #{event_index} 缺少 {concept_label} 数值项",
            event_index=event_index,
            concept=concept_label,
        ))
        return None
    if len(items) > 1:
        problems.append(_problem(
            sop_uid,
            output_file,
            REASON_DUPLICATE_VALUE,
            f"事件 #{event_index} 内 {concept_label} 出现 {len(items)} 次，"
            "无法确定唯一取值",
            event_index=event_index,
            concept=concept_label,
            count=len(items),
        ))
        return None
    return _parse_num_item(
        items[0],
        concept_label=concept_label,
        accepted_units=accepted_units,
        expected_units=expected_units,
        event_index=event_index,
        sop_uid=sop_uid,
        output_file=output_file,
        problems=problems,
    )


def _validate_event(
    event: _EventAccumulator,
    sop_uid: str,
    output_file: str,
    problems: list[dict],
) -> Optional[dict]:
    """校验单个事件的 CTDIvol/DLP 完备性与取值；有问题时返回 None。"""
    ctdi_value = _single_event_value(
        event.ctdivol_items,
        concept_label=CONCEPT_LABEL_CTDIVOL,
        accepted_units=UNITS_MGY,
        expected_units="mGy",
        event_index=event.index,
        sop_uid=sop_uid,
        output_file=output_file,
        problems=problems,
    )
    dlp_value = _single_event_value(
        event.dlp_items,
        concept_label=CONCEPT_LABEL_DLP,
        accepted_units=UNITS_MGY_CM,
        expected_units="mGy·cm",
        event_index=event.index,
        sop_uid=sop_uid,
        output_file=output_file,
        problems=problems,
    )
    if ctdi_value is None or dlp_value is None:
        return None
    return {
        "index": event.index,
        "ctdi_vol_mgy": ctdi_value,
        "dlp_mgy_cm": dlp_value,
    }


# --------------------------------------------------------------------------- #
# 文件级校验
# --------------------------------------------------------------------------- #
def _read_anonymized(path: str, sop_uid: str, output_file: str):
    """匿名化已强制写出 File Meta，这里要求严格 DICOM 解析。"""
    try:
        return pydicom.dcmread(path, force=False)
    except (InvalidDicomError, OSError, EOFError) as exc:
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            f"匿名输出不是可读取的 DICOM 文件：{exc}",
            sop_uid=sop_uid,
            problems=[_problem(
                sop_uid, output_file, REASON_UNREADABLE,
                f"匿名输出不是可读取的 DICOM 文件：{exc}",
            )],
        ) from exc
    except Exception as exc:  # 标签编码损坏等
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            f"读取匿名输出失败：{exc}",
            sop_uid=sop_uid,
            problems=[_problem(
                sop_uid, output_file, REASON_UNREADABLE,
                f"读取匿名输出失败：{exc}",
            )],
        ) from exc


def _check_reported_sop_uid(ds, sop_uid: str, output_file: str) -> None:
    actual = str(getattr(ds, "SOPInstanceUID", "") or "").strip()
    if actual and actual != sop_uid:
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            "文件内 SOPInstanceUID 与请求的匿名 SOP UID 不一致，拒绝解析",
            sop_uid=sop_uid,
            problems=[_problem(
                sop_uid, output_file, REASON_SOP_UID_MISMATCH,
                f"文件内 SOPInstanceUID 为 {actual}，与请求的 {sop_uid} 不一致",
            )],
        )


def _check_sop_class(ds, sop_uid: str, output_file: str) -> None:
    sop_class = str(getattr(ds, "SOPClassUID", "") or "").strip()
    if sop_class != str(XRayRadiationDoseSRStorage):
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            "该实例不是 X-Ray Radiation Dose SR，无法解析 CT 照射事件剂量",
            sop_uid=sop_uid,
            problems=[_problem(
                sop_uid, output_file, REASON_NOT_DOSE_SR,
                f"SOPClassUID 为 {sop_class or '<缺失>'}，不是 "
                f"X-Ray Radiation Dose SR（{XRayRadiationDoseSRStorage}）",
                sop_class_uid=sop_class,
            )],
        )


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def extract_dose_events(job, sop_uid: str) -> dict:
    """定位匿名剂量 SR 并解析 CT 照射事件剂量；任何错误均抛 DoseEventsError。"""
    result: JobResult = job.result
    report = _locate_unique_output(result, sop_uid)
    output_file = report.output_name
    path = os.path.join(result.output_dir, output_file)
    if not os.path.isfile(path):
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            "匿名输出文件在任务目录中不存在（可能已被清理）",
            sop_uid=sop_uid,
            problems=[_problem(
                sop_uid, output_file, REASON_OUTPUT_MISSING,
                "匿名输出文件在任务目录中不存在（可能已被清理）",
            )],
        )

    ds = _read_anonymized(path, sop_uid, output_file)
    _check_reported_sop_uid(ds, sop_uid, output_file)
    _check_sop_class(ds, sop_uid, output_file)

    problems: list[dict] = []
    events = [
        item
        for event in _collect_events(ds)
        if (item := _validate_event(event, sop_uid, output_file, problems))
        is not None
    ]
    if problems:
        raise DoseEventsError(
            "DOSE_SR_CONFLICT",
            409,
            "剂量 SR 内容存在缺项/重复项/非有限数值/单位不符等问题，"
            "无法给出完整的事件剂量与合计",
            sop_uid=sop_uid,
            problems=problems,
        )

    return {
        "sop_instance_uid": sop_uid,
        "output_file": output_file,
        "event_count": len(events),
        "events": events,
        "total_dlp_mgy_cm": math.fsum(
            item["dlp_mgy_cm"] for item in events
        ),
    }
