#!/usr/bin/env python3

"""
ABS-ONLY GOLDEN REGIONAL SOLVER
===============================

ABS economic data + golden-ratio regional scoring + tile heat map.

Explicitly excluded:
    - OpenStreetMap
    - Overpass
    - roads
    - road geometry
    - road density
    - geographic tile downloads
    - geographic APIs

ABS sources:
    - Total Value of Dwellings
    - Average Weekly Earnings
    - Producer Price Indexes
    - Building Activity

Outputs:
    abs_only_solver_output/
        downloads/
        abs_data.json
        predictions.json
        predictions.csv
        golden_baseline.json
        predicted_value_heatmap.png
        report.txt

Install:
    pip install requests pandas openpyxl matplotlib

Run:
    python golden_abs_only_regional_solver.py

Fresh ABS downloads:
    python golden_abs_only_regional_solver.py --force

Use cached XLSX files:
    python golden_abs_only_regional_solver.py --cached

Inspect XLSX files:
    python golden_abs_only_regional_solver.py --inspect
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("abs_only_solver_output")
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"

PHI = (1.0 + math.sqrt(5.0)) / 2.0

STATES = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "WA": "Western Australia",
    "TAS": "Tasmania",
    "NT": "Northern Territory",
    "ACT": "Australian Capital Territory",
}


# Official ABS release pages.
ABS_RELEASES = {

    "dwellings":
        "https://www.abs.gov.au/statistics/economy/"
        "price-indexes-and-inflation/"
        "total-value-dwellings/latest-release",

    "earnings":
        "https://www.abs.gov.au/statistics/labour/"
        "earnings-and-working-conditions/"
        "average-weekly-earnings-australia/"
        "may-2026",

    "ppi":
        "https://www.abs.gov.au/statistics/economy/"
        "price-indexes-and-inflation/"
        "producer-price-indexes-australia/"
        "jun-2026",

    "building":
        "https://www.abs.gov.au/statistics/industry/"
        "building-and-construction/"
        "building-activity-australia/"
        "latest-release",
}


HEADERS = {
    "User-Agent": (
        "GoldenABSRegionalSolver/2.0 "
        "(ABS XLSX research client)"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
}


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value)

    text = (
        text
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def number(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):

        try:
            if not math.isfinite(float(value)):
                return None
        except Exception:
            return None

        return float(value)

    text = clean_text(value)

    if not text:
        return None

    if text.lower() in {
        "-",
        "—",
        "–",
        "..",
        "...",
        "n.p.",
        "np",
        "n/a",
        "na",
    }:
        return None

    negative = (
        text.startswith("(")
        and text.endswith(")")
    )

    text = (
        text
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
    )

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        text,
    )

    if not match:
        return None

    value = float(
        match.group(0)
    )

    if negative:
        value = -value

    return value


def finite(value) -> bool:

    if value is None:
        return False

    try:
        return math.isfinite(
            float(value)
        )
    except Exception:
        return False


def safe_filename(name: str) -> str:

    name = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    return name[:180]


def json_clean(value):

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):

        return {
            str(k): json_clean(v)
            for k, v in value.items()
        }

    if isinstance(value, list):

        return [
            json_clean(v)
            for v in value
        ]

    return value


def save_json(
    filename,
    data,
):

    path = OUTPUT_DIR / filename

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_clean(data),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved: {path}"
    )


# ============================================================
# ABS DOWNLOADER
# ============================================================

class ABSDownloader:

    def __init__(
        self,
        force=False,
    ):

        self.force = force

        self.session = (
            requests.Session()
        )

        self.session.headers.update(
            HEADERS
        )

    # --------------------------------------------------------
    # Discover official XLSX links
    # --------------------------------------------------------

    def discover_xlsx(
        self,
        release_url,
    ):

        response = (
            self.session.get(
                release_url,
                timeout=60,
            )
        )

        response.raise_for_status()

        html = response.text

        candidates = []

        patterns = [

            r'href=["\']([^"\']+\.xlsx'
            r'(?:\?[^"\']*)?)["\']',

            r'(https?://[^"\'>\s]+\.xlsx'
            r'(?:\?[^"\'>\s]*)?)',

            r'((?:/|https?://)[^"\'>\s]+\.xlsx)',
        ]

        for pattern in patterns:

            candidates.extend(
                re.findall(
                    pattern,
                    html,
                    flags=re.IGNORECASE,
                )
            )

        links = []

        for raw in candidates:

            raw = raw.replace(
                "&amp;",
                "&",
            )

            url = urljoin(
                release_url,
                raw,
            )

            parsed = urlparse(
                url
            )

            if parsed.scheme not in {
                "http",
                "https",
            }:
                continue

            if parsed.netloc.lower() not in {
                "www.abs.gov.au",
                "abs.gov.au",
            }:
                continue

            if ".xlsx" not in (
                parsed.path.lower()
            ):
                continue

            if url not in links:
                links.append(url)

        return links

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    def download(
        self,
        url,
    ):

        filename = (
            urlparse(url)
            .path
            .split("/")[-1]
        )

        filename = safe_filename(
            filename
        )

        if not filename.lower().endswith(
            ".xlsx"
        ):
            filename += ".xlsx"

        destination = (
            DOWNLOAD_DIR / filename
        )

        if (
            destination.exists()
            and destination.stat().st_size > 0
            and not self.force
        ):

            print(
                f"  cached: {filename}"
            )

            return destination

        print(
            f"  downloading: {filename}"
        )

        response = (
            self.session.get(
                url,
                timeout=180,
                stream=True,
            )
        )

        response.raise_for_status()

        temporary = (
            destination.with_suffix(
                ".part"
            )
        )

        with temporary.open(
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

        if (
            not temporary.exists()
            or temporary.stat().st_size == 0
        ):

            raise RuntimeError(
                "Downloaded ABS XLSX is empty: "
                + url
            )

        temporary.replace(
            destination
        )

        time.sleep(
            0.4
        )

        return destination

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    def download_dataset(
        self,
        dataset,
        url,
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"ABS DATASET: {dataset.upper()}"
        )

        print(
            "=" * 70
        )

        print(
            url
        )

        links = self.discover_xlsx(
            url
        )

        if not links:

            raise RuntimeError(
                "No official ABS XLSX files "
                f"found at {url}"
            )

        print(
            f"Found {len(links)} XLSX file(s)."
        )

        files = []

        for link in links:

            try:

                files.append(
                    self.download(
                        link
                    )
                )

            except Exception as exc:

                print(
                    f"  download failed: "
                    f"{exc}"
                )

        return files

    # --------------------------------------------------------
    # Everything
    # --------------------------------------------------------

    def download_all(self):

        result = {}

        for dataset, url in (
            ABS_RELEASES.items()
        ):

            try:

                result[dataset] = (
                    self.download_dataset(
                        dataset,
                        url,
                    )
                )

            except Exception as exc:

                print(
                    f"\nABS {dataset} failed:"
                )

                print(
                    f"  {exc}"
                )

                result[dataset] = []

        return result


# ============================================================
# WORKBOOK HELPERS
# ============================================================

def workbook_sheets(
    path: Path,
):

    excel = pd.ExcelFile(
        path,
        engine="openpyxl",
    )

    return list(
        excel.sheet_names
    )


def read_sheet(
    path: Path,
    sheet,
):

    return pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        engine="openpyxl",
    )


def state_aliases(
    code,
):

    aliases = [
        STATES[code],
        code,
    ]

    if code == "ACT":
        aliases += [
            "Australian Capital Territory",
            "A.C.T.",
        ]

    if code == "NT":
        aliases += [
            "Northern Territory",
            "N.T.",
        ]

    return [
        x.lower()
        for x in aliases
    ]


def find_state_rows(
    df,
    code,
):

    aliases = state_aliases(
        code
    )

    matches = []

    for row_index in range(
        len(df)
    ):

        row_text = " ".join(
            clean_text(x).lower()
            for x in df.iloc[
                row_index
            ].tolist()
        )

        for alias in aliases:

            if re.search(
                rf"(?<![a-z])"
                rf"{re.escape(alias)}"
                rf"(?![a-z])",
                row_text,
            ):

                matches.append(
                    row_index
                )

                break

    return matches


def make_headers(
    df,
    rows=15,
):

    labels = {}

    rows = min(
        rows,
        len(df),
    )

    for col in range(
        len(df.columns)
    ):

        pieces = []

        for row in range(
            rows
        ):

            text = clean_text(
                df.iloc[
                    row,
                    col,
                ]
            )

            if text:
                pieces.append(
                    text
                )

        pieces = list(
            dict.fromkeys(
                pieces
            )
        )

        labels[col] = (
            " | ".join(
                pieces
            )
        )

    return labels


def header_score(
    header,
    required,
    optional=None,
    forbidden=None,
):

    text = header.lower()

    if forbidden:

        for word in forbidden:

            if word.lower() in text:
                return -999999

    for word in required:

        if word.lower() not in text:
            return -999999

    score = (
        100 * len(required)
    )

    if optional:

        for word in optional:

            if word.lower() in text:
                score += 10

    return score


def find_series(
    workbook: Path,
    state,
    required,
    optional=None,
    forbidden=None,
    preferred_sheets=None,
):

    best = None
    best_score = -999999

    try:

        sheets = workbook_sheets(
            workbook
        )

    except Exception:

        return None

    if preferred_sheets:

        sheets = sorted(
            sheets,
            key=lambda sheet:
                -sum(
                    keyword.lower()
                    in sheet.lower()
                    for keyword
                    in preferred_sheets
                ),
        )

    for sheet in sheets:

        try:

            df = read_sheet(
                workbook,
                sheet,
            )

        except Exception:

            continue

        state_rows = find_state_rows(
            df,
            state,
        )

        if not state_rows:
            continue

        headers = make_headers(
            df
        )

        for col, header in (
            headers.items()
        ):

            score = header_score(
                header,
                required,
                optional,
                forbidden,
            )

            if score <= best_score:
                continue

            for row in state_rows:

                if col >= len(
                    df.columns
                ):
                    continue

                value = number(
                    df.iloc[
                        row,
                        col,
                    ]
                )

                if value is None:
                    continue

                best_score = score

                best = {
                    "value": value,
                    "workbook":
                        workbook.name,
                    "sheet":
                        sheet,
                    "row":
                        int(row),
                    "column":
                        int(col),
                    "header":
                        header,
                    "score":
                        score,
                }

                break

    return best


# ============================================================
# DWELLINGS
# ============================================================

def parse_dwellings(
    files,
):

    result = {
        code: {
            "total_dwelling_value": None,
            "mean_dwelling_price": None,
            "dwelling_count": None,
            "sources": {},
        }
        for code in STATES
    }

    for code in STATES:

        for workbook in files:

            if result[code][
                "mean_dwelling_price"
            ] is None:

                match = find_series(
                    workbook,
                    code,
                    required=[
                        "mean",
                        "price",
                    ],
                    optional=[
                        "dwelling",
                        "residential",
                    ],
                    forbidden=[
                        "median",
                    ],
                    preferred_sheets=[
                        "mean",
                        "price",
                        "dwelling",
                        "state",
                    ],
                )

                if match:

                    result[code][
                        "mean_dwelling_price"
                    ] = match["value"]

                    result[code][
                        "sources"
                    ][
                        "mean_dwelling_price"
                    ] = match

            if result[code][
                "total_dwelling_value"
            ] is None:

                match = find_series(
                    workbook,
                    code,
                    required=[
                        "total",
                        "value",
                    ],
                    optional=[
                        "dwelling",
                        "residential",
                    ],
                    preferred_sheets=[
                        "value",
                        "dwelling",
                        "state",
                    ],
                )

                if match:

                    result[code][
                        "total_dwelling_value"
                    ] = match["value"]

                    result[code][
                        "sources"
                    ][
                        "total_dwelling_value"
                    ] = match

            if result[code][
                "dwelling_count"
            ] is None:

                match = find_series(
                    workbook,
                    code,
                    required=[
                        "number",
                        "dwellings",
                    ],
                    optional=[
                        "residential",
                    ],
                    preferred_sheets=[
                        "number",
                        "dwelling",
                        "state",
                    ],
                )

                if match:

                    result[code][
                        "dwelling_count"
                    ] = match["value"]

                    result[code][
                        "sources"
                    ][
                        "dwelling_count"
                    ] = match

    return result


# ============================================================
# AVERAGE WEEKLY EARNINGS
# ============================================================

def parse_earnings(
    files,
):

    result = {
        code: {
            "weekly_earnings": None,
            "sources": {},
        }
        for code in STATES
    }

    for code in STATES:

        for workbook in files:

            if result[code][
                "weekly_earnings"
            ] is not None:

                break

            match = find_series(
                workbook,
                code,
                required=[
                    "average",
                    "weekly",
                    "earnings",
                ],
                optional=[
                    "full-time",
                    "adult",
                    "ordinary",
                    "time",
                    "persons",
                ],
                forbidden=[
                    "cash",
                    "industry",
                ],
                preferred_sheets=[
                    "state",
                    "earnings",
                    "table 13",
                    "table 11",
                ],
            )

            if match:

                result[code][
                    "weekly_earnings"
                ] = match["value"]

                result[code][
                    "sources"
                ][
                    "weekly_earnings"
                ] = match

    return result


# ============================================================
# PPI
# ============================================================

def parse_ppi(
    files,
):

    result = {
        code: {
            "construction_price_index":
                None,
            "construction_quarterly_change":
                None,
            "construction_annual_change":
                None,
            "sources": {},
        }
        for code in STATES
    }

    for code in STATES:

        for workbook in files:

            if result[code][
                "construction_annual_change"
            ] is not None:

                break

            try:

                sheets = workbook_sheets(
                    workbook
                )

            except Exception:

                continue

            for sheet in sheets:

                try:

                    df = read_sheet(
                        workbook,
                        sheet,
                    )

                except Exception:

                    continue

                state_rows = find_state_rows(
                    df,
                    code,
                )

                if not state_rows:
                    continue

                headers = make_headers(
                    df,
                    rows=12,
                )

                quarterly_columns = []
                annual_columns = []

                for col, header in (
                    headers.items()
                ):

                    text = header.lower()

                    if (
                        "quarter"
                        in text
                        and "change"
                        in text
                    ):

                        quarterly_columns.append(
                            col
                        )

                    if (
                        "annual"
                        in text
                        and "change"
                        in text
                    ):

                        annual_columns.append(
                            col
                        )

                sheet_text = (
                    " ".join(
                        headers.values()
                    ).lower()
                )

                if (
                    "house construction"
                    not in sheet_text
                    and "construction prices"
                    not in sheet_text
                ):

                    continue

                row = state_rows[0]

                quarterly = None
                annual = None

                for col in quarterly_columns:

                    value = number(
                        df.iloc[
                            row,
                            col,
                        ]
                    )

                    if value is not None:

                        quarterly = value

                        break

                for col in annual_columns:

                    value = number(
                        df.iloc[
                            row,
                            col,
                        ]
                    )

                    if value is not None:

                        annual = value

                        break

                if (
                    quarterly is None
                    and annual is None
                ):

                    continue

                result[code][
                    "construction_quarterly_change"
                ] = quarterly

                result[code][
                    "construction_annual_change"
                ] = annual

                result[code][
                    "construction_price_index"
                ] = annual

                result[code][
                    "sources"
                ][
                    "construction_price_index"
                ] = {
                    "workbook":
                        workbook.name,
                    "sheet":
                        sheet,
                    "row":
                        int(row),
                    "quarterly":
                        quarterly,
                    "annual":
                        annual,
                }

                break

    return result


# ============================================================
# BUILDING ACTIVITY
# ============================================================

def parse_building(
    files,
):

    result = {
        code: {
            "building_work_value": None,
            "sources": {},
        }
        for code in STATES
    }

    for code in STATES:

        for workbook in files:

            if result[code][
                "building_work_value"
            ] is not None:

                break

            match = find_series(
                workbook,
                code,
                required=[
                    "value",
                    "building",
                    "work",
                ],
                optional=[
                    "done",
                    "sector",
                    "states",
                    "territories",
                ],
                forbidden=[
                    "commenced",
                ],
                preferred_sheets=[
                    "02",
                    "value",
                    "building",
                    "state",
                ],
            )

            if match:

                result[code][
                    "building_work_value"
                ] = match["value"]

                result[code][
                    "sources"
                ][
                    "building_work_value"
                ] = match

    return result


# ============================================================
# COMBINE ABS
# ============================================================

def combine_abs(
    dwellings,
    earnings,
    ppi,
    building,
):

    result = {}

    for code, name in STATES.items():

        d = dwellings.get(
            code,
            {},
        )

        e = earnings.get(
            code,
            {},
        )

        p = ppi.get(
            code,
            {},
        )

        b = building.get(
            code,
            {},
        )

        result[code] = {

            "state": name,

            "total_dwelling_value":
                d.get(
                    "total_dwelling_value"
                ),

            "mean_dwelling_price":
                d.get(
                    "mean_dwelling_price"
                ),

            "dwelling_count":
                d.get(
                    "dwelling_count"
                ),

            "weekly_earnings":
                e.get(
                    "weekly_earnings"
                ),

            "construction_price_index":
                p.get(
                    "construction_price_index"
                ),

            "construction_quarterly_change":
                p.get(
                    "construction_quarterly_change"
                ),

            "construction_annual_change":
                p.get(
                    "construction_annual_change"
                ),

            "building_work_value":
                b.get(
                    "building_work_value"
                ),

            "sources": {
                "dwellings":
                    d.get(
                        "sources",
                        {},
                    ),

                "earnings":
                    e.get(
                        "sources",
                        {},
                    ),

                "ppi":
                    p.get(
                        "sources",
                        {},
                    ),

                "building":
                    b.get(
                        "sources",
                        {},
                    ),
            },
        }

    return result


# ============================================================
# GOLDEN RATIO
# ============================================================

def golden_weights(
    n=16,
):

    weights = []

    for k in range(n):

        distance = min(
            k,
            n - k,
        )

        weights.append(
            PHI ** (-distance)
        )

    total = sum(
        weights
    )

    return [
        value / total
        for value in weights
    ]


def economic_sector_vector(
    values,
    sectors=16,
):

    finite_values = [
        float(value)
        for value in values
        if finite(value)
    ]

    if not finite_values:

        return [
            1.0 / sectors
            for _ in range(sectors)
        ]

    low = min(
        finite_values
    )

    high = max(
        finite_values
    )

    if high == low:

        normalized = [
            0.5
            for _ in values
        ]

    else:

        normalized = [
            (
                (
                    float(value)
                    - low
                )
                / (
                    high
                    - low
                )
            )
            if finite(value)
            else 0.5
            for value in values
        ]

    vector = [
        0.0
        for _ in range(sectors)
    ]

    for i, value in enumerate(
        normalized
    ):

        a = i % sectors

        b = (
            sectors
            - 1
            - i
        ) % sectors

        vector[a] += value
        vector[b] += value

    total = sum(
        vector
    )

    if total <= 0:

        return [
            1.0 / sectors
            for _ in range(sectors)
        ]

    return [
        value / total
        for value in vector
    ]


def similarity(
    observed,
    target,
):

    if len(observed) != len(target):

        return 0.0

    return sum(
        math.sqrt(
            max(a, 0.0)
            * max(b, 0.0)
        )
        for a, b in zip(
            observed,
            target,
        )
    )


# ============================================================
# NORMALIZATION
# ============================================================

def minmax(
    rows,
    field,
):

    values = [
        float(row[field])
        for row in rows
        if finite(
            row.get(field)
        )
    ]

    if not values:

        return {
            row["state"]: 0.5
            for row in rows
        }

    low = min(
        values
    )

    high = max(
        values
    )

    if high == low:

        return {
            row["state"]: 0.5
            for row in rows
        }

    output = {}

    for row in rows:

        value = row.get(
            field
        )

        if not finite(value):

            output[
                row["state"]
            ] = 0.5

        else:

            output[
                row["state"]
            ] = (
                float(value)
                - low
            ) / (
                high
                - low
            )

    return output


# ============================================================
# REGIONAL PREDICTION
# ============================================================

def calculate_scores(
    abs_data,
):

    rows = []

    for code, name in STATES.items():

        data = abs_data.get(
            code,
            {},
        )

        rows.append({

            "state":
                code,

            "name":
                name,

            "total_dwelling_value":
                data.get(
                    "total_dwelling_value"
                ),

            "mean_dwelling_price":
                data.get(
                    "mean_dwelling_price"
                ),

            "dwelling_count":
                data.get(
                    "dwelling_count"
                ),

            "weekly_earnings":
                data.get(
                    "weekly_earnings"
                ),

            "construction_price_index":
                data.get(
                    "construction_price_index"
                ),

            "construction_quarterly_change":
                data.get(
                    "construction_quarterly_change"
                ),

            "construction_annual_change":
                data.get(
                    "construction_annual_change"
                ),

            "building_work_value":
                data.get(
                    "building_work_value"
                ),
        })

    fields = [
        "total_dwelling_value",
        "mean_dwelling_price",
        "dwelling_count",
        "weekly_earnings",
        "construction_price_index",
        "building_work_value",
    ]

    normalized = {
        field:
            minmax(
                rows,
                field,
            )
        for field in fields
    }

    golden = golden_weights(
        16
    )

    for row in rows:

        feature_values = [
            normalized[field][
                row["state"]
            ]
            for field in fields
        ]

        mean_feature = (
            sum(feature_values)
            / len(feature_values)
        )

        denominator = sum(
            PHI ** (-j)
            for j in range(
                len(feature_values)
            )
        )

        weighted_feature = sum(
            value
            * PHI ** (-j)
            for j, value
            in enumerate(
                feature_values
            )
        ) / denominator

        state_vector_seed = (
            feature_values
            + [
                mean_feature,
                weighted_feature,
            ]
        )

        vector = (
            economic_sector_vector(
                state_vector_seed,
                16,
            )
        )

        row[
            "golden_alignment"
        ] = similarity(
            vector,
            golden,
        )

        row[
            "economic_sector_vector"
        ] = vector

    score_weights = {

        "mean_dwelling_price":
            0.25,

        "weekly_earnings":
            0.20,

        "building_work_value":
            0.15,

        "total_dwelling_value":
            0.10,

        "dwelling_count":
            0.10,

        "construction_price_index":
            0.10,

        "golden_alignment":
            0.10,
    }

    for row in rows:

        score = 0.0

        for field, weight in (
            score_weights.items()
        ):

            if field == (
                "golden_alignment"
            ):

                value = row.get(
                    field,
                    0.5,
                )

            else:

                value = normalized[
                    field
                ][
                    row["state"]
                ]

            score += (
                weight
                * value
            )

        row[
            "regional_score"
        ] = score

        row[
            "normalized_features"
        ] = {

            field:

                (
                    row.get(field)
                    if field ==
                    "golden_alignment"

                    else

                    normalized[
                        field
                    ][
                        row["state"]
                    ]
                )

            for field
            in score_weights
        }

    rows.sort(
        key=lambda row:
            row[
                "regional_score"
            ],
        reverse=True,
    )

    # Add ranking after sorting.
    for rank, row in enumerate(
        rows,
        1,
    ):

        row[
            "rank"
        ] = rank

    return rows


# ============================================================
# PREDICTED VALUE TILE HEAT MAP
# ============================================================

STATE_TILE_LAYOUT = {

    # row, column

    "WA":
        (0, 0),

    "NT":
        (0, 1),

    "QLD":
        (0, 2),

    "SA":
        (1, 0),

    "NSW":
        (1, 2),

    "VIC":
        (2, 1),

    "ACT":
        (2, 2),

    "TAS":
        (3, 1),
}


def generate_predicted_value_heatmap(
    predictions,
    output_path=None,
):

    """
    Generate a tile-style Australian
    predicted-value heat map.

    One tile:
        one state/territory.

    Tile colour:
        regional_score.

    Tile text:
        state abbreviation
        state name
        regional score
        rank

    The layout is a simplified visual
    state arrangement. It is NOT an
    OSM/geographic tile map.
    """

    if output_path is None:

        output_path = (
            OUTPUT_DIR
            / "predicted_value_heatmap.png"
        )

    scores = {}
    names = {}
    ranks = {}

    for row in predictions:

        code = row.get(
            "state"
        )

        score = row.get(
            "regional_score"
        )

        if code not in (
            STATE_TILE_LAYOUT
        ):

            continue

        if not finite(score):

            continue

        scores[code] = float(
            score
        )

        names[code] = row.get(
            "name",
            STATES.get(
                code,
                code,
            ),
        )

        ranks[code] = row.get(
            "rank",
            "?",
        )

    if not scores:

        raise ValueError(
            "No valid regional_score "
            "values were found."
        )

    minimum = min(
        scores.values()
    )

    maximum = max(
        scores.values()
    )

    if maximum == minimum:

        normalized = {
            code: 0.5
            for code in scores
        }

    else:

        normalized = {

            code:

                (
                    value
                    - minimum
                )
                / (
                    maximum
                    - minimum
                )

            for code, value
            in scores.items()
        }

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.set_xlim(
        -0.25,
        3.25,
    )

    ax.set_ylim(
        4.10,
        -0.25,
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )

    cmap = plt.get_cmap(
        "YlOrRd"
    )

    for code, (
        row,
        col,
    ) in STATE_TILE_LAYOUT.items():

        if code not in scores:

            continue

        x = col
        y = row

        score = scores[
            code
        ]

        normalized_value = (
            normalized[
                code
            ]
        )

        face_color = cmap(
            normalized_value
        )

        tile = Rectangle(

            (
                x,
                y,
            ),

            1.0,
            1.0,

            facecolor=
                face_color,

            edgecolor=
                "black",

            linewidth=
                2.0,
        )

        ax.add_patch(
            tile
        )

        r, g, b, _ = (
            face_color
        )

        luminance = (
            0.299 * r
            + 0.587 * g
            + 0.114 * b
        )

        text_color = (
            "black"
            if luminance > 0.60
            else "white"
        )

        # Rank
        ax.text(

            x + 0.10,
            y + 0.12,

            f"#{ranks[code]}",

            ha="left",
            va="top",

            fontsize=10,

            fontweight="bold",

            color=text_color,
        )

        # State abbreviation
        ax.text(

            x + 0.5,
            y + 0.37,

            code,

            ha="center",
            va="center",

            fontsize=23,

            fontweight="bold",

            color=text_color,
        )

        # Predicted regional score
        ax.text(

            x + 0.5,
            y + 0.60,

            f"{score:.4f}",

            ha="center",
            va="center",

            fontsize=13,

            color=text_color,
        )

        # Full state name
        ax.text(

            x + 0.5,
            y + 0.84,

            names[code],

            ha="center",
            va="center",

            fontsize=7.5,

            color=text_color,
        )

    ax.set_title(

        "Predicted Regional Value — "
        "ABS Economic Model",

        fontsize=18,

        fontweight="bold",

        pad=20,
    )

    sm = plt.cm.ScalarMappable(

        cmap=cmap,

        norm=plt.Normalize(

            vmin=minimum,

            vmax=maximum,
        ),
    )

    sm.set_array([])

    colorbar = fig.colorbar(

        sm,

        ax=ax,

        fraction=0.035,

        pad=0.025,
    )

    colorbar.set_label(

        "Predicted regional score",

        fontsize=11,
    )

    ax.text(

        1.5,
        3.88,

        (
            f"Low = {minimum:.4f}    "
            f"High = {maximum:.4f}"
        ),

        ha="center",

        va="center",

        fontsize=10,
    )

    fig.savefig(

        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved heat map: "
        f"{output_path}"
    )

    return output_path


# ============================================================
# REPORT
# ============================================================

def make_report(
    abs_data,
    predictions,
):

    lines = [

        "ABS-ONLY GOLDEN REGIONAL SOLVER",

        "=" * 70,

        f"Golden ratio phi = "
        f"{PHI:.12f}",

        "",

        "REGIONAL RANKING",

        "-" * 70,
    ]

    for rank, row in enumerate(
        predictions,
        1,
    ):

        lines.append(

            f"{rank:2d}. "
            f"{row['state']} "
            f"{row['name']} "
            f"score="
            f"{row['regional_score']:.6f} "
            f"golden="
            f"{row['golden_alignment']:.6f}"
        )

    lines += [

        "",

        "RAW ABS DATA",

        "-" * 70,
    ]

    for code, data in (
        abs_data.items()
    ):

        lines += [

            "",

            f"{code} - "
            f"{data['state']}",

            f"  total dwelling value: "
            f"{data['total_dwelling_value']}",

            f"  mean dwelling price: "
            f"{data['mean_dwelling_price']}",

            f"  dwelling count: "
            f"{data['dwelling_count']}",

            f"  weekly earnings: "
            f"{data['weekly_earnings']}",

            f"  construction quarterly change: "
            f"{data['construction_quarterly_change']}",

            f"  construction annual change: "
            f"{data['construction_annual_change']}",

            f"  building work value: "
            f"{data['building_work_value']}",
        ]

    path = (
        OUTPUT_DIR
        / "report.txt"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# INSPECTION
# ============================================================

def inspect_files(
    files,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ABS WORKBOOK INSPECTION"
    )

    print(
        "=" * 70
    )

    for dataset, paths in (
        files.items()
    ):

        print(
            f"\n[{dataset}]"
        )

        for path in paths:

            print(
                f"\n  {path.name}"
            )

            try:

                sheets = (
                    workbook_sheets(
                        path
                    )
                )

                for sheet in sheets:

                    df = read_sheet(
                        path,
                        sheet,
                    )

                    print(

                        f"    {sheet}: "

                        f"{df.shape[0]} x "

                        f"{df.shape[1]}"
                    )

            except Exception as exc:

                print(
                    f"    inspection failed: "
                    f"{exc}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "ABS-only regional economic "
            "golden-ratio solver with "
            "predicted-value tile heat map"
        )
    )

    parser.add_argument(

        "--force",

        action="store_true",

        help=(
            "Redownload ABS XLSX files."
        ),
    )

    parser.add_argument(

        "--cached",

        action="store_true",

        help=(
            "Use cached XLSX files only."
        ),
    )

    parser.add_argument(

        "--inspect",

        action="store_true",

        help=(
            "Inspect downloaded workbook sheets."
        ),
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ABS-ONLY GOLDEN REGIONAL SOLVER"
    )

    print(
        "=" * 70
    )

    print(
        "\nNo OSM."
    )

    print(
        "No Overpass."
    )

    print(
        "No roads."
    )

    print(
        "No road geometry."
    )

    print(
        "No geographic tile downloads."
    )

    print(
        "Official ABS XLSX data only."
    )

    # --------------------------------------------------------
    # Download ABS
    # --------------------------------------------------------

    downloader = ABSDownloader(
        force=args.force
    )

    if args.cached:

        print(
            "\nUsing cached ABS workbooks."
        )

        all_cached = list(
            DOWNLOAD_DIR.glob(
                "*.xlsx"
            )
        )

        files = {
            dataset:
                all_cached

            for dataset
            in ABS_RELEASES
        }

    else:

        try:

            files = (
                downloader.download_all()
            )

        except requests.RequestException as exc:

            print(
                "\nABS network failure:"
            )

            print(
                exc
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Inspect
    # --------------------------------------------------------

    if args.inspect:

        inspect_files(
            files
        )

    # --------------------------------------------------------
    # Parse ABS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PARSING OFFICIAL ABS XLSX DATA"
    )

    print(
        "=" * 70
    )

    dwellings = parse_dwellings(
        files.get(
            "dwellings",
            [],
        )
    )

    earnings = parse_earnings(
        files.get(
            "earnings",
            [],
        )
    )

    ppi = parse_ppi(
        files.get(
            "ppi",
            [],
        )
    )

    building = parse_building(
        files.get(
            "building",
            [],
        )
    )

    abs_data = combine_abs(

        dwellings,

        earnings,

        ppi,

        building,
    )

    save_json(

        "abs_data.json",

        abs_data,
    )

    # --------------------------------------------------------
    # ABS check
    # --------------------------------------------------------

    print(
        "\nABS DATA CHECK"
    )

    print(
        "-" * 70
    )

    for code, data in (
        abs_data.items()
    ):

        print(

            f"{code}: "

            f"price="
            f"{data['mean_dwelling_price']} | "

            f"earnings="
            f"{data['weekly_earnings']} | "

            f"building="
            f"{data['building_work_value']} | "

            f"construction="
            f"{data['construction_annual_change']}"
        )

    # --------------------------------------------------------
    # Calculate regional predictions
    # --------------------------------------------------------

    predictions = (
        calculate_scores(
            abs_data
        )
    )

    save_json(

        "predictions.json",

        predictions,
    )

    save_json(

        "golden_baseline.json",

        {
            "phi":
                PHI,

            "sector_count":
                16,

            "weights":
                golden_weights(
                    16
                ),
        },
    )

    # --------------------------------------------------------
    # Predicted value heat map
    # --------------------------------------------------------

    generate_predicted_value_heatmap(

        predictions,

        OUTPUT_DIR
        / "predicted_value_heatmap.png",
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_rows = []

    for row in predictions:

        output = {}

        for key, value in (
            row.items()
        ):

            if key in {

                "economic_sector_vector",

                "normalized_features",
            }:

                continue

            output[key] = value

        csv_rows.append(
            output
        )

    csv_path = (
        OUTPUT_DIR
        / "predictions.csv"
    )

    pd.DataFrame(
        csv_rows
    ).to_csv(

        csv_path,

        index=False,
    )

    print(
        f"Saved: {csv_path}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    make_report(

        abs_data,

        predictions,
    )

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL REGIONAL RESULTS"
    )

    print(
        "=" * 70
    )

    for rank, row in enumerate(
        predictions,
        1,
    ):

        print(

            f"{rank:2d}. "

            f"{row['state']:3s} "

            f"{row['name']:<30s} "

            f"score="
            f"{row['regional_score']:.6f} "

            f"golden="
            f"{row['golden_alignment']:.6f}"
        )

    print(
        "\nComplete."
    )

    print(
        "Heat map:"
    )

    print(
        (
            OUTPUT_DIR
            / "predicted_value_heatmap.png"
        ).resolve()
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":

    main()
