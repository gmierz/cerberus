# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Search for regression in a histogram dump directory produced by the
# node exporter.

import json
import numpy
import cv2
import sys
import os
import logging
import json
import os.path
import argparse
import requests

from google.cloud import bigquery
from time import mktime, strptime
from datetime import datetime, timedelta

histograms = None
args = None

# Whether to plot the found regressions, and what filename to save them with if plotting
OUTPUT_PLOTS, PLOT_FILENAME = (
    True,
    "plot-{histogram_name}-{date}.png",
)
# Path of JSON file containing a list of all found regressions
REGRESSION_FILENAME = "regressions.json"
# Path to JSON file containing histogram definitions map
HISTOGRAM_DB = "Histograms.json"

if OUTPUT_PLOTS:
    import matplotlib.pyplot as plt
    import pylab

def has_not_enough_data(hist):
    return numpy.sum(hist) < 1000 or numpy.max(hist) < 1000


def normalize(hist):
    """Returns a copy of the given histogram scaled such
    that its sum is 1, or 0 if this is not possible"""
    hist = hist.astype("float32")
    total = numpy.sum(hist)
    if total == 0:
        return hist
    return hist / total


def bat_distance(hist, ref):
    """Compute the Bhattacharyya distance between two distributions, using OpenCV"""
    return cv2.compareHist(hist, ref, 3)


def compare_range(series, idx, range, nr_ref_days):
    """Compare histogram at index `idx` to all of the histograms
    at indices in `range` in `series`."""
    assert iter(series) and all(len(item) == 2 for item in series), (
        "`series` must be of the form "
        "`[(DATETIME_1, HISTOGRAM_1), ..., (DATETIME_N, HISTOGRAM_N)]`"
    )
    assert 0 <= idx < len(series) and idx % 1 == 0, "`index` must be a valid index"
    assert iter(range) and all(
        0 <= i < len(series) and i % 1 == 0 for i in range
    ), "`range` must be an iterable of valid indices"
    dt, hist = series[idx]
    hist = normalize(hist)
    distances = []
    logging.debug("Comparing " + dt)

    for jdx in range:
        ref_dt, ref_hist = series[jdx]
        logging.debug("To " + ref_dt)

        if has_not_enough_data(ref_hist):
            logging.debug("Reference histogram has not enough data")
            ref_hist = normalize(ref_hist)
            continue
        ref_hist = normalize(ref_hist)

        distances.append(bat_distance(hist, ref_hist))

    # There are histograms that have enough data to be compared
    if len(distances):
        logging.debug("Bhattacharyya distance: " + str(distances[-1]))
        logging.debug(
            "Standard deviation of the distances: " + str(numpy.std(distances))
        )

    # The last compared histograms are significantly different,
    # and the differences have a very narrow spread
    if (
        len(distances) > nr_ref_days / 2
        and distances[-1] > 0.12
        and numpy.std(distances) <= 0.01
    ):
        logging.debug("*** Suspicious difference found ***")
        return (hist, ref_hist)  # Produce the last compared histogram pair
    else:
        logging.debug("No suspicious difference found")
        return (None, None)


def get_raw_histograms(comparisons):
    hist = None
    ref_hist = None

    for h, r in comparisons:
        if h is not None:
            return (h, r)

    assert False


def compare_histogram(series, histogram, buckets, nr_ref_days=3, nr_future_days=2):
    """Compare the past `nr_future_days` days worth of histograms
    in `series` to the past `nr_ref_days` days worth of histograms
    in `series`, returning a list of found regressions."""
    regressions = []
    series_items = sorted(
        series.items(), key=lambda x: x[0]
    )  # sorted list of pairs, each of the form (DATETIME, HISTOGRAM_FOR_THAT_DATE)

    for i, entry in enumerate(
        series_items[: -nr_future_days if nr_future_days else None]
    ):
        dt, hist = entry

        logging.debug("======================")
        logging.debug("Analyzing " + dt)

        if has_not_enough_data(
            hist
        ):  # Histogram doesn't have enough submissions to get a meaningful result
            logging.debug("Histogram does not have enough data")
            continue

        comparisons = []
        ref_range = range(max(i - nr_ref_days, 0), i)

        for j in range(i, min(i + nr_future_days + 1, len(series_items))):
            comparisons.append(compare_range(series_items, j, ref_range, nr_ref_days))

        # Check that a difference was detected against all other histograms
        if all(x is not (None, None) for x in comparisons):
            logging.debug(
                "Regression found for " + histogram + dt
            )
            regressions.append(
                (dt, histogram, buckets, get_raw_histograms(comparisons))
            )
            if (
                OUTPUT_PLOTS and len(buckets) < 300
            ):  # There are histograms with several hundred buckets
                # that cause plotting to fail since the resulting image is just too large
                file_name = PLOT_FILENAME.format(
                    histogram_name=histogram, date=dt
                )
                plot(file_name, histogram, buckets, get_raw_histograms(comparisons))
    print(regressions)
    return regressions


def process_file(filename):
    logging.debug("Processing " + filename)
    series = {}
    buckets = []

    regressions = []
    with open(
        filename
    ) as f:  # one of the JSON files of the form `histograms/MEASURE_NAME.json`
        measures = json.load(f)
        for (
            measure
        ) in (
            measures
        ):  # a measure in this context is a histogram and a date that the histogram applies to
            # determine the date of the entry
            assert "date" in measure, "Missing date in measure"
            conv = strptime(measure["date"][:10], "%Y-%m-%d")
            measure_date = datetime.fromtimestamp(mktime(conv))

            # add the histogram values to the corresponding entry in the time series
            if measure_date in series:
                if series[measure_date].shape != numpy.array(measure["values"]).shape:
                    print(
                        "Shape mismatch in {}: {} cannot be added to {}".format(
                            filename, series[measure_date], measure["values"]
                        )
                    )
                    continue
                series[measure_date] += numpy.array(measure["values"])
            else:
                series[measure_date] = numpy.array(measure["values"])
            buckets = measure["buckets"]

        measure_name, _ = os.path.splitext(
            os.path.basename(filename)
        )  # the measure name is the filename without the extension

    if series:
        # check that the series is valid
        reference_date, expected_bucket_count = None, None
        for date, histogram in series.items():
            if expected_bucket_count is None:
                reference_date, expected_bucket_count = date, len(histogram)
            elif len(histogram) != expected_bucket_count:
                # Can't we interpolate a few missing buckets?
                logging.warn(
                    "BUCKET COUNT MISMATCH - IGNORING HISTOGRAM "
                    "{} ({} has {} buckets, while {} has {} buckets)".format(
                        measure_name,
                        reference_date,
                        expected_bucket_count,
                        date,
                        len(histogram),
                    )
                )
                return []

    regressions += compare_histogram(series, measure_name, buckets)
    return regressions


def process_file2(json_obj):
    series = {}
    buckets = []

    all_buckets = set()
    measures = {}
    for entry in json_obj:
        if entry["segment"].lower() not in ("windows",): continue
        conv = strptime(entry["build_id"][:8], "%Y%m%d")
        if int(conv.tm_year) < 2023: continue
        measure_date = entry["build_id"]
        measures.setdefault(measure_date, {}).setdefault(
            "vals", []
        ).append((entry["bucket"], entry["counts"]))
        all_buckets |= set([entry["bucket"]])

    sorted_buckets = sorted(list(all_buckets))
    all_buckets = {
        b: ""
        for b in sorted_buckets
    }
    measures_list = []
    for date in sorted(list(measures.keys())):
        sorted_vals = sorted(measures[date]["vals"], key=lambda x: x[0])

        vals = []

        for bucket in sorted_buckets:
            for sbucket, val in sorted_vals:
                if sbucket == bucket:
                    vals.append(val)
                    break
            else:
                vals.append(0)

        print(vals)
        print(len(all_buckets))
        print(len(vals))

        measures_entry = {
            "measure": "JS_EXECUTION_MS",
            "buckets": sorted_buckets,
            "values": vals,
            "date": date,
            "count": sum([v[1] for v in sorted_vals])
        }
        measures_list.append(measures_entry)

    # Aggregate 7 future days into current date
    for i, measure in enumerate(measures_list):
        next_seven_inclusive = [m["values"] for m in measures_list[i:i+7]]
        measure["values"] = list(map(lambda x: sum(x), zip(*next_seven_inclusive)))

    regressions = []
    for measure in measures_list:
        # assert "build_id" in measure, "Missing mozilla-central commit info"
        # a measure in this context is a histogram and a date that the histogram applies to
        # determine the date of the entry
        # assert "date" in measure, "Missing date in measure"
        # conv = strptime(measure["date"][:8], "%Y%m%d")
        # measure_date = datetime.fromtimestamp(mktime(conv))
        measure_date = measure["date"]

        # add the histogram values to the corresponding entry in the time series
        if measure_date in series:
            if series[measure_date].shape != numpy.array(measure["values"]).shape:
                print(
                    "Shape mismatch in {}: {} cannot be added to {}".format(
                        filename, series[measure_date], measure["values"]
                    )
                )
                continue
            series[measure_date] += numpy.array(measure["values"])
        else:
            series[measure_date] = numpy.array(measure["values"])
        buckets = measure["buckets"]

    measure_name = "perf_first_contentful_paint_from_responsestart_ms"

    if series:
        # check that the series is valid
        reference_date, expected_bucket_count = None, None
        for date, histogram in series.items():
            if expected_bucket_count is None:
                reference_date, expected_bucket_count = date, len(histogram)
            elif len(histogram) != expected_bucket_count:
                logging.warn(
                    "BUCKET COUNT MISMATCH - IGNORING HISTOGRAM "
                    "{} ({} has {} buckets, while {} has {} buckets)".format(
                        measure_name,
                        reference_date,
                        expected_bucket_count,
                        date,
                        len(histogram),
                    )
                )
                return []

    regressions += compare_histogram(series, measure_name, buckets)
    return regressions


def plot(file_name, histogram_name, buckets, raw_histograms):
    hist, ref_hist = raw_histograms

    fig = pylab.figure(figsize=(len(buckets) / 3, 10))
    pylab.plot(hist, label="Regression", color="red")
    pylab.plot(ref_hist, label="Reference", color="blue")
    pylab.legend(shadow=True)
    pylab.title(histogram_name)
    pylab.xlabel("Bin")
    pylab.ylabel("Normalized Weight")

    pylab.xticks(range(0, len(buckets)))
    locs, labels = pylab.xticks()
    pylab.xticks(locs, buckets, rotation=45)

    pylab.savefig(file_name, bbox_inches="tight")
    pylab.close(fig)


def main():
    regressions = []

    # This is the same Histograms.json as the one in mozilla-central
    # It should always be the latest possible version when running
    with open("Histograms.json") as f:
        histograms = json.load(f)

    with open("Scalars.json") as f:
        scalars = json.load(f)

    probes = dict()
    probes.update(histograms)
    probes.update(scalars)

    # Build a set of data from the past 12 months to test with, then run the detection
    # on all of that data.
    month = "12"
    reset = False
    cache_file = f"temp-merged.json"
    if not os.path.exists(cache_file) or reset:
        print("Rerunning query...")
        client = bigquery.Client()
        job = client.query("""
            WITH raw as (
                SELECT
                    normalized_os as segment,
                    application.build_id as build_id,
                    JSON_EXTRACT(payload.processes.content.histograms.perf_first_contentful_paint_from_responsestart_ms, "$.values") as hist
                FROM
                    `moz-fx-data-shared-prod.telemetry.main`
                WHERE
                    -- submission_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                    submission_timestamp between TIMESTAMP('2023-12-01') and TIMESTAMP('2023-12-31')
                    AND normalized_channel = "nightly"
                    AND normalized_app_name = "Firefox"
                    AND payload.processes.content.histograms.perf_first_contentful_paint_from_responsestart_ms is not null
            ),
            bucketCounts as (
                SELECT
                    segment,
                    build_id,
                    SAFE_CAST(key AS float64) as bucket,
                    INT64(PARSE_JSON(hist)[key]) as count
                FROM
                    raw,
                    UNNEST(bqutil.fn.json_extract_keys(hist)) as key
            ),
            histogram as (
                SELECT
                    segment,
                    build_id,
                    bucket,
                    SUM(count) as counts
                FROM
                    bucketCounts
                GROUP BY
                    segment, build_id, bucket
                ORDER BY
                    segment, build_id, bucket
            )
            SELECT
                segment,
                build_id,
                bucket,
                counts
            FROM
                histogram
            ORDER BY 
                segment, build_id, bucket
        """)
        df = job.to_dataframe()
        print(df)
        json_obj = json.loads(df.to_json(orient='records'))
        with open(cache_file, "w") as f:
            json.dump(json_obj, f)
    else:
        with open(cache_file) as f:
            json_obj = json.load(f)

    regressions = process_file2(json_obj)
    firefoxrelease_info = None
    bug_list = None
    with open("firefoxreleases.json") as f:
        firefoxrelease_info = json.load(f)
    with open("sp3-bugs.json") as f:
        bug_list = json.load(f)
    with open("sp3-bugs-jeff.json") as f:
        bug_list.extend(json.load(f))
    for regression in regressions:
        build_id = regression[0]
        for release in firefoxrelease_info["builds"]:
            if release["buildid"] == build_id:
                print(f"regression found for {release['node']} with build id {build_id}")
                print(
                    f"https://treeherder.mozilla.org/jobs?repo=mozilla-central&"
                    f"revision={release['node']}"
                )

                resp = requests.get(
                    f"https://hg.mozilla.org/mozilla-central/json-log/{release['node']}"
                )
                json_resp = resp.json()
                for changeset in json_resp["changesets"]:
                    found_bugs = []
                    for bug in bug_list:
                        if str(bug) in changeset["desc"]:
                            found_bugs.append(bug)
                    for bug in found_bugs:
                        print(
                            f"Found sp3 bug in commit: "
                            f"https://bugzilla.mozilla.org/show_bug.cgi?id={bug}"
                        )

                print("")
                break

    raise Exception()

    # Load past regressions
    past_regressions = {}
    try:
        with open(REGRESSION_FILENAME) as f:
            past_regressions = json.load(f)
    except:
        pass

    # Print new regressions
    for regression in sorted(regressions, key=lambda x: x[0]):
        regression_date, histogram, buckets, raw_histograms = regression
        regression_timestamp = regression_date.isoformat()[:10]

        if (
            regression_timestamp in past_regressions
            and histogram in past_regressions[regression_timestamp]
        ):
            print("Regression found for " + histogram + ", " + regression_timestamp)
        else:
            print(
                "Regression found for "
                + histogram
                + ", "
                + regression_timestamp
                + " [new]"
            )

            if not regression_timestamp in past_regressions:
                past_regressions[regression_timestamp] = {}

            descriptor = past_regressions[regression_timestamp][histogram] = {}
            descriptor["buckets"] = buckets
            descriptor["regression"] = raw_histograms[0].tolist()
            descriptor["reference"] = raw_histograms[1].tolist()

            name = histogram
            if histogram.startswith("STARTUP"):
                name = histogram[8:]

            probe_def = probes.get(name, {})
            descriptor["description"] = probe_def.get("description", "")
            descriptor["alert_emails"] = probe_def.get(
                "alert_emails", probe_def.get("notification_emails", "")
            )

    # Store regressions found
    with open(REGRESSION_FILENAME, "w") as f:
        json.dump(past_regressions, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telemetry Regression Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    logging.getLogger().setLevel(logging.DEBUG)
    args = parser.parse_args()

    main()
