from typing import List

import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import os
from prefect import task

from core.data_source import TweetPair


def plot_timeline(contradictions: List[TweetPair], output_dir):
    timeline = defaultdict(int)

    for pair in contradictions:
        ts1 = pair.tweet1.timestamp
        ts2 = pair.tweet2.timestamp

        t = min(
            datetime.fromisoformat(ts1),
            datetime.fromisoformat(ts2)
        )

        date_str = t.date().isoformat()
        timeline[date_str] += 1

    dates = sorted(timeline.keys())
    values = [timeline[d] for d in dates]

    plt.figure()
    plt.plot(dates, values, marker='o')
    plt.xticks(rotation=45)
    plt.title("Timeline of Contradictions")
    plt.xlabel("Date")
    plt.ylabel("Number of Contradictions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeline.png"))
    plt.close()


def generate_contradiction_summary(contradictions, output_dir):
    with open(f"{output_dir}/contradictions_report.txt", "w", encoding="utf-8") as f:
        f.write("\n Contradiction Report")
        f.write("\n ====================")
        f.write(f"\n Found {len(contradictions)} contradictions")

        for i, pair in enumerate(contradictions, 1):
            f.write(repr(pair))


@task
def generate_report(contradictions):
    print('Generating contradiction report...')

    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    generate_contradiction_summary(contradictions, output_dir)
    # plot_timeline(contradictions, output_dir)
