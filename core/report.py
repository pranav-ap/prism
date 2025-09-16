from prefect import task


def generate_contradiction_summary(contradictions):
    with open(f"D:/code/prism/output/contradictions_report.txt", "w", encoding="utf-8") as f:
        f.write("\n Contradiction Report")
        f.write("\n ====================")
        f.write(f"\n Found {len(contradictions)} contradictions")

        for i, pair in enumerate(contradictions, 1):
            f.write(repr(pair))


@task
def generate_report(contradictions):
    print('Generating contradiction report...')

    generate_contradiction_summary(contradictions)
