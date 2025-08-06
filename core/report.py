def generate_report(contradictions):
    print('Generating contradiction report...')

    with open("../output/contradictions_report.txt", "w", encoding="utf-8") as f:
        f.write("\n Contradiction Report")
        f.write("\n ====================")
        f.write(f"\n Found {len(contradictions)} contradictions")

        for i, pair in enumerate(contradictions, 1):
            f.write(pair)
