def generate_report(contradictions):
    print('Generating contradiction report...')

    with open("output/contradictions_report.txt", "w", encoding="utf-8") as f:
        f.write("Contradiction Report\n")
        f.write("====================\n\n")
        f.write(f"Total contradictions found: {len(contradictions)}\n\n")

        for i, pair in enumerate(contradictions, 1):
            f.write(f"{i}. Similarity Score: {pair.similarity_score:.3f}, Contradiction Score: {pair.contradiction_score:.3f}\n")
            f.write(f"   - Reason   : {pair.contradiction_reason}\n")
            f.write(f"   - Tweet 1  : {pair.tweet1.text.strip()}\n")
            f.write(f"   - Tweet 2  : {pair.tweet2.text.strip()}\n\n")
