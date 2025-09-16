import json
from datetime import datetime, timedelta


def main():
    with open("output/generated_tweets.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    base_time = datetime(2025, 8, 1, 12, 0)
    delta = timedelta(hours=5, minutes=30)

    output = []
    tweet_id = 1

    for item in data:
        for key in ["tweet1", "tweet2"]:
            timestamp = base_time + (tweet_id * delta)
            formatted_time = timestamp.isoformat()

            output.append({
                "id": tweet_id,
                "text": item[key],
                "timestamp": formatted_time
            })

            tweet_id += 1

    username = 'clown'
    with open(f"output/extracted_tweets_{username}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("✅ Converted tweets written to output.json")


if __name__ == "__main__":
    main()
