"""
Bayes' Theorem engine for IMDB keyword sentiment analysis (Part 2).

For every chosen keyword w, we compute:
    Prior:      P(Positive)
    Likelihood: P(w | Positive)
    Marginal:   P(w) = P(w|Positive) * P(Positive) + P(w|Negative) * P(Negative)
    Posterior:  P(Positive | w) = P(w|Positive) * P(Positive) / P(w)

Team decision: we only ever compute P(Positive | keyword), never P(Negative | keyword)
per the assignment rule. For the negative-indicating keywords this naturally comes out
low, which is itself the signal that they predict negative sentiment.
"""

from data_loader import download_dataset, load_reviews
from keywords import KEYWORD_JUSTIFICATION, NEGATIVE_KEYWORDS, POSITIVE_KEYWORDS

ALL_KEYWORDS = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS


def count_reviews(reviews):
    """Tally total/positive/negative counts and per-keyword hit counts with plain dicts."""
    positive_total = 0
    hits_positive = {kw: 0 for kw in ALL_KEYWORDS}
    hits_negative = {kw: 0 for kw in ALL_KEYWORDS}

    for tokens, label in reviews:
        is_positive = label == "positive"
        if is_positive:
            positive_total += 1
        for kw in ALL_KEYWORDS:
            if kw in tokens:
                if is_positive:
                    hits_positive[kw] += 1
                else:
                    hits_negative[kw] += 1

    total = len(reviews)
    return {
        "total": total,
        "positive_total": positive_total,
        "negative_total": total - positive_total,
        "hits_positive": hits_positive,
        "hits_negative": hits_negative,
    }


def bayes_table_for_keyword(keyword, stats):
    """Apply Bayes' theorem for a single keyword and return every intermediate term."""
    total = stats["total"]
    positive_total = stats["positive_total"]
    negative_total = stats["negative_total"]

    prior_positive = positive_total / total
    prior_negative = negative_total / total

    likelihood_positive = stats["hits_positive"][keyword] / positive_total
    likelihood_negative = stats["hits_negative"][keyword] / negative_total

    marginal = likelihood_positive * prior_positive + likelihood_negative * prior_negative
    posterior_positive = (likelihood_positive * prior_positive / marginal) if marginal > 0 else 0.0

    return {
        "keyword": keyword,
        "prior_positive": prior_positive,
        "likelihood_positive": likelihood_positive,
        "marginal": marginal,
        "posterior_positive": posterior_positive,
    }


def render_markdown_table(rows):
    lines = [
        "| Keyword | Prior P(Positive) | Likelihood P(kw\\|Positive) | Marginal P(kw) | Posterior P(Positive\\|kw) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['keyword']} | {r['prior_positive']:.4f} | {r['likelihood_positive']:.4f} "
            f"| {r['marginal']:.4f} | {r['posterior_positive']:.4f} |"
        )
    return "\n".join(lines)


def run():
    csv_path = download_dataset()
    print(f"Loading data from: {csv_path}")

    reviews = load_reviews(csv_path)
    print(f"Total reviews parsed: {len(reviews)}")

    stats = count_reviews(reviews)

    print("\n=== Positive-indicating keywords: P(Positive | keyword) ===")
    positive_rows = [bayes_table_for_keyword(kw, stats) for kw in POSITIVE_KEYWORDS]
    print(render_markdown_table(positive_rows))

    print("\n=== Negative-indicating keywords: P(Positive | keyword) ===")
    negative_rows = [bayes_table_for_keyword(kw, stats) for kw in NEGATIVE_KEYWORDS]
    print(render_markdown_table(negative_rows))

    print("\n=== Keyword selection justification ===")
    for kw in ALL_KEYWORDS:
        print(f"- {kw}: {KEYWORD_JUSTIFICATION[kw]}")

    return positive_rows, negative_rows


if __name__ == "__main__":
    run()
