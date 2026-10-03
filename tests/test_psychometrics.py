"""Check KR-20 against independently computed dichotomous Cronbach alpha."""
from synbiobench.core.psychometrics import kr20


def test_kr20_matches_alpha_with_consistent_variance():
    rows = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0]]
    models = [str(i) for i in range(len(rows))]
    items = [{"item_id": str(j)} for j in range(3)]
    matrix = {m: {str(j): value for j, value in enumerate(row)}
              for m, row in zip(models, rows)}
    means = [sum(row[j] for row in rows) / len(rows) for j in range(3)]
    item_variances = [sum((row[j] - means[j]) ** 2 for row in rows) / len(rows)
                      for j in range(3)]
    totals = [sum(row) for row in rows]
    mean_total = sum(totals) / len(totals)
    total_variance = sum((v - mean_total) ** 2 for v in totals) / len(totals)
    expected = 3 / 2 * (1 - sum(item_variances) / total_variance)
    actual, count = kr20(items, models, matrix)
    assert count == 3
    assert abs(actual - expected) < 1e-12
