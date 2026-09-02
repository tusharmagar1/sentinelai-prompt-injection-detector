def calculate_risk(injection_probability):
    """
    Convert model probability into a risk score.
    """

    risk_score = round(
        injection_probability * 100,
        2
    )

    if injection_probability >= 0.80:

        action = "BLOCK"

    elif injection_probability >= 0.50:

        action = "SUSPICIOUS"

    else:

        action = "ALLOW"

    return {
        "risk_score": risk_score,
        "action": action
    }


if __name__ == "__main__":

    test_scores = [
        0.05,
        0.32,
        0.51,
        0.73,
        0.86,
        0.97
    ]

    for score in test_scores:

        result = calculate_risk(score)

        print(
            f"\nProbability: {score:.2f}"
        )

        print(
            f"Risk Score: {result['risk_score']}"
        )

        print(
            f"Action: {result['action']}"
        )
