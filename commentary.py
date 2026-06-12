import ollama


def generate_commentary(driver, avg_lap, best_lap, tyre_text, telemetry_text):
    prompt = f"""
    You are an expert Formula 1 race analyst similar to David Coulthard,
    Jolyon Palmer, Martin Brundle or an F1TV post-race analyst.

    Write a detailed post-race analysis of the driver's performance.

    IMPORTANT:

    - Speak ABOUT the driver, never TO the driver.
    - Never use phrases like "you", "we", "box", "push", or "try harder".
    - Never invent information not provided.
    - If corner-level information is unavailable, say so.
    - Explain what the telemetry, tyre data, pace profile and strategy suggest.
    - Discuss strengths and weaknesses.
    - Discuss tyre degradation.
    - Discuss race pace versus peak pace.
    - Discuss whether the strategy appears optimal.
    - Discuss possible alternative strategies.
    - Discuss where time may have been gained or lost.
    - Discuss risks that may have affected the result.
    - Write as if presenting a post-race analysis segment on F1TV.

    Driver:
    {driver}

    Average Lap Time:
    {avg_lap}

    Best Lap Time:
    {best_lap}

    Tyre Analysis:
    {tyre_text}

    Telemetry:
    {telemetry_text}

    Provide:

    1. Performance Overview
    2. Tyre & Strategy Analysis
    3. Telemetry Insights
    4. Key Takeaways

    Write 4-6 paragraphs.
    """

    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"""
        Local AI commentary unavailable.

        {driver} showed a best lap of {best_lap} and an average pace of {avg_lap}.
        Tyre behaviour suggests strategy flexibility may be important.

        Error: {e}
        """