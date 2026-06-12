import pandas as pd


def predict_lap(model, columns, lap, tyre_life, stint, compound, total_laps):
    row = pd.DataFrame([{
        "LapNumber": lap,
        "TyreLife": tyre_life,
        "TyreLifeSquared": tyre_life ** 2,
        "LapProgress": lap / total_laps,
        "Stint": stint,
        "IsEarlyStint": int(tyre_life <= 5),
        "IsOldTyre": int(tyre_life >= 15),
        "Compound": compound
    }])

    row = pd.get_dummies(
        row,
        columns=["Compound"]
    )

    row = row.reindex(
        columns=columns,
        fill_value=0
    )

    return model.predict(row)[0]


def simulate_strategy(model, columns, total_laps, pit_laps, pit_loss, compounds):
    total_time = 0
    tyre_life = 1
    stint = 1
    compound_index = 0
    stint_plan = []

    for lap in range(1, total_laps + 1):

        if lap in pit_laps:
            total_time += pit_loss
            tyre_life = 1
            stint += 1
            compound_index = min(
                compound_index + 1,
                len(compounds) - 1
            )

        compound = compounds[compound_index]

        lap_time = predict_lap(
            model=model,
            columns=columns,
            lap=lap,
            tyre_life=tyre_life,
            stint=stint,
            compound=compound,
            total_laps=total_laps
        )

        total_time += lap_time

        stint_plan.append({
            "Lap": lap,
            "Compound": compound,
            "TyreLife": tyre_life,
            "PredictedLapTime": lap_time,
            "Stint": stint
        })

        tyre_life += 1

    return total_time, pd.DataFrame(stint_plan)


def optimize_strategy(model, columns, total_laps, pit_loss):
    results = []

    compound_sets = {
        "1 Stop MED-HARD": ["MEDIUM", "HARD"],
        "1 Stop SOFT-HARD": ["SOFT", "HARD"],
        "2 Stop SOFT-MED-HARD": ["SOFT", "MEDIUM", "HARD"]
    }

    for strategy_name, compounds in compound_sets.items():

        if len(compounds) == 2:

            for pit1 in range(12, total_laps - 10, 6):

                total_time, stint_plan = simulate_strategy(
                    model=model,
                    columns=columns,
                    total_laps=total_laps,
                    pit_laps=[pit1],
                    pit_loss=pit_loss,
                    compounds=compounds
                )

                results.append({
                    "Strategy": strategy_name,
                    "Pit Laps": f"{pit1}",
                    "PitLap1": pit1,
                    "PitLap2": None,
                    "Predicted Total Time": total_time,
                    "Compounds": " → ".join(compounds),
                    "StintPlan": stint_plan
                })

        if len(compounds) == 3:

            for pit1 in range(10, total_laps - 20, 8):

                for pit2 in range(pit1 + 10, total_laps - 5, 8):

                    total_time, stint_plan = simulate_strategy(
                        model=model,
                        columns=columns,
                        total_laps=total_laps,
                        pit_laps=[pit1, pit2],
                        pit_loss=pit_loss,
                        compounds=compounds
                    )

                    results.append({
                        "Strategy": strategy_name,
                        "Pit Laps": f"{pit1}, {pit2}",
                        "PitLap1": pit1,
                        "PitLap2": pit2,
                        "Predicted Total Time": total_time,
                        "Compounds": " → ".join(compounds),
                        "StintPlan": stint_plan
                    })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df, None

    best_row = results_df.loc[
        results_df["Predicted Total Time"].idxmin()
    ]

    return results_df, best_row