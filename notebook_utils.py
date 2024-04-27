def find_compound_kinase_pairs_with_both_measurements(data):
    # Get all keys in SMILES column
    unique_compounds = set(data["SMILES"])

    # This is a dictionary of pairs in which there is a pKi and pIC50 for the same
    # protein and same compound
    comparison_pairs = {}

    for compound in list(unique_compounds):
        # Make a Series of all entries from a given compund
        compound_results = data[data["SMILES"] == compound]

        # Verify that we have entries that represent both types of output metric
        number_of_metric_types = len(set(compound_results["measurement_type"]))

        if number_of_metric_types == 2:
            # If our subset has both measurement types represented, can we find a case
            # where 2 of the same protein are involved (one for pKi, one for pIC50)
            for kinase in ["JAK1", "JAK2", "JAK3", "TYK2"]:
                value_counts = compound_results["Kinase_name"].value_counts()
                if kinase in value_counts and value_counts[kinase] > 1:

                    only_kinase = compound_results[
                        compound_results["Kinase_name"] == kinase
                    ]
                    pKi_val = float(
                        only_kinase[only_kinase["measurement_type"] == "pKi"][
                            ["measurement_value"]
                        ].iloc[0]
                    )
                    pIC50_val = float(
                        only_kinase[only_kinase["measurement_type"] == "pIC50"][
                            "measurement_value"
                        ].iloc[0]
                    )

                    if compound in comparison_pairs:
                        comparison_pairs[compound][kinase] = {
                            "pKi": pKi_val,
                            "pIC50": pIC50_val,
                        }
                    else:
                        comparison_pairs[compound] = {
                            kinase: {"pKi": pKi_val, "pIC50": pIC50_val}
                        }
    return comparison_pairs


def get_2metric_comparison_series(kinase_list, comparison_pairs):
    pKi = []
    pIC50 = []

    for compound in comparison_pairs:
        for kinase in kinase_list:
            if kinase in comparison_pairs[compound]:
                pKi.append(comparison_pairs[compound][kinase]["pKi"])
                pIC50.append(comparison_pairs[compound][kinase]["pIC50"])

    return pKi, pIC50


def find_multi_kinase_compound_responses(data):
    comparison_pairs = {}

    unique_compounds = set(data["SMILES"])

    for compound in list(unique_compounds):
        # Make a Series of all entries from a given compund
        compound_results = data[data["SMILES"] == compound]

        pki_results = compound_results[compound_results["measurement_type"] == "pKi"]
        pIC50_results = compound_results[
            compound_results["measurement_type"] == "pIC50"
        ]

        if len(pki_results) > 1:

            comparison_pairs[compound] = {"pKi": {}}

            for i in range(len(pki_results)):
                row = pki_results.iloc[i]
                kinase = row["Kinase_name"]
                value = row["measurement_value"]
                comparison_pairs[compound]["pKi"][kinase] = value

        if len(pIC50_results) > 1:

            if compound in comparison_pairs:
                comparison_pairs[compound]["pIC50"] = {}
            else:
                comparison_pairs[compound] = {"pIC50": {}}

            for i in range(len(pIC50_results)):
                row = pIC50_results.iloc[i]
                kinase = row["Kinase_name"]
                value = row["measurement_value"]
                comparison_pairs[compound]["pIC50"][kinase] = value
    return comparison_pairs


def get_2kinase_comparison_series(k1, k2, comparison_pairs):

    kinase_1 = []
    kinase_2 = []

    for key in comparison_pairs:
        if "pKi" in comparison_pairs[key]:
            if (
                k1 in comparison_pairs[key]["pKi"]
                and k2 in comparison_pairs[key]["pKi"]
            ):
                kinase_1.append(comparison_pairs[key]["pKi"][k1])
                kinase_2.append(comparison_pairs[key]["pKi"][k2])

        if "pIC50" in comparison_pairs[key]:
            if (
                k1 in comparison_pairs[key]["pIC50"]
                and k2 in comparison_pairs[key]["pIC50"]
            ):
                kinase_1.append(comparison_pairs[key]["pIC50"][k1])
                kinase_2.append(comparison_pairs[key]["pIC50"][k2])

    return kinase_1, kinase_2
