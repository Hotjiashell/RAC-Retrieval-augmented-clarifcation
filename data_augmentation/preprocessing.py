def clean_datasets(datasets):
    datasets["Qulac"] = datasets["Qulac"][datasets["Qulac"]["question"].notna()]
    datasets["Qulac"].rename(columns={'topic': 'initial_request'}, inplace=True)
    datasets["ClariQ_train"] = datasets["ClariQ_train"][datasets["ClariQ_train"]["question"].notna()]
    datasets["ClariQ_dev"] = datasets["ClariQ_dev"][datasets["ClariQ_dev"]["question"].notna()]
    datasets["ClariQfkw_train"] = datasets["ClariQfkw_train"][datasets["ClariQfkw_train"]["question"].notna()]
    datasets["ClariQfkw_dev"] = datasets["ClariQfkw_dev"][datasets["ClariQfkw_dev"]["question"].notna()]
    return datasets
