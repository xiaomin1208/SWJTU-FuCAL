def is_large_dataset(data_path):
    large_keywords = ['ECL', 'Traffic', 'Solar', 'electricity', 'traffic']
    return any(kw in data_path for kw in large_keywords)


def get_raw_model(model):
    return model.module if hasattr(model, 'module') else model
