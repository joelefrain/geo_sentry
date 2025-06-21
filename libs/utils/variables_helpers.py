def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))  # Desciende un nivel
        else:
            flat_list.append(item)  # Elemento final
    return flat_list
