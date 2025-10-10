from src.data_setup import create_dataloader_from_folder


def create_test_loaders(transforms_dicts, test_dir, batch_size=32):
    """
    Crea un diccionario {nombre_transform: dataloader} y devuelve también class_names.

    Parámetros:
        transforms_dicts (dict): Diccionario {nombre_transform: transform}
        test_dir (str): Directorio con los datos de test.
        cfg: Objeto de configuración que contiene cfg.test.batch_size.
        create_dataloader_from_folder (func): Función que devuelve (loader, class_names).

    Retorna:
        tuple: (loaders_dict, class_names)
    """
    loaders_dict = {}

    for k, transform in transforms_dicts.items():
        loader, class_names = create_dataloader_from_folder(
            data_dir=test_dir,
            batch_size=batch_size,
            transform=transform,
            subset_percentage=1.0,
            shuffle=False,
            seed=42,
        )
        loaders_dict[k] = loader  # solo el loader

    return loaders_dict, class_names
