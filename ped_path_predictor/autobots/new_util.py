import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


def load_data(file, n_obs, n_pred, absolute=False):
    with open(file, 'rb') as f:
        raw = np.load(f, allow_pickle=True)

    window = raw.shape[1] - n_obs - n_pred

    observed_data, predict_data = [], []
    for k in range(0, window, 2):
        observed = raw[:, k:n_obs + k, :]
        pred = raw[:, k + n_obs:n_pred + n_obs + k, 0:2]

        observed_data.append(observed)
        predict_data.append(pred)

    observed_data = np.concatenate(observed_data, axis=0)
    predict_data = np.concatenate(predict_data, axis=0)

    if not absolute:
        # make output relative to the last observed frame
        i_t = observed_data[:, n_obs - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        predict_data = predict_data - i_t

    return torch.tensor(observed_data), torch.tensor(predict_data)


def generate_data(raw, n_obs, n_pred, absolute=False):
    window = raw.shape[1] - n_obs - n_pred

    observed_data, predict_data = [], []
    for k in range(0, window, 2):
        observed = raw[:, k:n_obs + k, :]
        pred = raw[:, k + n_obs:n_pred + n_obs + k, 0:2]

        observed_data.append(observed)
        predict_data.append(pred)

    observed_data = np.concatenate(observed_data, axis=0)
    predict_data = np.concatenate(predict_data, axis=0)

    if not absolute:
        # make output relative to the last observed frame
        i_t = observed_data[:, n_obs - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        predict_data = predict_data - i_t

    return torch.tensor(observed_data), torch.tensor(predict_data)


def get_dat_sets(paths: list, n_obs, n_pred, absolute=False):
    assert len(paths) > 0, "No paths provided"

    train, test = [], []

    for path in paths:
        x, y = load_data(path, n_obs, n_pred, absolute=absolute)
        train.append(x)
        test.append(y)

    return np.concatenate(train), np.concatenate(test)


class IctsDataset(Dataset):
    def __init__(self, path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=False) -> None:
        super().__init__()

        input_ped, output_ped = get_dat_sets([path_int, path_non_int], n_obs, n_pred, absolute=absolute)
        input_car, output_car = get_dat_sets([path_int_car, path_non_int_car], n_obs, n_pred,absolute=absolute)

        self.x = np.concatenate([input_ped, input_car], axis=-1, dtype=np.float32)
        self.y = output_ped

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class CTSDataset(Dataset):
    def __init__(self, data_file_path, n_obs, n_pred, absolute=False)-> None:
        super().__init__()

        with open(data_file_path, 'r') as data_file:
            data = pd.read_json(data_file, lines=True)

        print(f"Number of rows in the dataset: {len(data)}")

        # Filter rows where both pedestrian_trajectory and ego_vehicle_trajectory have exactly 500 entries
        filtered_data = data[
            (data['pedestrian_trajectory'].apply(lambda x: len(x) if isinstance(x, list) else 0) == 500) &
            (data['ego_vehicle_trajectory'].apply(lambda x: len(x) if isinstance(x, list) else 0) == 500)
        ]

        # Print the number of rows remaining after filtering
        print(f"Number of rows after filtering: {len(filtered_data)}")


        # Extract pedestrian_trajectory and ego_vehicle_trajectory columns
        pedestrian_trajectory_array = np.stack(filtered_data['pedestrian_trajectory'].to_numpy())
        ego_vehicle_trajectory_array = np.stack(filtered_data['ego_vehicle_trajectory'].to_numpy())

        # Print the shapes of the combined arrays
        print("Pedestrian Trajectory Array Shape:", pedestrian_trajectory_array.shape)
        print("Ego Vehicle Trajectory Array Shape:", ego_vehicle_trajectory_array.shape)

        input_ped, output_ped = generate_data(pedestrian_trajectory_array, n_obs, n_pred, absolute=absolute)
        input_car, output_car = generate_data(ego_vehicle_trajectory_array, n_obs, n_pred,absolute=absolute)

        print(f"Input Pedestrian Shape: {input_ped.shape}, Output Pedestrian Shape: {output_ped.shape}" )
        print(f"Input Car Shape: {input_car.shape}, Output Car Shape: {output_car.shape}" )

        self.x = np.concatenate([input_ped, input_car], axis=-1, dtype=np.float32)
        self.y = output_ped

        print(f"Combined Input Shape: {self.x.shape}, Combined Output Shape: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class IctsDatasetDynGroup(Dataset):
    def __init__(self, path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=False) -> None:
        super().__init__()

        input_ped, output_ped = get_dat_sets([path_int, path_non_int], n_obs, n_pred, absolute=absolute)
        input_car, output_car = get_dat_sets([path_int_car, path_non_int_car], n_obs, n_pred, absolute=absolute)

        self.x = np.concatenate([input_ped, input_car], axis=-1, dtype=np.float32)
        self.y = np.concatenate([output_ped, output_car], axis=-1, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IctsTrajPed(Dataset):
    def __init__(self, path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=False) -> None:
        super().__init__()

        input_ped, output_ped = get_dat_sets([path_int, path_non_int], n_obs, n_pred, absolute=absolute)
        input_car, output_car = get_dat_sets([path_int_car, path_non_int_car], n_obs, n_pred, absolute=absolute)

        self.x = np.concatenate([input_ped], axis=-1, dtype=np.float32)
        self.y = np.concatenate([output_ped], axis=-1, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IctsTrajCar(Dataset):
    def __init__(self, path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=False) -> None:
        super().__init__()

        input_ped, output_ped = get_dat_sets([path_int, path_non_int], n_obs, n_pred, absolute=absolute)
        input_car, output_car = get_dat_sets([path_int_car, path_non_int_car], n_obs, n_pred, absolute=absolute)

        self.x = np.concatenate([input_car], axis=-1, dtype=np.float32)
        self.y = np.concatenate([output_car], axis=-1, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def getDataloadersDynGroup(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, batch_size=64, absolute=False):
    dataset = IctsDatasetDynGroup(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=absolute)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [int(0.5 * len(test_dataset)),
                                                                             len(test_dataset) - int(
                                                                                 0.5 * len(test_dataset))],
                                                              generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, val_dataloader


def getDataloadersTrajectron(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, batch_size=64, absolute=False):
    dataset_ped = IctsTrajPed(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=absolute)

    train_size = int(0.7 * len(dataset_ped))
    test_size = len(dataset_ped) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_ped, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [int(0.5 * len(test_dataset)),
                                                                             len(test_dataset) - int(
                                                                                 0.5 * len(test_dataset))],
                                                              generator=torch.Generator().manual_seed(42))

    train_dataloader_ped = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader_ped = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader_ped = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataset_car = IctsTrajCar(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=absolute)

    train_size = int(0.7 * len(dataset_car))
    test_size = len(dataset_car) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_car, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [int(0.5 * len(test_dataset)),
                                                                             len(test_dataset) - int(
                                                                                 0.5 * len(test_dataset))],
                                                              generator=torch.Generator().manual_seed(42))

    train_dataloader_car = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader_car = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader_car = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader_ped, test_dataloader_ped, val_dataloader_ped, train_dataloader_car, test_dataloader_car, val_dataloader_car


def getDataloaders(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, batch_size=64, absolute=False):

    dataset = IctsDataset(path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred, absolute=absolute)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))

    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [int(0.5 * len(test_dataset)),
                                                                             len(test_dataset) - int(
                                                                                 0.5 * len(test_dataset))],
                                                              generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, val_dataloader


def getDataloaders(data_file_path, n_obs, n_pred, batch_size=64, absolute=False):

    dataset = CTSDataset(data_file_path, n_obs, n_pred, absolute=absolute)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    test_dataset, val_dataset = torch.utils.data.random_split(
        test_dataset, 
        [int(0.5 * len(test_dataset)), len(test_dataset) - int(0.5 * len(test_dataset))], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader, val_dataloader


class SingleIctsDataset(Dataset):
    def __init__(self, path_ped, path_car, n_obs, n_pred) -> None:
        super().__init__()

        input_ped, output_ped = get_dat_sets([path_ped], n_obs, n_pred)
        input_cat, output_car = get_dat_sets([path_car], n_obs, n_pred)

        self.x = np.concatenate([input_ped, input_cat], axis=-1, dtype=np.float32)
        self.y = output_ped

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SingleIctsDatasetDynGroup(Dataset):
    def __init__(self, path_ped, path_car, n_obs, n_pred) -> None:
        super().__init__()

        input_ped, output_ped = get_dat_sets([path_ped], n_obs, n_pred)
        input_cat, output_car = get_dat_sets([path_car], n_obs, n_pred)

        self.x = np.concatenate([input_ped, input_cat], axis=-1, dtype=np.float32)
        self.y = np.concatenate([output_ped, output_car], axis=-1, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def singleDatasets(path_tuple, n_obs, n_pred, batch_size=64):
    dataset = SingleIctsDataset(path_tuple[0], path_tuple[1], n_obs, n_pred)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def singleDatasetsDynGroup(path_tuple, n_obs, n_pred, batch_size=64):
    dataset = SingleIctsDatasetDynGroup(path_tuple[0], path_tuple[1], n_obs, n_pred)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)