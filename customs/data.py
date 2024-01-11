import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# 시퀀스 데이터 생성 함수
def create_sequences(seq_length=5, num_sequences=1000, start=-1.0, end=1.0, step=0.01):
    sequences = []
    while len(sequences) < num_sequences:
        start_val = random.uniform(start, end - seq_length * step)
        sequence = [start_val + i * step for i in range(seq_length)]
        if all(start <= x <= end for x in sequence):  # 시퀀스 내 모든 값이 -1과 1 사이인지 확인
            sequences.append(sequence)
    return np.array(sequences)

# 데이터셋 클래스
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        #self.sequences = sequences
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequences = torch.tensor(sequences, dtype=torch.float).to(device)
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]  # 입력 시퀀스 (마지막 토큰 제외)
        target_seq = sequence[1:]  # 타겟 시퀀스 (첫 번째 토큰 제외)
        return input_seq.clone().detach(), target_seq.clone().detach()

def normalize_data(data, original_min, original_max, new_min, new_max):
    """
    Normalize the data from the original range to the new range.
    Asserts if any data value is outside the original range.

    :param data: List of data values to be normalized.
    :param original_min: Minimum value of the original data range.
    :param original_max: Maximum value of the original data range.
    :param new_min: Minimum value of the new data range.
    :param new_max: Maximum value of the new data range.
    :return: List of normalized data values.
    """
    assert all(original_min <= x.any() <= original_max for x in data), "Data values must be within the original range"
    return [(x - original_min) * (new_max - new_min) / (original_max - original_min) + new_min for x in data]

def denormalize_data(normalized_data, original_min, original_max, new_min, new_max):
    """
    Denormalize the data from the new range back to the original range.

    :param normalized_data: List of normalized data values.
    :param original_min: Minimum value of the original data range.
    :param original_max: Maximum value of the original data range.
    :param new_min: Minimum value of the new data range.
    :param new_max: Maximum value of the new data range.
    :return: List of denormalized data values.
    """
    return [(x - new_min) * (original_max - original_min) / (new_max - new_min) + original_min for x in normalized_data]
