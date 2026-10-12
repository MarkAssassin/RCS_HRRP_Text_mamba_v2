import scipy.io
import numpy as np
import json

# 该文件用于生成RCS的文本描述并保存为npy文件，供后续训练使用

def load_data(freq, snr):
    # rcs
    data_path_train = f'text_creator_v2/data/rcs_train_data_{freq}_{snr}dB.mat'
    rcs_data_train = scipy.io.loadmat(data_path_train)
    data_path_test = f'text_creator_v2/data/rcs_test_data_{freq}_{snr}dB.mat'
    rcs_data_test = scipy.io.loadmat(data_path_test)
    rcs_data_train = abs(rcs_data_train['train_data'])
    rcs_data_test = abs(rcs_data_test['test_data'])
    data_rcs = np.concatenate((rcs_data_train, rcs_data_test), axis=0)
    # hrrp
    data_path_hrrp = f'text_creator_v3/hrrp_data/hrrp_description_{freq}_{snr}dB_qwen3-vl-8b-local.jsonl'
    hrrp_text = []
    with open(data_path_hrrp, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "description" in obj:
                hrrp_text.append(obj["description"])
    return data_rcs, hrrp_text

def create_text_data(freq, snr, data_rcs, hrrp_text, file_path:str):
    if freq == 'C':
        carrier_freq_ghz = 6.0
    elif freq == 'S':
        carrier_freq_ghz = 3.0
    elif freq == 'X':
        carrier_freq_ghz = 10.0

    idx_list = [0, 1500, 3000, 4800]
    for i in idx_list:
        text = hrrp_text[i]
        print(text)

    text_data_list = []
    for i in range(data_rcs.shape[0]):
        data_max = max(data_rcs[i,:])
        idx_max = list(data_rcs[i,:]).index(data_max)
        data_min = min(data_rcs[i,:])
        idx_min = list(data_rcs[i,:]).index(data_min)
        data_mean = np.mean(data_rcs[i,:])
        data_std = np.std(data_rcs[i,:])
        if data_max - data_min > 0.01:
            text_data  = (
                        f"A radar target is observed. "+
                        f"The carrier frequency is {carrier_freq_ghz:.1f} GHz and the signal-to-noise ratio is {snr} dB.\n"+
                        f"RCS description: The sequence has an average value of {data_mean:.4f} and a standard deviation of {data_std:.4f}. "+
                        f"The maximum value {data_max:.4f} occurs at position {idx_max + 1}, "+
                        f"and the minimum value {data_min:.4f} occurs at position {idx_min + 1}. "+
                        f"The sequence shows noticeable peak-valley variations.\n"+
                        f"HRRP description: {hrrp_text[i]}"
                    )
        else:
            text_data = (
                        f"A radar target is observed. " +
                        f"The carrier frequency is {carrier_freq_ghz:.1f} GHz and the signal-to-noise ratio is {snr} dB.\n" +
                        f"RCS description: The sequence has an average value of {data_mean:.4f} and a standard deviation of {data_std:.4f}. " +
                        f"The maximum value {data_max:.4f} occurs at position {idx_max + 1}, " +
                        f"and the minimum value {data_min:.4f} occurs at position {idx_min + 1}.\n"+
                        f"HRRP description: {hrrp_text[i]}"
                        )
        text_data_list.append(text_data)
    np.save(file_path, np.array(text_data_list))
    # np.save(file_path, np.array(hrrp_text))


if __name__ == '__main__':
    freq = 'X' # C/S/X
    snr = 20
    data_rcs, hrrp_text = load_data(freq, snr)
    file_path = f'text_creator_v3/text_data/data_{freq}_{snr}dB.npy'
    create_text_data(freq, snr, data_rcs=data_rcs, hrrp_text=hrrp_text, file_path=file_path)
    print("Npy-file generated successfully")
