import os 
import numpy as np
import csiread


def read_csi(csifile):
    csidata = csiread.Nexmon(csifile, chip='4366c0', bw=80)
    csidata.read()
    return csidata


def process_csi(csidata, start=None, end=None, n_ss=1, n_core=1):
    csi_buff = []
    for i in range(csidata.csi.shape[0]):
        if csidata.core[i] < 4 and csidata.spatial[i] == 0:
            csi_buff.append(csidata.csi[i])
    
    csi_buff = np.array(csi_buff, dtype=np.complex_)
    
    if end is not None:
        csi_buff = csi_buff[start:end, :]

    csi_buff = np.fft.fftshift(csi_buff, axes=1)

    delete_idxs = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
    csi_buff = np.delete(csi_buff, delete_idxs, axis=0)
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)

    n_tot = n_ss * n_core

    start = 0
    end = int(np.floor(csi_buff.shape[0]/n_tot))
    signal_complete = np.zeros((csi_buff.shape[1] - delete_idxs.shape[0], end-start, n_tot), dtype=complex)

    for stream in range(0, n_tot):
        signal_stream = csi_buff[stream:end*n_tot + 1:n_tot, :]
        signal_stream = signal_stream[start:end, :]
        
        signal_stream[:, 64:] = - signal_stream[:, 64:]

        signal_stream = np.delete(signal_stream, delete_idxs, axis=1) 
        mean_signal = np.mean(np.abs(signal_stream), axis=1, keepdims=True)
        H_m = signal_stream/mean_signal
        
        signal_complete[:, :, stream] = H_m.T
    
    return signal_complete


if __name__ == "__main__":
    recording_session = input("Recording Session: ")

    csi_root = f"Dataset/{recording_session}/Wifi/"
    combined_timestamp = f"Dataset/{recording_session}/combined_timestamp.txt"
    out_dir = f"Dataset/{recording_session}/segmented_dataset/"
    out_file = f"Dataset/{recording_session}/segmented_metadata.txt"

    ofid = 0
    
    with open(combined_timestamp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split('\n')[0].split(' ')
            
            csi_start_file, csi_start_packet, csi_end_file, csi_end_packet = int(words[5]), int(words[6]), int(words[7]), int(words[8])
            activity = words[9]
            
            if csi_start_file == csi_end_file:
                csi_data = read_csi(csi_root+str(csi_start_file)+".pcap")
                full_signal = process_csi(csidata=csi_data, start=csi_start_packet, end=csi_end_packet, n_ss=1, n_core=1)
            else:
                full_signal = []
                curr_file = csi_start_file
                while curr_file <= csi_end_file:
                    csi_data = read_csi(csi_root+str(curr_file)+".pcap")
                    if curr_file == csi_end_file:
                        signal = process_csi(csidata=csi_data, start=0, end=csi_end_packet, n_ss=1, n_core=1)
                    elif curr_file == csi_start_file:
                        signal = process_csi(csidata=csi_data, start=csi_start_packet, end=15000, n_ss=1, n_core=1)
                    curr_file += 1
                    for i in range(signal.shape[1]):
                        full_signal.append(signal[:, i, 0])
                full_signal = np.array(full_signal)
                full_signal = np.swapaxes(full_signal, 0, 1)
                full_signal = full_signal[:, :, np.newaxis]
            
            """Saving the segmented signal and write metadata for this session"""
            save_file = out_dir + str(ofid) + ".npy"
            np.save(save_file, full_signal) 
            with open(out_file, 'a') as f1:
                out_str = save_file + " " + activity + "\n"
                f1.writelines(out_str)
            f1.close()

            ofid += 1
    f.close()