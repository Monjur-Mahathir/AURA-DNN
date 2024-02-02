# AURA-DNN
This is the repository for the WiFi CSI dataset, processing and patient activity recognition for our multi-year project- "AURA". In this project we deployed WiFi sensing systems along with a RGB-D camera in 8 different environments with 16 participants involved. We also collected additional WiFi CSI data only for different activities from 3 different participants in 5 different environments.

A shortened version of our dataset can be found here: https://drive.google.com/drive/folders/1Figo2jOCYac7mg7z2waPvwi89Aam2nDu?usp=sharing. The full version will be updated soon.

Detailed system setup, configuration and other information for this project can be found in this repository: https://github.com/Monjur-Mahathir/AURA.git

# Dataset Structure
Data from each unique recording sessions of our study are provided in separate folder. The folders are named by the following convention: <ENV_ID>_<PARTICIPANT_ID>, where ENV_ID represent a unique identifier for each unique environment, followed by a unique identifier for each person.

Inside each recording session, we provide a directory "Camera" which contains single or multiple subfolders- each containing a maximum of 10,000 image frames. We also provide the "WiFi" directory, which contains multiple pcap files with WiFi CSI data- where each pcap file has 15,000 WiFi packets.

We also provide 3 files that contain relevant activity labels, timestamp of each image frames, and timestamp of the CSI packets in three different files:
1. "timestamp.txt": Each row provides timestamp of each image frame captured during this session with the Intel D435 depth camera. There are three columns: the first column gives the date and time of the capture in <YYYY-MM-DD>_<hh_mm_ss> format. The second column represents a subfolder under the "Camera" directory, and the third column represents the image frame # under that subfolder. For example, "2022-05-06_18:50:02 1 1" maps to an image frame that can be found in "Camera/1/1.png" path, and was captured at 6:50:02 on 2022-05-06.
2. "csi_timestamp.txt": This file contains the time delay between the first WiFi frame received during this session and every other subsequent frames in each row with a microsecond level accuracy. This information can also be extracted by reading the pcap files using csiread library.
3. "combined_timestamp.txt": Each column in this file contains the following information:
   <CAMERA_SUBFOLDER> <IMAGE_FRAME_#> <hh::mm::ss> <IMAGE_FRAME_#> <hh::mm::ss> <PCAP_FILE_ID> <PACKET_> <PCAP_FILE_ID> <PACKET_#> <ACTIVITY>
   For example, this row- "1 3092 19:02:17 3104 19:02:20 10 6945 10 7600 walking" maps to the following statement: From "Camera/1/3092.png" to "Camera/1/3104.png"- these 12 image frames, and from the 6945-th packet in "Wifi/10.pcap" to 7600-th packet in "Wifi/10.pcap"- these 655 WiFi packets correspond to the "walking" activity by the participant.

# How to process CSI data
From the provided Camera- Wifi dataset and corresponding label, data that maps to specific activity can be extracted by using the "Dataset-Processing/segmentation.py" file and providing the recording session. An example of activity class "squat" can be found in "Dataset/2_2/1/", from 1640 to 1670 image frames. The corresponding activity will be extracted:

https://github.com/Monjur-Mahathir/AURA-DNN/assets/80934192/6b018f56-3aee-4b3b-8e25-91dbb9dc651d

as well as the corresponding WiFi CSI (after processing):

![csi](https://github.com/Monjur-Mahathir/AURA-DNN/assets/80934192/ff98d1cf-235c-44cd-b52d-338a23326564)
