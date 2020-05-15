# HiFIll Image Inpainting on Ascend AI Processor \(Python\)

## Project Overview
This project is implementation of HiFill Image Inpainting on Atlas200DK board.
 
## Application Deployment Steps

You can deploy this application on the Atlas 200 DK to inpaint on images with respective masks.

The current application adapts to  [DDK&RunTime](https://www.huaweicloud.com/ascend/resources/Tools)  of 1.32.0.0.

### Prerequisites

Before deploying this sample, ensure that the development environment for Atlas200DK has been setup according to this [guide](https://www.huaweicloud.com/intl/en-us/ascend/doc/Atlas200DK/1.32.0.0(beta)/en/en-us_topic_0204328954.html), or with [ADKInstaller](https://www.huaweicloud.com/intl/en-us/ascend/doc/Atlas200DK/1.32.0.0(beta)/en/en-us_topic_0238626392.html), or with [docker image](https://www.huaweicloud.com/intl/en-us/ascend/resources/Tools)

### Software Preparation

Before running this application, download or git clone the whole repository to any directory on Ubuntu Server where  Mind Studio  is located, and copy [samples](https://github.com/Atlas200dk/sample-imageinpainting-HiFill/tree/master/samples) folder to the directory of [Huawei_Ascend](https://github.com/Atlas200dk/sample-imageinpainting-HiFill/tree/master/Huawei_Ascend).


### Setup Python Environment on Atlas200DK Board

Note: If the HiAI library, OpenCV library, and related dependencies have been installed on the developer board, skip this step.

1.  Configure the network connection of the developer board.

    Configure the network connection of the Atlas DK developer board by referring to  [https://github.com/Atlas200dk/sample-README/tree/master/DK\_NetworkConnect](https://github.com/Atlas200dk/sample-README/tree/master/DK_NetworkConnect).

2.  Install the environment dependencies（please deploy in python3）.

    Configure the environment dependency by referring to  [https://github.com/Atlas200dk/sample-README/tree/master/DK\_Environment](https://github.com/Atlas200dk/sample-README/tree/master/DK_Environment).

### Deployment

Copy the application codes and samples to the developer board.

Go to the directory of 'Huawei_Ascend', and run the following command to copy the code and samples to the developer board.

**scp -r ../Huawei_Ascend/ HwHiAiUser@192.168.1.2:/home/HwHiAiUser/HIAI\_PROJECTS**

Type the password of the developer board as prompted. The default password is **Mind@123**.

### Run

1.  Log in to the host side as the  **HwHiAiUser**  user in SSH mode on Ubuntu Server where  Mind Studio  is located.

    **ssh HwHiAiUser@192.168.1.2**

    >![](public_sys-resources/icon-note.gif) **NOTE:**   
    >-   The following uses the USB connection mode as an example. In this case, the IP address is 192.168.1.2. Replace the IP address as required.  

2.  Go to the directory where the application code is stored as the  **HwHiAiUser**  user.

3.  Run the application.

    **python3 test_Dchip.py**

## Project Layout
    Huawei_Ascend
    ├── ModelManager.py              # Graph creation and inference 
    ├── inpaint.om                   # Huawei Ascend Offline model for HiFill Image inpainting        
    ├── matmul.om                    # Huawei Ascend Offline model to accelerate matrix muliplication in postprocessing        
    ├── inpaint_inference.py         # inpainting graph
    ├── matmul_inference.py          # matmul graph
    ├── test_Dchip.py                # script to run the application
  
