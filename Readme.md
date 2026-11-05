# Shadow package
This application is intended for automated generation of 3D tomgrams using our "volume shadow montage" applied on 4D-STEM tilt series dataset acquired by DECTRIS ARINA camera. <br>
The idea is to generate mrc stack from each tiltview by a cone-beam reconstruction, adjusting for oblique view, and summing the results to create unweighted tilt-series backprojection, then perform 3D deconvolution. <br> 
The microscope acquisition parameters should be loaded from a JSON file, which is easily prepared by the Session-JSON program https://github.com/elbaum-lab/Session-JSON. <br>
A certain naming convention is assumed in the dataset (namely, the tiltview index specified in the filenames as _S# and each file should contain up to 100000 ARINA images). <br>
The datasets are most easily acquired using RedPitaya Savvyscan and SerialEM (https://doi.org/10.1107/S2059798326009630).<br>
The volume shadow montage method is described in https://doi.org/10.1093/mam/ozaf126 and its basic implementations are found in https://github.com/Pr4Et/Supplementaries/tree/main/Shadow_Montage. <br>
The final step is Melnikovsky's "3dcon - 3D deconvolution" described in https://doi.org/10.1107/S205979832600999X. The code originates from https://github.com/elbaum-lab/3dcon. <br>
IMOD is used for fiducial-based alignment https://doi.org/10.1006/jsbi.1996.0013. <br>

# Installation instructions (Described for Windows, adpatable for Linux) 

Download the content of the shadow package code (by git or download ZIP) and extract it as the project folder, for example: <br>
C:\Users\stem\PycharmProjects\shadow_package <br>

Check that python 3.13 exists in your system <br>
py -3.13 --version  <br>
or install it from:  https://www.python.org/ <br>

In powershell create the python 3.13 environment: <br>
cd <project folder> <br>
py -3.13 -m venv .venv <br>
call .venv\Scripts\activate.bat <br>

pip3 install torch  --index-url https://download.pytorch.org/whl/cu126 <br>
python -m pip install scipy==1.15.2 tqdm <br>
pip install numpy scipy h5py hdf5plugin mrcfile scikit-image <br>

For IMOD installation: <br>
Find the system python311 installation folder (independent of the python 3.13 used for running the code) and run python install there: <br>
C:\Users\stem\AppData\Local\Programs\Python\Python311\python.exe -m pip install psutil <br>
Make the Python311 folder exposed as environment variable IMOD_PYTHON: <br>
[Environment]::SetEnvironmentVariable( <br>
    "IMOD_PYTHON", <br>
    "C:\Users\stem\AppData\Local\Programs\Python\Python311\python.exe", <br>
    "User" <br>
) <br>

Download and install IMOD for windows from https://bio3d.colorado.edu/imod/download.html, for example imod_5.1.13_win64_CUDA12.1.exe. <br>
This should create anvironment variable IMOD_DIR pointing to "C:\Program Files\IMOD". <br>
Also install java runtime from https://adoptium.net/temurin/releases/ OpenJDK 21 LTS, 64-bit, JRE. <br>

Note that IMOD is using template.adoc settings located in the project folder. 3dcon is using 3dcon.conf settings. <br>

# How to run 
Run from the project folder: <br> 
call .venv\Scripts\activate.bat <br>
python shadow_montage_tiltseries_gui_2d_3d.py --gui <br>

Most convenient is to prepare a shortcut and specify the target as the Run_Shadow_Montage_GUI.bat in your project folder.  <br>
After loading the JSON file a form is displayed and essentially the workflow is completed by clicking the buttons: Run 3D, IMOD ali, Backprojection, 3D deconvolution. <br>
Best results are obtained with underfocus settings, for which the defocus is specified as a negative number. <br>


images/ 
    shadow_package_ScreenCapture.png
