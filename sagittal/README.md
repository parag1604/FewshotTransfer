# README #

This is the saggital segmentation approach to the problem.
This folder contains the files and sample data to pretrain
and fine-tune the model.

There are two approahes to fine-tuning:
* Simple Transfer Learning
* MO-Net based learning

### Environment setup instructions ###

Install the following dependencies in a virtual/conda environment:

* pytorch
* cudatoolkit (appropriate version according to your driver)
* pillow

(optional:)

* scikit-image
* matplotlib
* torchvision
* nibabel
* jupyter

### Preprocessing instructions ###

Sample data provided in the data folder.
Main guidelines are as follows:

* Slices should be in the saggital view
* Slices need not be square shape, they will be resized automatically
* Slices of all volumes must be of the same numbers, say 100 (achieve by interpolation)
* Nomenclature should be <volume_id>_<slice_num>.<image_extension(png/bmp/jpg)>
* Volumes must be min-max scaled and images should stored as 8 bit unsigned integers (and not doubles).
* For more details refer documentaion.

### Execution ###

General execution format:
```
python scripts/train_[unet/monet].py <epochs> <batch_size>
        --train-data-path=<path_to_dir>
        --val-data-path=<path_to_dir>
        --test-data-path=<path_to_dir>
        --output-path=<path_to_dir>
        [--pre-training]/[--model-path=<path_to_dir>]
        [--supervision=<fraction_of_supervision>]
        [--method=<method_of_fine_tuning_supervision>]
```

For example, while pretraining execute:
```
python scripts/train_unet.py 10 3 --train-data-path=data/train --val-data-path=data/validate --test-data-path=data/test --output-path=models/lumbar_80.pth --pre-training --supervision=0.8
```
and using this pretrained model, finetuning using:
```
python scripts/train_unet.py 5 5 --train-data-path=data/train --val-data-path=data/validate --test-data-path=data/test --output-path=models/model_rnd_vol_80_20.pth --model-path=models/lumbar_pre.pth --supervision=0.2 --method=vol-r
```

Valid options for Supervision modes are as follows:

* "vol-r" (random volumes)
* "vol-s" (sequential volumes)
* "slice-r" (random slices per volume)
* "slice-c" (central slices of every volume)

### Dependencies ###

The following is a list of dependencies along with their versions (for reference
  in case of problem setting up the environment):
```
  # Name                    Version                   Build  Channel
  _libgcc_mutex             0.1                        main  
  _pytorch_select           0.2                       gpu_0  
  alabaster                 0.7.12                   pypi_0    pypi
  argon2-cffi               20.1.0           py36h7b6447c_1  
  async_generator           1.10             py36h28b3542_0  
  attrs                     20.2.0                     py_0  
  babel                     2.8.0                    pypi_0    pypi
  blas                      1.0                         mkl  
  bleach                    3.2.1                      py_0  
  ca-certificates           2020.11.8            ha878542_0    conda-forge
  certifi                   2020.11.8        py36h5fab9bb_0    conda-forge
  cffi                      1.14.3           py36he30daa8_0  
  chardet                   3.0.4                    pypi_0    pypi
  cloudpickle               1.6.0                      py_0  
  cudatoolkit               10.0.130                      0  
  cudnn                     7.6.5                cuda10.0_0  
  cycler                    0.10.0                   py36_0  
  cytoolz                   0.11.0           py36h7b6447c_0  
  dask-core                 2.30.0                     py_0  
  dbus                      1.13.18              hb2f20db_0  
  decorator                 4.4.2                      py_0  
  defusedxml                0.6.0                      py_0  
  docutils                  0.16                     pypi_0    pypi
  entrypoints               0.3                      py36_0  
  expat                     2.2.10               he6710b0_2  
  fontconfig                2.13.0               h9420a91_0  
  freetype                  2.10.4               h5ab3b9f_0  
  future                    0.18.2                   pypi_0    pypi
  glib                      2.66.1               h92f7085_0  
  gputil                    1.4.0                    pypi_0    pypi
  gst-plugins-base          1.14.0               hbbd80ab_1  
  gstreamer                 1.14.0               hb31296c_0  
  icu                       58.2                 he6710b0_3  
  idna                      2.10                     pypi_0    pypi
  imageio                   2.9.0                      py_0  
  imagesize                 1.2.0                    pypi_0    pypi
  importlib-metadata        2.0.0                      py_1  
  importlib_metadata        2.0.0                         1  
  intel-openmp              2020.2                      254  
  ipykernel                 5.3.4            py36h5ca1d4c_0  
  ipython                   5.8.0                    py36_1    conda-forge
  ipython_genutils          0.2.0                    py36_0  
  ipywidgets                7.5.1                      py_1  
  itk                       5.1.1            py36h9f0ad1d_2    conda-forge
  itk-core                  5.1.1                    pypi_0    pypi
  itk-filtering             5.1.1                    pypi_0    pypi
  itk-numerics              5.1.1                    pypi_0    pypi
  itk-registration          5.1.1                    pypi_0    pypi
  itk-segmentation          5.1.1                    pypi_0    pypi
  jinja2                    2.11.2                     py_0  
  joblib                    0.16.0                   pypi_0    pypi
  jpeg                      9b                   h024ee3a_2  
  jsonschema                3.2.0                      py_2  
  jupyter                   1.0.0                    py36_7  
  jupyter_client            6.1.7                      py_0  
  jupyter_console           5.2.0                    py36_1  
  jupyter_core              4.6.3                    py36_0  
  jupyterlab_pygments       0.1.2                      py_0  
  kiwisolver                1.2.0            py36hfd86e86_0  
  lcms2                     2.11                 h396b838_0  
  ld_impl_linux-64          2.33.1               h53a641e_7  
  libedit                   3.1.20191231         h14c3975_1  
  libffi                    3.3                  he6710b0_2  
  libgcc-ng                 9.1.0                hdf63c60_0  
  libgfortran-ng            7.3.0                hdf63c60_0  
  libpng                    1.6.37               hbc83047_0  
  libprotobuf               3.13.0.1             h8b12597_0    conda-forge
  libsodium                 1.0.18               h7b6447c_0  
  libstdcxx-ng              9.1.0                hdf63c60_0  
  libtiff                   4.1.0                h2733197_1  
  libuuid                   1.0.3                h1bed415_2  
  libxcb                    1.14                 h7b6447c_0  
  libxml2                   2.9.10               hb55368b_3  
  lz4-c                     1.9.2                heb0550a_3  
  markupsafe                1.1.1            py36h7b6447c_0  
  matplotlib                3.3.1                         1    conda-forge
  matplotlib-base           3.3.1            py36h5ffbc53_1    conda-forge
  mistune                   0.8.4            py36h7b6447c_0  
  mkl                       2020.2                      256  
  mkl-service               2.3.0            py36he904b0f_0  
  mkl_fft                   1.2.0            py36h23d657b_0  
  mkl_random                1.1.1            py36h0573a6f_0  
  nbclient                  0.5.1                      py_0  
  nbconvert                 6.0.7                    py36_0  
  nbformat                  5.0.8                      py_0  
  ncurses                   6.2                  he6710b0_1  
  nest-asyncio              1.4.1                      py_0  
  networkx                  2.5                        py_0  
  nibabel                   2.2.1                    pypi_0    pypi
  ninja                     1.10.1           py36hfd86e86_0  
  notebook                  6.1.3            py36h9f0ad1d_0    conda-forge
  numpy                     1.16.4                   pypi_0    pypi
  numpy-base                1.16.4           py36hde5b4d6_0  
  olefile                   0.46                     py36_0  
  openssl                   1.1.1h               h516909a_0    conda-forge
  packaging                 20.4                       py_0  
  pandoc                    2.11                 hb0f4dca_0  
  pandocfilters             1.4.2                    py36_1  
  pcre                      8.44                 he6710b0_0  
  pexpect                   4.8.0                    py36_0  
  pickleshare               0.7.5                    py36_0  
  pillow                    8.0.0            py36h9a89aac_0  
  pip                       20.2.4                   py36_0  
  prometheus_client         0.8.0                      py_0  
  prompt_toolkit            1.0.15                     py_1    conda-forge
  protobuf                  3.13.0                   pypi_0    pypi
  ptyprocess                0.6.0                    py36_0  
  pycparser                 2.20                       py_2  
  pygments                  2.7.1                      py_0  
  pyparsing                 2.4.7                      py_0  
  pyqt                      5.9.2            py36h05f1152_2  
  pyrsistent                0.17.3           py36h7b6447c_0  
  python                    3.6.10               h7579374_2  
  python-dateutil           2.8.1                      py_0  
  python_abi                3.6                     1_cp36m    conda-forge
  pytorch                   1.3.1           cuda100py36h53c1284_0  
  pytz                      2020.1                   pypi_0    pypi
  pywavelets                1.1.1            py36h7b6447c_2  
  pyyaml                    5.3.1            py36h7b6447c_1  
  pyzmq                     19.0.2           py36he6710b0_1  
  qt                        5.9.7                h5867ecd_1  
  qtconsole                 4.7.7                      py_0  
  qtpy                      1.9.0                      py_0  
  readline                  8.0                  h7b6447c_0  
  requests                  2.24.0                   pypi_0    pypi
  scikit-image              0.16.2           py36h0573a6f_0  
  scikit-learn              0.21.3                   pypi_0    pypi
  scipy                     1.1.0                    pypi_0    pypi
  send2trash                1.5.0                    py36_0  
  setuptools                50.3.0           py36hb0f4dca_1  
  simplegeneric             0.8.1                    py36_2  
  sip                       4.19.8           py36hf484d3e_0  
  six                       1.15.0                     py_0  
  snowballstemmer           2.0.0                    pypi_0    pypi
  sphinx                    3.2.1                    pypi_0    pypi
  sphinxcontrib-applehelp   1.0.2                    pypi_0    pypi
  sphinxcontrib-devhelp     1.0.2                    pypi_0    pypi
  sphinxcontrib-htmlhelp    1.0.3                    pypi_0    pypi
  sphinxcontrib-jsmath      1.0.1                    pypi_0    pypi
  sphinxcontrib-qthelp      1.0.3                    pypi_0    pypi
  sphinxcontrib-serializinghtml 1.1.4                    pypi_0    pypi
  sqlite                    3.33.0               h62c20be_0  
  tensorboardx              2.1                        py_0    conda-forge
  terminado                 0.9.1                    py36_0  
  testpath                  0.4.4                      py_0  
  tk                        8.6.10               hbc83047_0  
  toolz                     0.11.1                     py_0  
  torch                     1.4.0+cu100              pypi_0    pypi
  torchvision               0.5.0+cu100              pypi_0    pypi
  tornado                   6.0.4            py36h7b6447c_1  
  tqdm                      4.48.2                   pypi_0    pypi
  traitlets                 4.3.3                    py36_0  
  urllib3                   1.25.10                  pypi_0    pypi
  wcwidth                   0.2.5                      py_0  
  webencodings              0.5.1                    py36_1  
  wheel                     0.35.1                     py_0  
  widgetsnbextension        3.5.1                    py36_0  
  xz                        5.2.5                h7b6447c_0  
  yaml                      0.2.5                h7b6447c_0  
  zeromq                    4.3.3                he6710b0_3  
  zipp                      3.3.1                      py_0  
  zlib                      1.2.11               h7b6447c_3  
  zstd                      1.4.5                h9ceee32_0
```
