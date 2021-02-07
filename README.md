# How to execute my code 
B07902054 資工三 林子權
## Preparing Data
首先，必須先在跟src資料夾同個目錄下，創立一個叫做Train的目錄，並且在Train目錄下建立四個子目錄：clean_voice、noise_voice、spectrogram、sound。再來，創立以下幾個資料夾以方便之後的使用：demo、demo_ans、model、picture。這個步驟結束後，目錄的結構應該像下面這樣。
```
final_project_b07902054
|--- src
     |--- some code
|--- Train
     |--- clean_voice
     |--- noise_voice
     |--- spectrogram
     |--- sound
|--- report.pdf
|--- README.pdf
|--- model
|--- demo
|--- demo_ans
|--- picture
```
再來，把訓練用的clean speech audio全部放到clean_voice這個資料夾、把訓練用的noise audio全部放到noise_voice這個資料夾。接著就可以跑generate_data.py來產生訓練資料了：
```
python generate_data.py
```
跑完了以後，會產生一個長度為"num_sample"秒左右的訓練資料，音檔放在sound這個資料夾，會有noise、noisy voice、clean speech這三個版本；頻譜圖的numpy array data放在spectrogram這個資料夾裡面，一樣會有以上三種版本。

如果要跑training的話，要把generate_data.py裡面的TRAIN改成True，如果只是要做demo data要改成False。

## Training 
產生完data之後，執行下面這一行指令：
```
python train.py
```
就會開始train40個epoch，train好的model會放到model這個資料夾，training loss和validation loss對epoch的折線圖會放到picture這個資料夾裡。

因為我的best model是分兩次training train出來的，所以助教應該沒辦法執行一次就得到一個performance一樣的model。我是先用learning rate=1e-5跑20個epoch，再用learning rate=1e-4跑40個epoch。
## Predicting(demo)
執行下面這一行指令：
```
python predict.py
```
會把"demo"資料夾裡面的noisy speech audio們輸出成一個predicting clean speech audio叫做"denoice.wav"到根目錄。

# Conda Environment
```
channels:
  - pytorch
  - defaults
dependencies:
  - blas=1.0=mkl
  - boto=2.49.0=py37_0
  - boto3=1.9.66=py37_0
  - botocore=1.12.67=py37_0
  - ca-certificates=2020.12.8=haa95532_0
  - cachetools=4.2.0=pyhd3eb1b0_0
  - certifi=2020.12.5=py37haa95532_0
  - cffi=1.14.4=py37hcd4344a_0
  - chardet=4.0.0=py37haa95532_1003
  - cryptography=3.3.1=py37hcd4344a_0
  - cudatoolkit=10.2.89=h74a9793_1
  - cycler=0.10.0=py37_0
  - docutils=0.16=py37_1
  - freetype=2.10.4=hd328e21_0
  - gensim=3.8.0=py37hf9181ef_0
  - google-api-core=1.22.2=py37h21ff451_0
  - google-auth=1.21.3=py_0
  - google-cloud-core=1.5.0=pyhd3eb1b0_0
  - google-cloud-storage=1.35.0=pyhd3eb1b0_0
  - google-crc32c=1.1.0=py37h2bbff1b_1
  - google-resumable-media=1.2.0=pyhd3eb1b0_1
  - googleapis-common-protos=1.52.0=py37h21ff451_0
  - icc_rt=2019.0.0=h0cc432a_1
  - icu=58.2=ha925a31_3
  - idna=2.10=py_0
  - intel-openmp=2020.2=254
  - jmespath=0.10.0=py_0
  - jpeg=9b=hb83a4c4_2
  - kiwisolver=1.3.0=py37hd77b12b_0
  - libcrc32c=1.1.1=ha925a31_2
  - libpng=1.6.37=h2a8f88b_0
  - libprotobuf=3.13.0.1=h200bbdf_0
  - libtiff=4.1.0=h56a325e_1
  - lz4-c=1.9.2=hf4a77e7_3
  - matplotlib=3.3.2=0
  - matplotlib-base=3.3.2=py37hba9282a_0
  - mkl=2020.2=256
  - mkl-service=2.3.0=py37h196d8e1_0
  - mkl_fft=1.2.0=py37h45dec08_0
  - mkl_random=1.1.1=py37h47e9c7a_0
  - multidict=5.1.0=py37h2bbff1b_2
  - ninja=1.10.2=py37h6d14046_0
  - numpy=1.19.2=py37hadc3359_0
  - numpy-base=1.19.2=py37ha3acd2a_0
  - olefile=0.46=py37_0
  - openssl=1.1.1i=h2bbff1b_0
  - pandas=1.1.5=py37hf11a4ad_0
  - pillow=8.0.1=py37h4fa10fc_0
  - pip=20.3.1=py37haa95532_0
  - protobuf=3.13.0.1=py37ha925a31_1
  - pyasn1=0.4.8=py_0
  - pyasn1-modules=0.2.8=py_0
  - pycparser=2.20=py_2
  - pyopenssl=20.0.1=pyhd3eb1b0_1
  - pyparsing=2.4.7=py_0
  - pyqt=5.9.2=py37h6538335_2
  - pysocks=1.7.1=py37_1
  - python=3.7.5=h8c8aaf0_0
  - python-dateutil=2.8.1=py_0
  - pytorch=1.6.0=py3.7_cuda102_cudnn7_0
  - pytz=2020.4=pyhd3eb1b0_0
  - qt=5.9.7=vc14h73c81de_0
  - requests=2.25.1=pyhd3eb1b0_0
  - rsa=4.6=py_0
  - s3transfer=0.1.13=py37_0
  - setuptools=51.0.0=py37haa95532_2
  - sip=4.19.8=py37h6538335_0
  - six=1.15.0=py37haa95532_0
  - smart_open=2.0.0=py_0
  - sqlite=3.33.0=h2a8f88b_0
  - tk=8.6.10=he774522_0
  - torchvision=0.7.0=py37_cu102
  - tornado=6.1=py37h2bbff1b_0
  - urllib3=1.24.3=py37_0
  - vc=14.2=h21ff451_1
  - vs2015_runtime=14.27.29016=h5e58377_2
  - wheel=0.36.1=pyhd3eb1b0_0
  - win_inet_pton=1.1.0=py37haa95532_0
  - wincertstore=0.2=py37_0
  - xz=5.2.5=h62dcd97_0
  - yarl=1.5.1=py37he774522_0
  - zlib=1.2.11=h62dcd97_4
  - zstd=1.4.5=h04227a9_0
  - pip:
    - joblib==0.17.0
    - scikit-learn==0.23.2
    - scipy==1.5.4
    - threadpoolctl==2.1.0
```