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

