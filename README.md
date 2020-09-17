# CTC speech enhancement
## Dependencies
   python3.0, pytorch=1.0.1, glob, librosa, numpy, pypesq, pystoi
## Prepare Dataset
* Download Timit dataset  
Get Timit dataset at https://github.com/philipperemy/timit, contains a total of 6300 sentences, 630 speakers, we used these as clean dataset.  
* Download noise dataset  
Get noise dataset at  http://web.cse.ohiostate.edu/pnl/corpus/HuNonspeech/ HuCorpus.html or  http://home.ustc.edu.cn/˜xuyong62/demo/115noises.html.
* Edit config.yaml  
```
cd config
vim config.yaml
```
* preprocess ctc_data file
```
python ctcse_data.py
```
## Train ctc_speech enhancement
```
python train_ctcse.py
```
## Test_ctcse
```
python test_ctcse.py
```
## License
MIT License
