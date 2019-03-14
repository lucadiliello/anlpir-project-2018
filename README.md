# ANLPIR project by Luca Di Liello and Martina Paganin

Please launch with:
```bash
./main.py [-p] [-m <GoogleRed|Google|Glove|Wiki|LearnPyTorch|LearnGensim>] [-d <TrecQA|WikiQA>] [-n <CNN|biLSTM|AP-CNN|AP-biLSTM>]
```
* Note that using `-p` option requires a dedicated CUDA GPU with at least 4GB of VRAM.

Or customize and launch:
```bash
./optimizer.py
```
to greedy search for the best parameters configuration.

* Word Embedding models have to be added manually to the `./models` folder since they are really large files.
