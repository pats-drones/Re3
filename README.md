# PATS tutorial on training and testing re3

For any missing information see the READ.md from original repository called READ_OLD.MD

## Adding data to the dataset
Add the video you want to add to the dataset to the headless csv file in:
```bash
/home/pats/Documents/datasets/contains_moth.csv
```

After this you annotate the data by starting the annotation program
```bash
cd Documents/annotation/OpenLabeling/main
conda activate re3
python main.py -i /home/pats/Documents/datasets/contains_moth.csv
```

Now convert the labels to a format suitable for the re3 model by running:
```bash
cd Documents/models/re3-tensorflow
conda activate re3
python training/videos_to_format.py
```

## converting annotated data to a re3 suitable format

## training re3
Get the weights from https://drive.google.com/drive/folders/13H-FnhMfjNDbcUX7RYF_V9GjWRt4WNnW.  
The weights_base.zip contains the weights that have not seen the moth data yet, but only images from ImageNet.  
The zip with a datetime in the name are the weights of one of the better performing models.  
Put in the logs folder within the re3 folder.  

First check if tensorflow is working on the GPU by running the following script in the terminal that is located in the main folder of of re3
```bash
cd Documents/models/re3-tensorflow
conda activate re3
python test_gpu.py
```

Open three terminals and run in terminal 1:
```bash
cd Documents/models/re3-tensorflow
conda activate re3
python training/batch_cache.py -n 2
```
Run in terminal 2:
```bash
cd Documents/models/re3-tensorflow
conda activate re3
python training/unrolled_solver.py -rtc -n 2 -b 64 -m 1000 --run_val
```

Now your model with start training, now open the third terminal to start up tensorboard to track training: 

```bash
cd Documents/models/re3-tensorflow
conda activate re3
tensorboard --logdir="/home/pats/Documents/models/re3-tensorflow/logs/train"
```
Where "log_dir" is the directory where the logs for the tensorboard can be found.
Here you can see the losses of your model to make it easier to compare different models or track the loss of the currently trained model/


## testing trained model
To test your data on an validation set:
```bash
cd Documents/models/re3-tensorflow
conda activate re3
python test_net.py -m val --weight_dir {dir_of_weights_of_tested_model}#checkpoint_2020_07_22_16_53_05_n_2_b_64_4677
```

