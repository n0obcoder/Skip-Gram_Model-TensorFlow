# Skip-Gram-Model-TensorFlow
TensorFlow implementation of the word2vec (skip-gram model)


# Requirements
* tensorflow >= 2.0    
* numpy >= 1.18      
* matplotlib       
* tqdm 
* nltk
* gensim


# Training
```
python main.py
```

# Visualizing real-time training loss in Tensorboard
```
tensorboard --logdir <PATH_TO_TENSORBOARD_EVENTS_FILE>
```
<strong>NOTE:</strong> By default, <strong>PATH_TO_TENSORBOARD_EVENTS_FILE</strong> is set to <strong>SUMMARY_DIR</strong> in config.py

# Testing
```
python test.py
```

# Inference

| war      | india     | crime       | guitar  | movies  | desert    | physics      | religion  | football     | computer   |  
| -------- | --------- | ------------| ------- | ------- | --------- | ------------ | --------- | ------------ | ---------- |
| invasion | provinces | will        | bass    | movie   | shore     | mathematics  | judaism   | baseball     | digital    |
| soviet   | pakistan  | prosecution | drum    | albums  | hilly     | mathematical | islam     | championship | computers  |
| troop    | mainland  | accusations | solo    | songs   | plateau   | chemistry    | religions | basketball   | software   |
| army     | asian     | provoke     | quartet | cartoon | basin     | theoretical  | religious | coach        | electronic |
| ally     | colonial  | prosecute   | vocals  | animate | highlands | analysis     | jewish    | wrestler     | interface  |
