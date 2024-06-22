# Beyond Pattern Variance: Unsupervised 3D Action Representation Learning with Point Cloud Sequence #

## Introduction
This is the official implementation for the paper **"Beyond Pattern Variance: Unsupervised 3D Action Representation Learning with Point Cloud Sequence"**, IEEE Transactions on Neural Networks and Learning System



## Installation
 ***
  Install the corresponding dependencies in the `requirement.txt`:

```python
    pip install requirement.txt
 ```   

## Data generation
    first place the depth map of NTU 120 dataset on ../ntu120dataset
```python   
    cd /generate_data
    run python generate_NTU.py
```

## Train
```python  
    cd /training_code
    python cn3d_train_motion_GL.py
    python extract_motion_feature.py

    python cn3d_train_apperance_GL.py
    python extract_apperance_feature.py
```

## Test
```python
    cd /linear_classify
    python linercls.py
```



