# Imitation Learning
|       Envs    | Expert Reward Mean | Behavior Cloning | DAgger |
|:-------------:|------------------------:|-----------------:|-------:|
|Ant-v1         | 4802.707680  | 4767.6190| 4784.7029|
|HalfCheetah-v1 | 4126.918521  | 4107.9111| 4128.2978|
|Hopper-v1      | 3777.821053  | 2910.7559|3776.3047|
|Humanoid-v1    | 10429.852380 | 8692.7145| 10306.7463 |
|Reacher-v1     | -3.894341    | -6.0473  |-4.9099 |
|Walker2d-v1    | 5523.786277  | 5521.5687|5510.4232|


| | Behavior Cloning | Dagger |
|----|--------------|------|
|Ant-v1|![Ant-v1-bc](/hw1/assets/bc_ant.gif)|![Ant-v1-da](/hw1/assets/da_ant.gif)|
|HalfCheetah-v1|![halfcheetah-v1-bc](/hw1/assets/bc_cheetah.gif)|![halfcheetah-v1-da](/hw1/assets/da_cheetah.gif)|
|Hopper-v1|![hopper-v1-bc](/hw1/assets/bc_hopper.gif)|![hopper-v1-da](/hw1/assets/da_hopper.gif)|
|Humanoid-v1|![humanoid-v1-bc](/hw1/assets/bc_humanoid.gif)|![humanoid-v1-da](/hw1/assets/da_humanoid.gif)|
|Reacher-v1|![reacher-v1-bc](/hw1/assets/bc_reacher.gif)|![humanoid-v1-da](/hw1/assets/da_reacher.gif)|
|Walker2D-v1|![walker2d-v1-bc](/hw1/assets/bc_walker2d.gif)|![walker2d-v1-da](/hw1/assets/da_walker2d.gif)|

<p> The code implementation uses Pytorch and two layered neural net with (64, 64) neurons in the hidden layer. Since, the loss function was taken to be Mean Squared Error (MSE). </p>
<p> To run the behavorial cloning code just run the following commands </p>

```
python cloning.py --envname (environment name) 
```

<p> Similarly to run the behavorial cloning code just run the following commands </p>

```
python dagger.py --envname (environment name) 
```


