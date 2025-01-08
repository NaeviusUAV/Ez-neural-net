
hey guys, i thought it was rlly annoying to implement a neural network for small quick tasks or hobby projects, even though it shouldnt be. 

so i made nn_utils.py with which you can just:
```python
from nn_utils import train_neural_network

# logic for getting values to train it with (theyre lists in a list)
# in this example we are teaching it to understand the 69 seed in randint()
max_training_time = 1000 # in seconds, but it will cut off automatically at an optimal time.

train_neural_network(input_values, target_values, output_path="ez", max_training_time) 
```

it will output a folder with the model files and a use_util.py

to use it make a new script and import it:
```python
from use_util.py import use_model

input = [45, 21, 34, 39, 6, 13, 24, 16, 0, 0, 50] # single list this time
output = use_model(input)

print(output) 
```
which resulted in 
[[46.88339614868164], [22.18029022216797], [35.56113815307617], [40.70762252807617], [6.7408447265625], [13.945919036865234], [25.268178939819336], [17.03380584716797], [0.5591211318969727], [0.5591211318969727], [52.02988052368164]] 

im not sure why, but its rand int, its 1:14 am so i dont have time to extensively test it. 


This was a small little side project i wanted to share with anyone here getting into neural networks, or advanced people who want an easy implementation, or someone whos interested in improving this further. im sure it already exists but i couldnt find it.

