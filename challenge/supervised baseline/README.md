# Generate games

```
python pyrat.py -d 0 -md 0 --rat AIs/manh.py --python AIs/manh.py --nonsymmetric --nodrawing --tests 1000 --synchronous --save
```

# Generate the dataset

```
python generate_dataset.py
```

# Train the Random Forest Classifier

```
python train.py
```

# Test in pyrat

copy save.pkl, utils.py to the pyrat root folder and supervised to the pyrat root/AIs folder

```
python pyrat.py -d 0 -md 0 --rat AIs/manh.py --python AIs/supervised.py --nonsymmetric --nodrawing --tests 1000 --synchronous --save
```

The expected results are something like:

```
{
	"miss_python": 0.27
	"miss_rat": 0.0
	"moves_python": 64.716
	"moves_rat": 64.986
	"prep_time_python": 0.22859769797325133
	"prep_time_rat": 3.48663330078125e-06
	"score_python": 18.283
	"score_rat": 19.256
	"stucks_python": 0.0
	"stucks_rat": 0.0
	"turn_time_python": 0.0012132460593000472
	"turn_time_rat": 1.7806066873676845e-05
	"win_python": 0.422
	"win_rat": 0.533
}
```
