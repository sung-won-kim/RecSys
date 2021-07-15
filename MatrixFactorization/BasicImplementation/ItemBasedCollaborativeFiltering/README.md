# ItemBasedCollaborativeFiltering.py  
Implementation of basic level of Item based collaborative filtering.  
CosineSimilarity(__item에 대한 user들의 ratings__)  
```bash
python ItemBasedCollaborativeFiltering.py --title '영화제목'
```
### Example  
```bash
python ItemBasedCollaborativeFiltering.py --title 'The Dark Knight Rises'
```

### Results  
```bash
$ python ItemBasedCollaborativeFiltering.py --title "The Dark Knight"
title
Prom Night               0.536339
Soccer Dog: The Movie    0.515711
The Matrix               0.515711
The Contender            0.515711
Fire Birds               0.515711
Highlander               0.515711
The Desperado Trail      0.515711
My Best Fiend            0.515711
Johnny Mad Dog           0.515711
Name: The Dark Knight, dtype: float64
```
