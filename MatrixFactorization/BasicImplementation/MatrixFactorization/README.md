# MatrixFactorization.py  
Implementation of basic level of MatrixFactorization  
using SVD(Singualr Value Decomposition) with Pearson Correlation Coefficient  
___item(영화)와 비슷한 items(영화들)을 찾아주는 목적___
```bash
python MatrixFactorization.py --title '영화제목'
```
### Example  
```bash
python MatrixFactorization.py --title 'Back to the Future'
```

### Results  
```bash
$ python MatrixFactorization.py --title "Back to the Future"
title
Berlin: Symphony of a Great City    0.928658
Terminator 2: Judgment Day          0.918555
The Dark                            0.906547
Young Adam                          0.905744
The Devil Wears Prada               0.905391
Notting Hill                        0.894599
My Own Private Idaho                0.890994
Man of Marble                       0.889025
A Beautiful Mind                    0.886904
Name: Back to the Future, dtype: float64
```
---  
# MatrixFactorization2.py  
Implementation of basic level of MatrixFactorization  
using SVD(Singualr Value Decomposition)  
___user(사용자)에게 items(영화)를 추천해주는 목적___
```bash
python MatrixFactorization2.py --userid 132
```
### Example  
```bash
python MatrixFactorization2.py --userid 10 --recNum 5
```

### Results  
```bash
$ python MatrixFactorization2.py --userid 10 --recNum 5
already_rated
    userId  movieId  rating                             genres  vote_average  vote_count popularity                            title                                            tagline                                           overview     score
19      10     2926       5      Action Adventure Comedy Drama           6.8        71.0  11.811966             The Three Musketeers                 . . . One for All and All for Fun!  The young D'Artagnan arrives in Paris with dre...  6.573915
13      10     1923       5                      Drama Mystery           7.3       429.0   8.736538    Twin Peaks: Fire Walk with Me  Meet Laura Palmer... In a town where nothing i...  In the questionable town of Deer Meadow, Washi...  7.075508
7       10     1358       5                        Documentary           7.4        37.0     2.4312          A Brief History of Time                                                NaN  A documentary film based on the life of scient...  6.647585
11      10     1719       5                      Drama Romance           7.1        30.0   2.137534                    The Soft Skin           The Eternal Triangle At Its Most Eternal  Pierre Lachenay is a well-known publisher and ...  6.572496
9       10     1611       5                       Drama Comedy           6.1        37.0  12.485973              The Miracle of Bern                                                NaN  The movie deals with the championship-winning ...  6.403423
0       10      152       4  Science Fiction Adventure Mystery           6.2       541.0   8.277765    Star Trek: The Motion Picture             The human adventure is just beginning.  When a destructive space entity is spotted app...  6.262446
1       10      318       4                     Drama Thriller           5.9        76.0   4.938231         The Million Dollar Hotel                                                NaN  The Million Dollar Hotel starts with a jump fr...  6.288874
17      10     2890       4                        Documentary           6.7         3.0   0.113732  André Hazes, Zij Gelooft in Mij                                                NaN  Portrait of the popular Dutch singer André Hazes.  6.477757
16      10     2841       4                              Drama           7.1       352.0  10.896295           A Very Long Engagement                                       Never let go  In 1919, Mathilde was 19 years old. Two years ...  6.904247
15      10     2539       4                             Comedy           5.8       375.0  11.202764                        Spanglish                           Every family has a hero.  Mexican immigrant and single mother Flor Moren...  6.001447
here's recommendation for you
     movieId                                   genres  vote_average  vote_count popularity                               title                                            tagline                                           overview     score  Predictions
211      296          Action Thriller Science Fiction           5.9      2177.0  20.818907  Terminator 3: Rise of the Machines                            The Machines Will Rise.  It's been 10 years since John Connor saved Ear...  5.939270     0.840845
392      593  Drama Science Fiction Adventure Mystery           7.7       364.0  11.059785                             Solaris                                                NaN  Ground control has been receiving strange tran...  7.325524     0.673939
668     1259                            Drama Romance           6.9       239.0   8.593087                  Notes on a Scandal  One woman's secret is another woman's power. O...  A veteran high school teacher befriends a youn...  6.729008     0.545824
654     1213                     Thriller Crime Drama           7.1       790.0   11.65502             The Talented Mr. Ripley       How far would you go to become someone else?  Tom Ripley is a calculating young man who beli...  6.994499     0.504750
348      527                                    Drama           7.6       106.0   4.025276                  Once Were Warriors  A family in crisis, a life in chaos... Nothing...  A drama about a Maori family lving in Auckland...  6.922460     0.498243
```
