# Recommendations via Optimization  
__Goal__ : Make good recommendations  
- Quantify goodness using __RMSE__ : __Lower RMSE → better recommendations__  
- Let's set build a system such that it works well on known (user, item) ratings  
  And hope the system will also predict well the unknown ratings.   

---
 
## ContentBasedFiltering.py  
Implement of basic level of content based filtering.  
CosineSimilarity(__genres__)  
```bash
python ContentBasedFiltering.py --title '영화제목'
```
### Example  
```bash
python ContentBasedFiltering.py --title 'The Dark Knight Rises'
```

### Results  
![스크린샷 2021-07-09 오전 3 10 05](https://user-images.githubusercontent.com/37684658/124970682-2ca00200-e063-11eb-8509-4e2e4cbd79bb.png)

---  
