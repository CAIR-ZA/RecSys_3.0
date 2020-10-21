#Naive Bayes CF Model-
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB 
from surprise import SVD
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split,GridSearchCV

#Downloading the dataset - Movielens.
!wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip ml-1m.zip

ratings = pd.read_csv('./ml-1m/ratings.dat', engine='python',
                          sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies = pd.read_csv('./ml-1m/movies.dat', engine='python',
                          sep='::', names=['MovieID', 'Title', 'Genre'])
users = pd.read_csv("./ml-1m/users.dat", sep="::", names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'] )

ratings_user = pd.merge(ratings,users, on=['UserID'])
ratings_movie = pd.merge(ratings,movies, on=['MovieID'])

data = pd.merge(ratings_user,ratings_movie,
                       on=['UserID', 'MovieID', 'Rating'])[['MovieID', 'Title', 'UserID', 'Age', 'Gender', 'Occupation', "Rating"]]
data.head(10)

#Feature engineering:
data.shape
data.info()

#determine categorical variables:
categorical = [var for var in data.columns if data[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

for var in categorical: 
    print(data[var].value_counts())
    
for var in categorical: 
    print(data[var].value_counts()/np.float(len(data)))

#determine numerical variables:
numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)
data[numerical].isnull().sum()

#Compute the Naive bayes classifier algorithm:
benchmark = []
for algorithm in [SVD(), GaussianNB() ]:
    # Perform k-cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=10, verbose=False)   
    # attain results & change name of algorithm:
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],index=['Algorithm']))
    benchmark.append(tmp)

Results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')


gnb=SVD()
predict =gnb.fit(iid,uid)
    
df_pred = pd.DataFrame(predict, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_pred['Iu'] = df_pred.uid.apply(get_Iu)
df_pred['Ui'] = df_pred.iid.apply(get_Ui)
df_pred['err'] = abs(df_pred.est - df_pred.rui)

df_pred.head()

best_pred = df_pred.sort_values(by='err')[:9]
worst_pred = df_pred.sort_values(by='err')[-9:]

#Calculate TP,FP,TN,FN at every threshold level (0.0 - 5.0)

final = []

for threshold in np.arange(0, 5.5, 0.5):
  tp=0,fn=0,fp=0,tn=0
  temp = []

  for uid, _,r_true , est, _ in predict:
    if(r_true>=threshold):
      if(est>=threshold):
        tp = tp+1
      else:
        fn = fn+1
    else:
      if(est>=threshold):
        fp = fp+1
      else:
        tn = tn+1   

    if tp == 0:
      precision = 0
      recall = 0
      f1 = 0
    else:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * (precision * recall) / (precision + recall)  

  temp = [threshold, tp,fp,tn ,fn, precision, recall, f1]
  final.append(temp)

results = pd.DataFrame(final)
results.rename(columns={0:'Threshold', 1:'TP', 2: 'FP', 3: 'TN', 4:'FN', 5: 'Precision', 6:'Recall', 7:'f1'}, inplace=True)
results

def precision_k_recall_k(predict, k, threshold):
  # First map the predictions to each user.
    user_Est = defaultdict(list)
    for uid, _, r_true, est, _ in predict:
        user_Est[uid].append((est, r_true))

    precisions = dict()
    recalls = dict()
    for uid, UserRatings in user_Est.items():

        #Sort user ratings by estimated value
        UserRatings.sort(key=lambda x: x[0], reverse=True)

        #No. of relevant items
        n_rel = sum((r_true >= threshold) for (_, r_true) in UserRatings)

        #No. of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in UserRatings[:k])

        #No. of relevant and recommended items in top k
        n_relrec_k = sum(((r_true >= threshold) and (est >= threshold))
                              for (est, r_true) in UserRatings[:k])

        # Precision@K: 
        precisions[uid] = n_relrec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: 
        recalls[uid] = n_relrec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls
    
results=[]
for i in range(2, 10):
    precisions, recalls = precision_k_recall_k(predict, k=i, threshold=2.5)
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    results.append({'K': i, 'Precision': prec, 'Recall': rec})  
results

Rec=[]
Precision=[]
Recall=[]
for i in range(0,9):
    Rec.append(results[i]['K'])
    Precision.append(results[i]['Precision'])
    Recall.append(results[i]['Recall'])

from matplotlib import pyplot as plt
plt.plot(Rec, Precision)
plt.xlabel('No. of Recommendations')
plt.ylabel('MAP@k')
plt2 = plt.twinx()
plt2.plot(Rec, Recall, 'r')
plt.ylabel('MAR@k')
for tl in plt2.get_yticklabels():
    tl.set_color('r')
    
def get_all_predict(predict):
    
    
    topN = defaultdict(list)    
    for uid, iid, r_true, est, _ in predict:
        topN[uid].append((iid, est))

    for uid, UserRatings in topN.items():
        UserRatings.sort(key=lambda x: x[1], reverse=True)
    return topN

all_pred = get_all_predict(predict)

#To acquiring top 5 movies' reommendations
n = 5
for uid, UserRatings in all_pred.items():
    UserRatings.sort(key=lambda x: x[1], reverse=True)
    all_pred[uid] = UserRatings[:n]