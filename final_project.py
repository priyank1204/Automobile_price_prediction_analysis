#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING NECESSARY LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# In[2]:


import os


# In[3]:


os.chdir(r"C:\Users\poorvi\Desktop\auto_project")


# # NOW READING THE DATASET OF AUTOMOBILES

# OUR DATA HAS SOME MISSING VALUES IN THE FORM OF "?" SO NOW USING na_values ARGUMENT TO RECOGNIZE TRHEM AS NP.NAN

# In[4]:


auto_data = pd.read_csv(r"C:\Users\poorvi\Desktop\INEURON\imports-85.data",header= None,na_values = '?') 
pd.set_option('display.max_columns',29) ## FOR SHOWING FIRST 10 AND LAST 15 COLUMNS
##pd.set_option("display.max_rows",50)


# In[5]:


auto_data.head()


# OUR DATASET HAS NO COLUMN INDEXES , SO NOW SETTING COLUM INDEXES FOR THE DATA

# In[6]:


columns  = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style",'drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinder','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']


# In[7]:


auto_data.shape


# import sweetviz as svz

# x = svz.analyze(auto_data)

# x.show_html('Advertising.html')

# #### ---->OUR DATASET HAS 205 RECORDS AND FEATURES IN THE DATASET ARE 26 IN NUMBER

# ## SETTING COLUMN INDEXES IN THE DATSET WHICH WERE NOT THERE IN IT

# In[8]:


auto_data.columns = columns


# In[9]:


auto_data.head()


# In[10]:


auto_data.columns


# In[11]:


auto_data.to_csv("myfile.csv",index =False)


# ##### DATA TYPES OF EACH COLUMN

# In[12]:


auto_data.dtypes 


# ##### NOW WE WILL GATHER SOME INFORMATION ABOUT THE DATA USING THE INFO() COMMAND TO SEEE COUNTOF NULL VALUES OR DATA TYPES OR COLUMNS NAME

# In[13]:


auto_data.info()

##FROM THE ABOVE TABLE WE HAVE SEEN THAT THERE ARE NO NULL VALUES AS SUCH IN OUR DATA
# ##### NOW WE WILL GAIN MORE INFORMATION USING THE DESCRIBE() FUNCTION FOR THE NUMERICAL ,WE WILL SEEE THE 
# MEDAIN
# 
# MODE
# 
# FIRST QUARTILE,
# 
# SECOND AND THIRD QUARTILE 
# 
# AND ALSO THE MAXIMUM
# 
# AND MINIMUM VALUES OF THE COLUMNS
# 

# In[14]:


auto_data.describe()


# #### SIMILARLY FOR THE NUMERICLA DATASET WE WILL GATHER INFO FOR THE CATEGORICAL COLUMNS 
#  LIKE,
#  
#  COUNT OF VALUES
#  
#  UNIQUE VALUES
#  
#  TOP VALUES
#  
#  MOST FREQUENT VALUE

# In[15]:


auto_data.describe(exclude=[np.number])


# #### NOW WE WILL CHANGE THE DATATYPE OF THE COLUMNS IN THE REQUIRED DATATYPE

# As we can see in our data many of the columns are not in the correct data type so we will change them like normalised-losses,bore,stroke,horsepower etc

# In[16]:


auto_data["normalized-losses"] = auto_data["normalized-losses"].astype("float")
auto_data["bore"] = auto_data["bore"].astype("float")
auto_data["stroke"] = auto_data["stroke"].astype("float")
auto_data["horsepower"] = auto_data["horsepower"].astype("float")
auto_data["peak-rpm"] = auto_data["peak-rpm"].astype("float")


# In[17]:


auto_data["price"] = auto_data["price"].astype("float")


# # SHOWING NULL VALUES COUNT IN EACH COLUMN

# now we will count missing values in each columns so we can handle them further

# In[18]:


print("Missing values in Each column\n")
print("#"*25 + "\n")
print("feature-name" + "             " + "count")
auto_data.isnull().sum()


# # SHOWING COUNT OF ALL NAN VALUES OF THOSE COLUMNS WHICH CONTAINS NAN VALUES 

# In[19]:


print("Missing values in Each column\n")
print("#"*25 + "\n")
print("feature-name" + "             " + "count")
auto_data.isnull().sum()[auto_data.isnull().sum()>0]


# In[20]:


auto_data.dtypes


# # NOW WE WILL HANDLE THE MISSSING VALUES IN THE DATA SET

# #### WE WILL IMPUTE THE MEDIAN IN THE PLACE OF NAN VALUES FOR THE FLOAT VALUES COLUMN 

# #### now we will impute median in the numerical columns for which we will make an array of that 

# #### NOW WE WILL EXTRACT THE DATASET OF ONLY NUMERICAL COLUMNS FROM THEM

# In[21]:


num_col = auto_data.select_dtypes(include=np.number).columns


# In[22]:


num_col = num_col.drop("price")


# In[23]:


num_col


# ## IMPUTING THE MISSING VALUES

# NOW WE WILL IMPUTE THE MISSING VALUES WITH THE CORRECT VALUES
# WE WILL USE THE SIMPLEIMPUTER CLASS TO IMPUTER THE MISSING VALUES
# 
# 
# 1)FOR THE NUMERICAL COLUMNS WE WILL USE:
# 
# 
# -------> STRATEGY = "MEDIAN"
# 
# 2) FOR THE CATEGORICAL COLUMNS :
# 
# 
# ------------->STRATEGY  = "MOST_FREQUENT"
# 
# 3)FOR THE TARGET VARIABLE 'PRICE' , AS WE DO NOT KNOW THESE W\VALUES SO THEY WILL NOT  HELP IN MODEL BUILDING SO WILL DROP THOSE ROW WHERE MISSING VALUES ARE FOUND

# In[24]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(auto_data[num_col])
auto_data[num_col] = imputer.transform(auto_data[num_col])


# In[25]:


auto_data.head()


# In[26]:


print("Missing values in Each column\n")
print("#"*25 + "\n")
print("feature" + "             " + "count")
auto_data.isnull().sum()[auto_data.isnull().sum()>0]


# ##### NOW WE WILL HANDLE OUR TARGET VARIABLE 'PRICE'

# In[27]:


auto_data.dropna(subset=["price"],axis = 0,inplace=True)


# In[28]:


print("Missing values in Each column\n")
print("#"*25 + "\n")
print("feature" + "             " + "count")
auto_data.isnull().sum()[auto_data.isnull().sum()>0]


# In[29]:


auto_data.shape


# ##### NOW CHANGING THE NAN VALUES PRESENT IN THE CATEGORICAL COLUMN WITH THE MOST OCCURED VALUES 

# In[30]:


y = auto_data["num-of-doors"].mode()


# In[31]:


y


# In[32]:


y[0]


# In[33]:


auto_data["num-of-doors"].replace(np.nan,y[0],inplace = True)


# In[34]:


auto_data.isnull().sum()[auto_data.isnull().sum()>0]


# In[35]:


auto_data.head()


# # NOW AGAIN CHECKING COLUMNS IN WHICH NAN VALUES ARE PRESENT

# In[36]:


auto_data.isnull().sum()[auto_data.isnull().sum()>0]


# In[37]:


auto_data.head()


# # WE HAVE TO RESET INDEXES OF ALL ROWS AS WE HAVE DELETED SOME COLUMNS

# In[38]:


auto_data.reset_index(drop =True,inplace = True)


# In[39]:


auto_data.head()


# In[40]:


plt.subplots(figsize=(10,6))
ax=auto_data['make'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xticks(rotation='vertical')
plt.xlabel('Car maker',fontsize=20)
plt.ylabel('Number of Cars',fontsize=20)
plt.title('Cars Count By Manufacturer',fontsize=30)
ax.tick_params(labelsize=15)
#plt.yticks(rotation='vertical')
plt.savefig("cars_counts.png",)
plt.show()


# #### FROM THE ABOVE COUNT PLOTS OF ALL CAR makeRS WE HAVE GATHERED THAT:
#     
#     1)TOYOTA ,nissan and mazda has the highest number of cars in the data
#     
#     2)mercury isuzu and renault has the minimum number of cars very less i.e 1,2,2
#     
#     3)Japaneese cars has most share in the data i.e toyota ,nissan

# In[41]:


import plotly.express as px



# In[42]:


fig = px.pie(auto_data,names = 'drive-wheels',title = "share of all drive wheels",height=350)
fig.show()


# In[43]:


fig = px.pie(auto_data,height=350,names = 'num-of-doors',title = "share of all NUM-OF-DOORS")
fig.show()


# In[44]:


fig = px.pie(auto_data,names = 'engine-type',title = "share of all engine-types",height=350)
fig.show()


# In[45]:


fig = px.pie(auto_data,names = 'engine-location',title = "share of all engine-locations",height=350)
fig.show()


# In[46]:


fig = px.pie(auto_data,names = 'body-style',title = "share of all body-style",height=350)
fig.show()


# In[47]:


fig = px.pie(auto_data,names = 'fuel-type',title = "share of all fuel types",height=350)
fig.show()


# In[48]:


fig = px.pie(auto_data,names = 'aspiration',title = "share of all aspiration types",height=350)
fig.show()


# #### FROM THE ABOVE PIE CHARTS IT IS CLEAR THAT :
# 
# 1) most cars has the fwd types of drive-wheels which has 58.27% share
# 
# 2)most of the cars has four doors  approx 57.2% cars has 4 doors
# 
# 3)aspiration of maximum cars is standard beacuse it is very outdated data when turbo aspirated cars were not used more
# 
# 4)body-syle of maximum cars is sedan and hatchback
# 
# 5)engine location is mostly preferred to frontrather than rear
# 
# 6)engine types is ohc mostly preferrred 72%
# 
# 7) maximum no of cars has the fuel used as gas and very less uses diesel

# ### NOW WE WILL DROP THE make COLUMN AS IT IS NOT A PREDICTOR OF THE PRICE ,,,,,,,, COMPANY NAME  OF THE CAR DOES NOT DECIDES ITS PRICE

# In[49]:


auto_data.drop("make",axis =1,inplace=True)


# In[50]:


auto_data


# # NOW WE CAN SAVE THIS CLEAN DATASET INTO AN CSV FILE FOR FURTHUR USES

# In[51]:


auto_data.to_csv(r"C:\Users\poorvi\Desktop\INEURON\auto_clean.csv")


# In[52]:


auto_data.dtypes


# In[53]:


auto_data.head()


# # NOW WE WILL NORMALIZE SOME COLUMNS THAT MEANS WE WILL make THEIR MEAN TO 0 AND VARIANCE TO 1

# In[54]:


from scipy import stats


# In[55]:


auto_data.head()


# In[56]:


auto_data.dtypes


# In[57]:


auto_data.corr()


# # NOW WE FIND CORRELATION BETWEEN PRICE VALUE AND NUMERICAL VARIABLES USING PEARSONS CORRELATION COEFFICIENT

# POSITIVE PEARSON CORREALTION COEFFICIENT MEANS THAT THERE IS POSITIVE CORRELATION WHILE NEGATIVE CORRALATION COEFFICIENT MEANS NEGATIVE CORREALTION 
# 
# WE WILL ALSO SEE THE P-VALUE FOR THIS PURPOSE : -
# 
# 
# p-value is < 0.001 we say there is strong evidence that the correlation is significant.
# 
# the p-value is < 0.05: there is moderate evidence that the correlation is significant.
# 
# the p-value is < 0.1: there is weak evidence that the correlation is significant.
# 
# the p-value is > 0.1: there is no evidence that the correlation is significant.
# 
# 

# In[58]:


from scipy import stats


# In[59]:


x = stats.pearsonr(auto_data["height"],auto_data["price"])
print("correlation coefficient between height and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[60]:


x =stats.pearsonr(auto_data["normalized-losses"],auto_data["price"])
print("correlation coefficient between noromalized -losses and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[61]:


x =stats.pearsonr(auto_data["wheel-base"],auto_data["price"])
print("correlation coefficient between wheel-base and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[62]:


x= stats.pearsonr(auto_data["length"],auto_data["price"])
print("correlation coefficient between length and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[63]:


x =stats.pearsonr(auto_data["width"],auto_data["price"])
print("correlation coefficient between width and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[64]:


x =stats.pearsonr(auto_data["curb-weight"],auto_data["price"])
print("correlation coefficient between curb-weight and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[65]:


x =stats.pearsonr(auto_data["engine-size"],auto_data["price"])
print("correlation coefficient between engine-size and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[66]:


x =stats.pearsonr(auto_data["bore"],auto_data["price"])
print("correlation coefficient between bore and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[67]:


x =stats.pearsonr(auto_data["stroke"],auto_data["price"])
print("correlation coefficient between stroke and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[68]:


x =stats.pearsonr(auto_data["compression-ratio"],auto_data["price"])
print("correlation coefficient between compression-ratio and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[69]:


x =stats.pearsonr(auto_data["horsepower"],auto_data["price"])
print("correlation coefficient between horsepower and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[70]:


x=stats.pearsonr(auto_data["peak-rpm"],auto_data["price"])
print("correlation coefficient between peak-rpm and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# In[71]:


x = stats.pearsonr(auto_data["city-mpg"],auto_data["price"])
print("correlation coefficient between city-mpg and price is ---->",x[0], "\n, p-value is---> ", x[1])


# In[72]:


x =stats.pearsonr(auto_data["highway-mpg"],auto_data["price"])
print("correlation coefficient between highway-mpg and price is ---->",x[0], "\n\n, p-value is---> ", x[1])


# Observing the p vslues of the above correlation
# The top features who have significant relation with the price variable are:
# 1. length,
# 2. width,
# 3. curb-weight,
# 4. engine-size,
# 5. bore,
# 6. horsepower,
# 7. city-mpg,
# 8. highway-mpg

# #### PAIR PLOT OF THE HIGHLY CORRELATED FEATURES WITH THE TARGET VARIABLE

# In[73]:



ax = sns.pairplot(auto_data[["length","width", "curb-weight","engine-size","horsepower","highway-mpg","price","bore","city-mpg"]], palette='dark',diag_kind="hist") 
plt.savefig("scatter_plots.png")


# #### FROM THE ABOVE PAIR PLOT WE CAN FIND OUT THAT:
# 
# 1)the car with the high price has low mileage,beacusae expensive cars focuses more on luxury rather than mileage
# 
# 2)more the horsepower values of the car more is its price
# 
# 3)Length,width,curb-weight ,engine-size are highly positively correlated with the price variable
# 

# In[74]:


auto_data.head(2)


# In[75]:


cat_data = auto_data.select_dtypes(include=[np.object])
cat_data


# In[76]:


for i in cat_data:
    x = cat_data[i].value_counts()
    print("uniques values in " ,i ,"is--->\n",x,"\n")


# #### now we will find out the categorical features from the actual data

# #### THEN we will apply ANOVA test in the to find out the most significant features among them

# In[77]:


cat_data = auto_data.select_dtypes(exclude=[np.number])


# In[78]:


cat_data


# In[79]:


cat   = cat_data.columns
cat
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['price'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

cat_data['price'] = auto_data.price.values
k = anova(cat_data) 
k['disparity'] = np.log(1/k['pval'].values) 
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt.savefig("anova_test.png")
plt 


# FROM THE ABOVE CHART IT IS FOUND THAT THE SIGNIFICANT FEATURES FOR CAR PRICE PREDICTION ARE
# 
# num-of-cylinder
# 
# drive-wheels
# 
# fuel-system
# 
# engine-type

# ###### here we can seee that the 'twelve' and 'three' groups in the num of cylinder columns is not very significant thus we can replace them with the least frequent groups i.e 'eight'

# In[80]:


cat_data["num-of-cylinder"].replace({"twelve":"eight","three":'eight',"two":"eight"},inplace = True)


# In[81]:


cat_data["num-of-cylinder"].value_counts()


# In[82]:


cat_data["drive-wheels"].replace({"4wd","rwd"},inplace = True)


# In[83]:


cat_data["num-of-cylinder"].replace({"4bbl":"spdi","spfi":"spdi",'mfi':"spdi"},inplace = True)


# In[84]:


target  =auto_data["price"]


# In[85]:


target


# # NOW WE WILL  EXTRACT THE IMPORTANT FEATURES FROM THE DATA FOR FURTHER MODEL BUILDING

# In[86]:


auto_data.columns


# In[87]:


imp_feat  = auto_data[["length","width",'horsepower','curb-weight',"engine-size","city-mpg","highway-mpg",'drive-wheels','num-of-cylinder',]]


# In[88]:


imp_feat


# In[89]:


imp_feat.describe()


# ##    CHECKING AND HANDLING OUTLIERS

# In[90]:


for column in imp_feat.select_dtypes(include=[np.number]):
    plt.figure()
    imp_feat.boxplot([column])
    plt.savefig("{}_boxplots.png".format(column)) 
    plt.close()


# #### FROM THE ABOVE BOX PLOTS OF THE NUMERICAL COLUMNS WE CAN SEE THAT THERE ARE MANY OUTLIERS IN THE ["WIDTH,"ENGINE SIZE"] COLUMNS SO WE CAN HANDLE THEM BY IMPUTING A SUITABLE VALUE INSTEAD OF THEIR PLACE

# #### IN THE ENGINE-SIZE COLUMN WE CAN SEE THAT WE HAVE OUTLIERS GREATER THAN 200
# #### IN THE WIDTH COLUMN WE CAN SEE THAT WE HAVE OUTLIERS GREATER THAN 2

# #### WE WILL  REPLACE ALL THE OUTLIERS WITH THE MAX VALUES OF THESE COLUMNS

# NOW WE WILL FIRSTLY KNOW THE INDEXES WHERE THESE OUTLIERS ARE LOCATED

# #### FOR WIDTH 

# In[91]:


loc= imp_feat[imp_feat['width']>2].index.tolist() 
imp_feat["width"].iloc[loc]


# In[92]:


imp_feat["width"].iloc[loc] =np.quantile(imp_feat["width"],0.75)
imp_feat["width"].iloc[loc]


# #### FOR ENGINE-SIZE

# In[93]:


loc= imp_feat[imp_feat["engine-size"]>200].index.tolist()


# In[94]:


loc


# In[95]:


imp_feat["engine-size"].iloc[loc] =np.quantile(imp_feat["engine-size"],0.75)


# ### IN OUR DATA WE HAVE COLUMNS SUCH AS ["CITY-MPG'HIGHWAY-MPG'] AND ['LENGTH','WIDTH'] WHICH ARE VERY CORRELATED TO EACH OTHER SO WE CAN REMOVE THEM BY SOME OPERATIONS

# #### -------------------> 
# IN THE CITY-MP AND HIGHWAY-MPG WE WILL make A NEW FEAUTRE OF THEIR DIFFERNECE AND DROP BOTH OF THEM

# In[96]:


imp_feat["miles"] = imp_feat["city-mpg"]-imp_feat["highway-mpg"]
imp_feat.drop("highway-mpg",axis =1,inplace=True)
imp_feat.drop("city-mpg",axis =1,inplace=True)


# #### --------------->
# IN THE LENGTH AND WIDTH COLUMN WE WILL DROP BOTH OF THE SE FEATURES BY ADDING A NEW FEATURE CALLED AREA WHICH IS THE PRODUCT OG BOTH OF THEM

# In[97]:


imp_feat["area"] = imp_feat["length"]*imp_feat["width"]
imp_feat.drop("length",axis =1,inplace=True)
imp_feat.drop("width",axis =1,inplace=True)


# In[98]:


imp_feat.columns


# In[99]:


from sklearn.preprocessing import MinMaxScaler


# In[100]:


mc = MinMaxScaler()


# In[101]:


num_col = imp_feat.select_dtypes(include=[np.number]).columns


# In[102]:


mc.fit(imp_feat[num_col])


# In[103]:


num_col


# In[104]:


imp_feat


# In[105]:


imp_feat[num_col] = mc.transform(imp_feat[num_col])


# In[106]:


imp_feat


# In[107]:


encoded_data = pd.get_dummies(imp_feat,drop_first=True)


# In[108]:


encoded_data["price"] = auto_data["price"]


# In[109]:


encoded_data


# In[110]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(encoded_data.drop("price",axis=1),encoded_data["price"])


# In[111]:


model.feature_importances_
x = pd.Series(model.feature_importances_,index=encoded_data.drop("price",axis=1).columns)


# In[112]:


x.sort_values().index


# In[113]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[114]:


from sklearn.metrics import mean_squared_error


# In[115]:


x_train,x_test,y_train,y_test = train_test_split(encoded_data.drop("price",axis =1),encoded_data["price"],test_size = 0.25,random_state = 3)


# In[116]:


x_train.columns


# In[117]:


x_train.head()


# In[118]:


x_test.head()


# In[119]:


y_train.head()


# In[120]:


y_test.head()


# #### NOW LETS DEFINE TWO FUNCTIONS SHICH WILL CALCULATE R2 ,ADJ_R2 AND ERROR OF THE PREDICTION OF DIFFERENT MODELS

# In[121]:


##FOR R2 AND ADJUSTED_R2 
def r_and_adj_r(x,y,model):
    n = x.shape[0]
    p = x.shape[1]
    r = model.score(x,y)
    adjusted_r2 = 1-(1-r)*(n-1)/(n-p-1)
    return "r2 =",r , "adj_r2",adjusted_r2


# In[122]:


def DistributionPlot(Redfunction,Bluefunction,RedName,BlueName,Title):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))
    
    ax1  = sns.distplot(Redfunction, hist=False, color="r", label=RedName)
    ax2  = sns.distplot(Bluefunction, hist=False, color="b", label=BlueName, ax=ax1)
    
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.savefig("{}.png".format(Title))
    plt.show()
    plt.close()
    


# In[123]:


from sklearn.ensemble import RandomForestRegressor


# In[124]:


rf = RandomForestRegressor()


# In[125]:


rf.fit(x_train,y_train)


# In[126]:


ypred1 = rf.predict(x_test)


# In[127]:


ypred1


# In[128]:


r_and_adj_r(x_train,y_train,rf)


# In[129]:


r_and_adj_r(x_test,y_test,rf)


# In[130]:


DistributionPlot(rf.predict(x_test),y_test,"pred","actual","predicted-actual_by_randomforest")


# from sklearn.linear_model import Lasso,LassoCV

# In[131]:


import xgboost


# In[132]:


xgr =xgboost.XGBRegressor(n_threads = -1)


# In[133]:


xgr.fit(x_train,y_train)


# In[134]:


ypred4 = xgr.predict(x_test)


# In[135]:


ypred4


# In[136]:


r_and_adj_r(x_train,y_train,xgr)


# In[137]:


r_and_adj_r(x_test,y_test,xgr)


# In[138]:


DistributionPlot(ypred4,y_test,"pred","actual","predicted-actual_by_xgboost")


# In[139]:


from sklearn.model_selection import GridSearchCV


# In[140]:


params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}


# In[141]:


grid = GridSearchCV(xgr, params)
grid.fit(x_train,y_train)


# In[142]:


from sklearn.metrics import r2_score

print(r2_score(y_test, grid.best_estimator_.predict(x_test))) 


# In[143]:


y = grid.best_estimator_


# In[144]:


y_pred5 = grid.best_estimator_.predict(x_test)


# In[145]:


y_pred5


# In[146]:


DistributionPlot(y_pred5,y_test,'pred','actual',"prediction by xgboost after tuning")


# In[147]:


#Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV


# In[148]:



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[149]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[150]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, random_state=42, n_jobs = 1)


# In[151]:


rf_random.fit(x_train,y_train)


# In[152]:


pred = rf_random.predict(x_test)


# In[153]:


rf_random.score(x_test,y_test)


# In[154]:


sns.distplot(y_test-pred)


# In[155]:


DistributionPlot(pred,y_test,'pred','actual','predictoin by randomforest after tuning')


# In[156]:


plt.scatter(y_test,pred)


# In[157]:



import joblib


joblib.dump(rf_random,'model.pkl')

