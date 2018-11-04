
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series,DataFrame


# In[2]:


train=pd.read_csv('train.csv')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


import numpy as np


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#Gender
sns.countplot('Sex',data=train)


# In[8]:


sns.countplot('Pclass',data=train,hue='Sex')


# In[9]:


# women and child will be given preference

def gender(passenger):
    age,sex=passenger
    if age<16:
        return 'child'
    else:
        return sex


# In[10]:


train['Person']=train[['Age','Sex']].apply(gender,axis=1)


# In[11]:


train[0:10]


# In[12]:


sns.countplot('Pclass',data=train,hue='Sex')


# In[13]:


train['Age'].hist(bins=70)


# In[14]:


train['Age'].mean()


# In[15]:


train['Person'].value_counts()


# In[16]:


fig=sns.FacetGrid(train,hue='Person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)
oldest=train['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[17]:


train.head()


# In[18]:


deck=train['Cabin'].dropna()


# In[19]:


deck.head()


# In[20]:


levels=[]

for level in deck:
    levels.append(level[0])
    
cabin_df=DataFrame(levels)
cabin_df.columns=['Cabin']
sns.countplot('Cabin',data=cabin_df,palette='winter_d')


# In[21]:


cabin_df=cabin_df[cabin_df.Cabin!='T']


# In[22]:



sns.countplot('Cabin',data=cabin_df,palette='summer')


# In[23]:


train.head()


# In[24]:


sns.countplot('Embarked',data=train,hue='Pclass')


# In[25]:


train['Alone']=train.SibSp+train.Parch


# In[26]:


train['Alone']


# In[27]:


train['Alone'].loc[train['Alone']>0]='With Family'
train['Alone'].loc[train['Alone']==0]='Alone'


# In[28]:


train.head()


# In[29]:


sns.countplot('Alone',data=train,palette='Blues')


# In[30]:


train['Survivor']=train.Survived.map({0:'no',1:'yes'})


# In[31]:


sns.countplot('Survivor',data=train)


# In[32]:


sns.factorplot('Pclass','Survived',hue='Person',data=train)


# In[33]:


sns.lmplot('Age','Survived',hue='Pclass',data=train,palette='winter')


# In[34]:


generations=[10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=train,x_bins=generations)


# In[35]:


sns.lmplot('Age','Survived',hue='Sex',data=train)


# In[36]:


train.head()


# In[37]:


sns.lmplot('Age','Survived',data=train)


# In[38]:


sns.countplot('Survived',data=train,hue='Person')


# In[39]:


train.head()


# In[40]:


sns.countplot('Alone',data=train)


# In[41]:


sns.countplot('Alone',hue='Survived',data=train)


# In[42]:


#Feature Engineering


# In[43]:


train.head()


# In[44]:


train.info()


# In[45]:


train.isnull().sum()


# In[46]:


# We can see Age and Cabin value is missing in many rows


# In[47]:


from IPython.display import Image
Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")


# In[48]:


test=pd.read_csv('Test.csv')


# In[49]:


test


# In[50]:


dataset=[train,test]


# In[51]:


#AGE

map_sex={"male":0,"female":1}
for i in dataset:
    i['Sex']=i['Sex'].map(map_sex)


# In[52]:


train.head()


# In[53]:


test.head()


# In[54]:


#NAME

def title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return unknown



# In[55]:


train['Title']=train['Name'].apply(title)
test['Title']=test['Name'].apply(title)


# In[56]:


train.head()


# In[57]:


test.head()


# In[58]:


train['Title'].value_counts()


# In[59]:


test['Title'].value_counts()


# In[60]:


'''Title map
Mr : 0
Miss : 1
Mrs: 2
Others: 3'''


# In[61]:


map_title={ "Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3}


# In[62]:


for i in dataset:
    i['Title']=i['Title'].map(map_title)


# In[63]:


train.head()


# In[64]:


test.head()


# In[65]:


#We have converted names in titles(numerical values)


# In[66]:


train.drop('Name',axis=1)


# In[67]:


train.drop('Survivor',axis=1)


# In[68]:


train.head()


# In[69]:


train.drop('Survivor',axis=1,inplace=True)


# In[70]:


train.head()


# In[71]:


train.info()


# In[72]:


train.drop('Person',axis=1,inplace=True)


# In[73]:


train.head()


# In[74]:


# Drop cabin as there are lot of null values


# In[75]:


test.drop('Cabin',axis=1,inplace=True)


# In[76]:


train.drop('Cabin',axis=1,inplace=True)


# In[77]:


train.head()


# In[78]:


train.drop('PassengerId',axis=1,inplace=True)


# In[79]:


test.drop('PassengerId',axis=1,inplace=True)


# In[80]:


train.head()


# In[81]:


test.head()


# In[82]:


test['Alone']=test['SibSp']+test['Parch']


# In[83]:


train['Alone']=train['SibSp']+train['Parch']


# In[84]:


train.head()


# In[85]:


test.drop('SibSp',axis=1,inplace=True)
train.drop('SibSp',axis=1,inplace=True)


# In[86]:


test.drop('Parch',axis=1,inplace=True)
train.drop('Parch',axis=1,inplace=True)


# In[87]:


train.head()


# In[88]:


train.drop('Name',axis=1,inplace=True)


# In[89]:


train.head()


# In[90]:


test.head()


# In[91]:


test.drop('Name',axis=1,inplace=True)


# In[92]:


test.head()


# In[93]:


train['Alone'].loc[train['Alone']>0]=1


# In[94]:


train['Alone'].loc[train['Alone']==0]=0


# In[95]:


test['Alone'].loc[test['Alone']>0]=1


# In[96]:


test['Alone'].loc[test['Alone']==0]=0


# In[97]:


train.head()


# In[98]:


test.head()


# In[99]:


train.info()


# In[100]:


sns.countplot('Embarked',data=train)


# In[101]:


test.info()


# In[102]:


#EMBARKED
#Since 'S' is most occured ,we will fill all the null values with 'S'

train['Embarked']=train['Embarked'].fillna('S')


# In[103]:


train.info()


# In[104]:


train['Title']=train['Title'].fillna('0')


# In[105]:


train.info()


# In[106]:


train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)


# In[107]:


train.head()


# In[108]:


test.head()


# In[109]:


#FARE
#Missing fare value in our test data
test['Fare'].fillna(test['Fare'].median(),inplace=True)


# In[110]:


test.info()


# In[111]:


fig=sns.FacetGrid(train,hue='Survived',aspect=1)

fig.map(sns.kdeplot,'Fare',shade=True)
maxi=train['Fare'].max()

fig.set(xlim=(0,20))

fig.add_legend()


# In[112]:


fig=sns.FacetGrid(train,hue='Survived',aspect=1)

fig.map(sns.kdeplot,'Fare',shade=True)
maxi=train['Fare'].max()

fig.set(xlim=(20,50))

fig.add_legend()


# In[113]:


fig=sns.FacetGrid(train,hue='Survived')

fig.map(sns.kdeplot,'Fare',shade=True)
maxi=train['Fare'].max()

fig.set(xlim=(50,maxi))

fig.add_legend()


# In[114]:


train.loc[train['Fare']<=17,'Fare']=0
train.loc[(train['Fare'] > 17) & (train['Fare'] <= 30), 'Fare'] = 1,
train.loc[(train['Fare'] > 30) & (train['Fare'] <= 100), 'Fare'] = 2,
train.loc[ train['Fare'] > 100, 'Fare'] = 3




# In[115]:


train.head()


# In[116]:


train['Fare'].value_counts()


# In[117]:


test.head()


# In[118]:


train['Age'].fillna(train['Age'].mean(),inplace=True)


# In[119]:


test['Age'].fillna(test['Age'].mean(),inplace=True)


# In[120]:


train.info()


# In[121]:


test.info()


# In[122]:


test.head()


# In[123]:


train.head()


# In[124]:



train.loc[train['Age']<=16,'Age']=0,
train.loc[(train['Age']>16)&(train['Age']<=26),'Age']=1,
train.loc[(train['Age']>26)&(train['Age']<=38),'Age']=2,
train.loc[(train['Age']>38)&(train['Age']<=62),'Age']=3,
train.loc[train['Age']>62,'Age']=4


# In[125]:


train.head()


# In[126]:



	
test.loc[test['Age']<=16,'Age']=0,
test.loc[(test['Age']>16)&(test['Age']<=26),'Age']=1,
test.loc[(test['Age']>26)&(test['Age']<=38),'Age']=2,
test.loc[(test['Age']>38)&(test['Age']<=62),'Age']=3,
test.loc[test['Age']>62,'Age']=4


# In[127]:


test.head()


# In[128]:


test.loc[test['Fare']<=17,'Fare']=0
test.loc[(test['Fare'] > 17) & (test['Fare'] <= 30), 'Fare'] = 1,
test.loc[(test['Fare'] > 30) & (test['Fare'] <= 100), 'Fare'] = 2,
test.loc[ test['Fare'] > 100, 'Fare'] = 3


# In[129]:


test.head()


# In[130]:


train.head()


# # Feature Engineering Done

# In[132]:





# In[153]:


x['Embarked'].value_counts()


# In[145]:


y.head()


# In[155]:


train.loc[train['Embarked']=='S','Embarked']=0
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2


# In[159]:


test.loc[test['Embarked']=='S','Embarked']=0
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2


# In[136]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn


# In[165]:


x=train.drop('Survived',axis=1)


# In[166]:


y=train['Survived']


# In[167]:


x.head()


# In[168]:


x.info()


# In[183]:


clf=RandomForestClassifier(n_estimators=50)
clf.fit(x,y)

y_pred=clf.predict(test)

clf.score(x,y)


# In[179]:


svc=SVC()
svc.fit(x,y)

y_pred_1=svc.predict(test)
svc.score(x,y)


# In[181]:


log=LogisticRegression()
log.fit(x,y)
y_pred_2=log.predict(test)
log.score(x,y)


# In[184]:


y_pred


# # K fold cross validation to improve accuracy

# In[192]:


from sklearn.model_selection import cross_val_score


# In[200]:


clf=RandomForestClassifier(n_estimators=50)
clf.fit(x,y)

y_pred=clf.predict(test)

accuracies=cross_val_score(clf,x,y,cv=10)
accuracies.mean()


# In[202]:


svc=SVC(kernel='rbf')
svc.fit(x,y)

y_pred_1=svc.predict(test)
svc.score(x,y)

accuracies=cross_val_score(svc,x,y,cv=10)
accuracies.mean()


# In[203]:


log=LogisticRegression()
log.fit(x,y)
y_pred_2=log.predict(test)
log.score(x,y)

accuracies=cross_val_score(log,x,y,cv=10)
accuracies.mean()


# In[204]:


y_final=y_pred_1


# In[205]:


hell=pd.read_csv('test.csv')

submission=pd.DataFrame({"PassengerId":hell['PassengerId'],"Survived":y_pred})


# In[206]:


submission.to_csv('titanic.csv',index=False)
submission.head()

