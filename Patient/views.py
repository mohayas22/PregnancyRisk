from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from .models import PlacementData
# Create your views here.
def index(request):
    return render(request,'index.html')

def login(request):
    if request.method == 'POST':
        #v = DoctorReg.objects.all()
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('dataentry')
        else:
            messages.info(request, 'Invalid credentials')
            return render(request,'login.html')
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password = request.POST['password']
        password2= request.POST['password2']
        email = request.POST['email']

        if password == password2:
            if User.objects.filter(username=username).exists():
                messages.info(request, 'Username is already exist')
                return render(request,'register.html')
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email is already exist')
                return render(request,'register.html')
            else:

                #save data in db
                user = User.objects.create_user(username=username, password=password, email=email,
                                                first_name=first_name, last_name=last_name)
                user.save();
                print('user created')
                return redirect('login')

        else:
            messages.info(request, 'Invalid Credentials')
            return render(request,'register.html')
        return redirect('/')
    else:
        return render(request, 'register.html')


def prediction(request):
    if (request.method == 'POST'):
        age= int(request.POST['age'])
        sp = int(request.POST['DiastolicBP'])
        dp = int(request.POST['DiastolicBP'])
        bs=int(request.POST['BS'])
        bt =int(request.POST['BodyTemp'])
        hr = int(request.POST['HeartRate'])
        
        import numpy as np # linear algebra
        import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

        # Input data files are available in the read-only "../input/" directory
        # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

        import os
        for dirname, _, filenames in os.walk(''):
            for filename in filenames:
                print(os.path.join(dirname, filename))
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import LabelEncoder #Encodes Categorical Data into Numerical
        from sklearn.model_selection import train_test_split #For spliting data into training and testing sets
        from sklearn.linear_model import LinearRegression #Linear Regression model
        from sklearn.ensemble import RandomForestRegressor#Random Forest Regression Model
        from sklearn.metrics import mean_absolute_error 
        from sklearn.linear_model import LogisticRegression #Lasso regression model
        df = pd.read_csv(r"static/Dataset/Maternal Health Risk Data Set.csv")
        print(df.head())
        df['RiskLevel'].replace({"high risk": "3", "mid risk": "2", "low risk" : "1"}, inplace=True)
        df['RiskLevel'] = df['RiskLevel'].astype(float)
        print(df.head())
        cols = [i for i in df.columns]
        cols = [i for i in cols if i not in ['RiskLevel']]
        plt.figure(figsize=(20,12))
        for i in enumerate(cols): 
            num = i[0]+1
            plt.subplot(3,5,num)
            sns.violinplot(data=df, x=i[1])
        plt.show()
        for column in df.columns[:6]:  # Loop over all columns except 'Location'
            sns.set()
            sns.set(style="ticks")
            sns.displot(df[column])
            plt.show()
        plt.figure(figsize=(15,12))
        sns.heatmap(df.corr(),annot=True,cmap='RdPu_r')
        plt.show()
        train, test = train_test_split(df, test_size=0.2, random_state=25)

        print(f"No. of training examples: {train.shape[0]}")
        print(f"No. of testing examples: {test.shape[0]}")
        y_test=test['RiskLevel']
        x_test=test.drop('RiskLevel', axis=1)
        x_test.head()
        #Setting training data into x_train and y_train
        x_train=train.drop('RiskLevel',axis=1)
        y_train=train['RiskLevel']

        #Shapes of x_train,y_train and test data
        print(x_train.shape, y_train.shape, x_test.shape)
        #Linear Regression Modeling and Training
        linear_model=LinearRegression()
        linear_model.fit(x_train,y_train)
        print(linear_model.score(x_train,y_train))

        #testing the model and Displaying the output
        linear_predict=linear_model.predict(x_test)
        linear_result=pd.DataFrame({'Id':test.index,'Predicted Risk':linear_predict,'Actual Risk':y_test})
        print("-------Linear Regression-----")
        print(linear_result)
        #Random Forest Regression
        random_model=RandomForestRegressor(n_estimators=50)
        random_model.fit(x_train,y_train)
        print(random_model.score(x_train,y_train))

        #Making predictions on test set 
        random_predict=random_model.predict(x_test)

        random_result=pd.DataFrame({'Id':test.index,'Predicted Risk':random_predict,'Actual Risk':y_test})
        print("-----Random Forest-----")
        print(random_result)
        #Random Forest Regression
        random_model=RandomForestRegressor(n_estimators=50)
        random_model.fit(x_train,y_train)
        print(random_model.score(x_train,y_train))

        #Making predictions on test set 
        random_predict=random_model.predict(x_test)

        random_result=pd.DataFrame({'Id':test.index,'Predicted Risk':random_predict,'Actual Risk':y_test})
        print(random_result)
        plt.figure(figsize=(18,7))
        ax=sns.lineplot(x=np.arange(0,len(random_predict)),y=random_predict,label = 'Predicted Random Forest Risk Value')
        ax = sns.lineplot(x=np.arange(0,len(y_test)),y=y_test,label = 'Actual Risk Value')
        ax.set_xlabel('ID',fontsize=12)
        ax.set_ylabel('Risk Value',fontsize=12)
        prop3 = dict(boxstyle='round',facecolor='orange',alpha=0.5)
        plt.legend(prop={'size':'15'})
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()

        plt.figure(figsize=(18,7))
        ax=sns.lineplot(x=np.arange(0,len(linear_predict)),y=linear_predict,label = 'Predicted Regression Risk Value', color='red')
        ax = sns.lineplot(x=np.arange(0,len(y_test)),y=y_test,label = 'Actual Risk Value')
        ax.set_xlabel('ID',fontsize=12)
        ax.set_ylabel('Risk Value',fontsize=12)
        prop3 = dict(boxstyle='round',facecolor='red',alpha=0.5)
        plt.legend(prop={'size':'15'})
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.show()
        from sklearn.ensemble import RandomForestClassifier
        r=RandomForestClassifier()
        X_train=df[["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate"]]
        Y_train=df[["RiskLevel"]]
        r.fit(X_train,Y_train)
        prediction=r.predict([[age,sp,dp,bs,bt,hr]])
        return render(request, 'prediction.html',
                      {'data':prediction,'age':age,'sp':sp,'dp':dp,'bs':bs,'bt':bt,'hr':hr})

    else:
        return render(request, 'prediction.html')

def dataentry(request):
    return render(request,"dataentry.html")


def logout(request):
    auth.logout(request)
    return redirect('/')