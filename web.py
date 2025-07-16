import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
from sklearn.cluster import KMeans


with open("modelS.pkl", "rb") as f:
    model = pkl.load(f)

st.sidebar.title('SIDE BAR')
option = st.sidebar.radio('Navigation', ['HOME', 'CURRENCY CONVERTER', 'SUPERVISED', 'UNSUPERVISED'])


def home():
    st.title('HOME')
    st.write("LET'S DRIVE INTO MACHINE LEARNING!")

    st.subheader('DEFINITION OF MACHINE LEARNING')
    st.write("*Machine Learning* is a *subset* of *Artificial Intelligence (AI)*.")
    st.write("- It creates a *predictive model*.")
    st.write("- It *trains* a model on a dataset for *predictive analysis*.")

    st.subheader('TYPES OF MACHINE LEARNING')
    t = st.selectbox('SELECT ANY TYPE OF MACHINE LEARNING', options=['TYPE', 'Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning'])

    if t == 'Supervised Learning':
        st.markdown('SUPERVISED LEARNING')

        st.subheader('DEFINITION')
        st.write("- Supervised Learning is a type of Machine Learning where the model learns from *labeled data*.")

        st.subheader('TECHNOLOGY')
        st.write('- Classification (0 or 1)')
        st.write('- Regression (Continuous Values)')

        st.subheader('ALGORITHMS')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Algorithm*")
            st.write("- Linear Regression")
            st.write("- Logistic Regression")
            st.write("- Decision Tree")
            st.write("- Random Forest")

        with col2:
            st.markdown("*Type*")
            st.write("--> Regression")
            st.write("--> Classification")
            st.write("--> Both")
            st.write("--> Both")

    elif t == 'Unsupervised Learning':
        st.markdown('UNSUPERVISED LEARNING')

        st.subheader('DEFINITION')
        st.write("- Unsupervised Learning is a type of Machine Learning where the model learns from *unlabeled data*.")
        st.write("- After clustering we get unlabelled data (y feature/output) then it goes in supervised.")

        st.subheader('TECHNOLOGY')
        st.write('- Clustering')

        st.subheader('ALGORITHMS')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Algorithm*")
            st.write("- K-Means Clustering")
            st.write("- DBSCAN")

        with col2:
            st.markdown("*Type*")
            st.write("--> Clustering")
            st.write("--> Clustering")

    elif t == 'Reinforcement Learning':
        st.markdown('REINFORCEMENT LEARNING')

        st.subheader('DEFINITION')
        st.write("- Reinforcement Learning is a type of Machine Learning where the model *learns through feedback*.")
        st.write("- It does not require any data.")

        st.subheader('TECHNOLOGY')
        st.write("- Model-Based RL")
        st.write("- Model-Free RL")

        st.subheader('ALGORITHMS')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Algorithm*")
            st.write("- Q-Learning")
            st.write("- SARSA (State-Action-Reward-State-Action)")

        with col2:
            st.markdown("*Type*")
            st.write("--> Model-Free")
            st.write("--> Model-Free")

    st.subheader("SELECT AN ALGORITHM")
    alg = st.selectbox("Choose an algorithm", options=['CHOOSE', "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors (KNN)", "K-Means Clustering"])

    if alg == "Linear Regression":
        st.markdown('LINEAR REGRESSION')
        st.subheader('DEFINITION')
        st.write("- Linear Regression is the data which is *continuous*.")
        st.write("- It creates a *Best Fit Line*.")
        st.subheader('CODE')
        st.code('''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
ypred=lr.predict(x)
print(ypred)
        ''', language='python')
        st.subheader('GRAPH')
        st.image("linear.png")

    elif alg == "Logistic Regression":
        st.markdown('LOGISTIC REGRESSION')
        st.subheader('DEFINITION')
        st.write("Logistic Regression is a supervised learning algorithm used for *classification problems*.")
        st.write("It uses the sigmoid function.")
        st.subheader("CODE")
        st.code('''
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(x, y)
ypred=lor.predict(x)
print(ypred)
        ''', language='python')
        st.subheader("GRAPH")
        st.image("logistic.png")

    elif alg == "Decision Tree":
        st.markdown("DECISION TREE")
        st.subheader('DEFINITION')
        st.write("Used for both classification and regression.")
        st.subheader("CODE")
        st.code('''
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x, y)
ypred=dt.predict(x)
print(ypred)
        ''', language='python')
        st.subheader("GRAPH")
        st.image("decision.png")

    elif alg == "Random Forest":
        st.markdown("RANDOM FOREST")
        st.subheader('DEFINITION')
        st.write("It is an ensemble method that has multiple decision trees.")
        st.subheader("CODE")
        st.code('''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x, y)
ypred=rf.predict(x)
print(ypred)
        ''', language='python')
        # st.subheader("GRAPH")
        # st.image("random.png")

    elif alg == "K-Nearest Neighbors (KNN)":
        st.markdown('K-NEAREST NEIGHBORS (KNN)')
        st.subheader('DEFINITION')
        st.write("- KNN can be used for both classification and clustering.")
        st.subheader("CODE")
        st.code('''
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(x,y)
ypred=kn.predict(x)
print(ypred)
        ''', language='python')
        # st.subheader("GRAPH")
        # st.image("knn.png")

    elif alg == "K-Means Clustering":
        st.markdown('K-MEANS CLUSTERING')
        st.write("- K-Means is an unsupervised clustering algorithm where K is the number of clusters.")
        st.subheader("CODE")
        st.code('''
from sklearn.cluster import KMeans
inertia = []
for k in range(1, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df)
    inertia.append(model.inertia_)
print(inertia)
        ''', language='python')
        # st.subheader("GRAPH")
        # st.image("kmeans.png")


def currency_converter():
    st.title('CURRENCY CONVERTER')
    st.write('THIS IS THE CURRENCY CONVERTER PAGE.')

    from_ = st.selectbox("From Currency", options=['India ₹', 'United States $', 'euro €', 'Japanese Yen ¥', 'Russia ₽'])
    amount = st.number_input("Enter the amount: ")
    to = st.selectbox("To Currency", options=['India ₹', 'United States $', 'euro €', 'Japanese Yen ¥', 'Russia ₽'])
    btn = st.button("CONVERT")
    st.balloons()

    convert = {
        'India ₹': 1.00,
        'United States $': 85.99,
        'euro €': 100.34,
        'Japanese Yen ¥': 0.58,
        'Russia ₽': 1.10
    }

    if btn:
        try:
            if amount == 0.0:
                raise ValueError("Amount cannot be zero.")
            if from_ == to:
                st.success(f"{amount}  {from_} = {amount}  {to} (Same Currency)")
            else:
                c = amount * convert[from_]
                converted = c / convert[to]
                rate = convert[from_] / convert[to]
                reverse_rate = 1 / rate
                st.success(f"{amount} {from_} = {round(converted, 2)} {to}")
                st.info(f"1 {from_} = {round(rate, 4)} {to}")
                st.info(f"1 {to} = {round(reverse_rate, 4)} {from_}")
        except ValueError as e:
            st.error(f"{str(e)}")


def supervised():
    st.title('SUPERVISED LEARNING')
    st.subheader('USING DATASET: Salary_dataset.csv')
    df = pd.read_csv("Salary_dataset.csv")

    st.subheader("SELECT")
    analysis = st.selectbox("Choose an analysis option:", ['All', 'Show describe()', 'Show columns', 'Show info'])

    if analysis == 'Show describe()':
        st.subheader("DESCRIBE SALARY CSV")
        st.write(df.describe())
    elif analysis == 'Show columns':
        st.subheader("COLUMNS OF SALARY CSV")
        st.write(df.columns)
    elif analysis == 'Show info':
        st.subheader("INFO OF SALARY CSV")
        st.write(df.info())
    elif analysis == 'All':
        st.subheader("DESCRIBE SALARY CSV")
        st.write(df.describe())
        st.subheader("COLUMNS OF SALARY CSV")
        st.write(df.columns)
        st.subheader("INFO OF SALARY CSV")
        st.write(df.info())

    
    x = df[['YearsExperience']]
    y = df['Salary']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)

    knn = KNeighborsRegressor()
    lr = LinearRegression()
    dt = DecisionTreeRegressor()
    rf = RandomForestRegressor()

    knn.fit(xtrain, ytrain)
    lr.fit(xtrain, ytrain)
    dt.fit(xtrain, ytrain)
    rf.fit(xtrain, ytrain)

    ypred_knn = knn.predict(xtest)
    ypred_lr = lr.predict(xtest)
    ypred_dt = dt.predict(xtest)
    ypred_rf = rf.predict(xtest)

    mse_knn = mean_squared_error(ytest, ypred_knn)
    mse_lr = mean_squared_error(ytest, ypred_lr)
    mse_dt = mean_squared_error(ytest, ypred_dt)
    mse_rf = mean_squared_error(ytest, ypred_rf)

    models = ["KNN", "Linear Regression", "Decision Tree", "Random Forest"]
    mse_values = [mse_knn, mse_lr, mse_dt, mse_rf]
    st.subheader("MSE")
    fig, ax = plt.subplots()
    ax.bar(models, mse_values)
    ax.set_xlabel("Models")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Model-wise MSE Comparison")
    st.pyplot(fig)

    st.subheader("PREDICT SALARY")
    ex = st.number_input("Enter your experience:", min_value=0)
    prSa = st.button("PREDICT")
    if prSa:
        saPr = model.predict([[ex]])
        st.success(f"Predicted Salary: ₹{saPr[0]:,.2f}")


def unsupervised():
    st.title('UNSUPERVISED LEARNING')
    st.subheader('USING DATASET: Mall_Customers.csv')
    df = pd.read_csv("Mall_Customers.csv")

    st.subheader("SELECT")
    analysis = st.selectbox("Choose an analysis option:", ['All', 'Show describe()', 'Show columns', 'Show info'])

    if analysis == 'Show describe()':
        st.subheader("DESCRIBE MALL CSV")
        st.write(df.describe())

    elif analysis == 'Show columns':
        st.subheader("COLUMNS OF MALL CSV")
        st.write(df.columns)

    elif analysis == 'Show info':
        st.subheader("INFO OF MALL CSV")
        st.write(df.info())

    elif analysis == 'All':
        st.subheader("DESCRIBE MALL CSV")
        st.write(df.describe())
        st.subheader("COLUMNS OF MALL CSV")
        st.write(df.columns)
        st.subheader("INFO OF MALL CSV")
        st.write(df.info())

    df['Gender'].unique()
    ec=LabelEncoder()
    ec.fit(df['Gender'])
    df['Gender']=ec.transform(df['Gender'])
    st.subheader("KMeans Clustering")
    inertia=[]
    for k in range(1, 10):
        model=KMeans(n_clusters=k, random_state=42)
        model.fit(df)
        inertia.append(model.inertia_)
    print(inertia)
    fig, ax = plt.subplots()
    ax.plot(inertia, marker='o', color='orange')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    df=pd.read_csv("Mall_Customers.csv")
    df['Gender'].unique()
    ec=LabelEncoder()
    ec.fit(df['Gender'])
    df['Gender']=ec.transform(df['Gender'])
    x=df.drop('CustomerID', axis=1)
    k=2
    kmeans=KMeans(n_clusters=k, random_state=42)
    y=kmeans.fit_predict(x)
    df["Cluster"]=y
    
    try:
        with open("modelM.pkl", "rb") as f:
            model1 = pkl.load(f)

        gender=st.selectbox("GENDER",['Male', 'Female'])
        age=st.number_input("AGE",min_value=1)
        income=st.number_input("ANNUAL INCOME",min_value=0)
        score=st.number_input("SPENDING SCORE",min_value=0)

        if st.button("PREDICT"):
           
            if age == 0 or income == 0 or score == 0:
                st.error("Please enter all fields.")
            else:
                g = 1 if gender == 'Male' else 0
                inputdata = [[g, age, income, score]]
                prediction = model1.predict(inputdata)
                if prediction[0]==0:
                    st.warning("Cluster 0: Less Frequent Vistor")
                else:
                    st.success("Cluster 1: More Frequent Visitor")
    except FileNotFoundError:
        st.error("Model not found")

    

    
    

if option == "HOME":
    home()
elif option == "CURRENCY CONVERTER":
    currency_converter()
elif option == "SUPERVISED":
    supervised()
elif option == "UNSUPERVISED":
    unsupervised()