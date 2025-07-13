import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression

data=pd.read_csv(r"C:\Users\NeelLaptop\Desktop\Python\Youtube1\smartwatches.csv")
# print(data.head(10))
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
data.dropna(inplace=True)  # Removing rows with NaN values
# print(data.isnull().sum())
data.drop_duplicates(inplace=True)  # Removing duplicate rows
# print(data.info())
data.drop("Unnamed: 0",axis=1,errors="ignore",inplace=True)# Removed Unnamed


data["Bluetooth"] = data["Bluetooth"].replace(["Yes", "No"], ["True", "False"])
data["Bluetooth"] = data["Bluetooth"].astype(bool)
# Converting Bluetooth column to Object to Bool



# data["Touchscreen"]=data["Touchscreen"].replace("Yes","")# Converting Touchscreen column to Object to Bool
# data["Touchscreen"]=data["Touchscreen"].astype(bool)

# data["Display Size"] = (data["Display Size"].astype(str).str.replace(" inches", "", regex=False).str.strip())# astype converts to string, strip removes extra spaces 
# data["Display Size"]=pd.to_numeric(data["Display Size"])# converting to float
# print(data.info())



# # Now we have a clean dataset, with no duplicates & no NaN values

# #numeric = [feature for feature in data.columns if data["feature"].dtype != "object"]
# # char=[feature for feature in data.columns if data['feature'].dtype == "object"]
# #print(numeric)

# # row=data["Brand"].value_counts()
# # balel=row.index
# # plt.pie(row,autopct="%.1f%%",labels=balel)
# # plt.title("List of all Companies")
# # plt.savefig("Company List.png")
# # plt.show()

# # sb.barplot(x="Brand",y="Rating",data=data,ci=None,errorbar=None)
# # plt.title("Brand vs Rating")
# # plt.xlabel("Brand")
# # plt.ylabel("Rating")
# # plt.savefig("Brand vs Rating.png")
# # plt.show()

# # sb.lineplot(y="Current Price",x="Original Price",color="skyblue",marker="o",data=data)
# # plt.title("Current Price vs Original Price")
# # plt.xlabel("Current Price")
# # plt.ylabel("Original Price")
# # plt.savefig("Current Price vs Original Price.png")
# # plt.show()


# # row=data["Strap Material"].value_counts()
# # balel=row.index
# # plt.pie(row,autopct="%.1f%%",labels=balel)
# # plt.title("List of all Types of Straps")
# # plt.savefig("Strap-List.png")
# # plt.show()


def convert(x):
    if x=="True":
        return 1
    else:
        return 0
    
# print(data.info())
#Using Linear Regression &  LOgistic Regression
numeric = [feature for feature in data.columns if data[feature].dtype != "object"]  
f=["Rating", "Battery Life (Days)","Display Size"]
data["Bluetooth"]=data["Bluetooth"].apply(convert)
X= data[f].values  # Features
Y= data["Current Price"]  # Target variable

X_train, X_test, Y_train, Y_test,= train_test_split(X, Y,test_size=0.2, random_state=42)#Testing 20% data
urating=float(input("Enter the Rating to predict the Current Price: "))
ubattery=float(input("Enter the Batterylife in days: "))
udisplay=float(input("Enter the Display Size "))
model = LinearRegression()
model.fit(X_train, Y_train)# Training 80% data
ans=model.predict([[urating,ubattery,udisplay]])
print(f"Rating for:- {urating} & Battery for:- {ubattery} & Display Size is{udisplay} The Price is : {ans[0]:.2f}")

