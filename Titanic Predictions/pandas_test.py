import pandas as pd


####
'''https://www.youtube.com/watch?v=vmEHCJofslg'''

###3


# series



# DataFrames (basically Tables) - 2D data

df = pd.read_csv('data_src/train.csv')
# print(df)
# print(df.head(3))
# print(df.tail(3))


print(df.columns)
# print(df["Survived"][0]) # basically an array of array

# print(df["Name"][0:5])


#read each row
print(df.iloc[0:2])