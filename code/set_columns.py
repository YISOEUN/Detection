import pandas as pd

#기준이 되는 Data set
pre = 'Dataset_Misuse_AttributeSelection.csv'
#column 정리가 필요한 data file
test = 'all_file_separate_test.csv'
#다시 저장할 file 이름
restore = 'all_file_separate_test(1).csv'

#read csv files
a_df = pd.read_csv(pre)
b_df = pd.read_csv(test)
#b_df = b_df.drop('Unnamed: 0',axis = 1)

a_col = a_df.columns
b_col = b_df.columns

#column 정리
#같은 column이지만 다른 이름으로 작성 된 것은 개별적으로 변경
for i in range(len(b_col)):
    if not b_col[i] in a_col and not b_col[i] == 'srv_error_rate':
         b_df = b_df.drop(b_col[i],axis = 1)
         print(i,'.',a_col[i],"  ", b_col[i])
    elif b_col[i] == 'srv_error_rate':
        b_df.rename(columns = {b_col[i]: a_col[i]}, inplace = True)

b_col = b_df.columns
a_col = a_df.columns
## check
#for i in a_col :
#    if not i in b_col :
#        print(i)

#chcek
print(a_df.columns)
print()
print(b_df.columns)

#check
for i in range(len(a_col)) :
    if a_col[i] == b_col[i] :
        print(i,'.',a_col[i],"  ", b_col[i])
    else :
         print(i,'.',a_col[i],"  ", b_col[i])
         break

#모든 column이 있으면 다시 저장
if len(b_col) == len(a_col):
    b_df.to_csv(restore)
    print('store complete -',restore)
else :
    print(len(b_col))


