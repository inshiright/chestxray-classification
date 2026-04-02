from sklearn.model_selection import train_test_split

def split_data(df):
    patients = df["Patient ID"].unique()

    train_p, temp = train_test_split(
        patients,
        test_size=0.3,
        random_state=42
    )

    val_p, test_p = train_test_split(
        temp,
        test_size=0.5
    )

    train_df = df[df["Patient ID"].isin(train_p)]
    val_df = df[df["Patient ID"].isin(val_p)]
    test_df = df[df["Patient ID"].isin(test_p)]

    return train_df,val_df,test_df