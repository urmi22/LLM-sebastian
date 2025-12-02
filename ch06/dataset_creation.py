import pdb
import pandas as pd

def random_split(df, train_frac, validation_frac):

    # Shuffles the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculates split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Splits the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df



def create_balanced_dataset(df):

    # Counts the instances of “spam”
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly samples “ham” instances to match the number of “spam” instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combines ham subset with “spam”
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df



def main():
    data_file_path = "./ch06/data/sms_spam_collection/SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # print(df)

    # class label distribution, label “ham” (i.e., not spam)
    print(df["Label"].value_counts())
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    # we convert the “string” class labels "ham" and "spam" into integer class labels 0 and 1, respectively:
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    '''
    Next, we create a random_split function to split the dataset into three parts: 
    70% for training, 10% for validation, and 20% for testing.

    '''
    # Test size is implied to be 0.2 as the remainder.
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # save the dataset as CSV (comma-separated value) files so we can reuse it later
    train_df.to_csv("./ch06/input_dataset/train.csv", index=None)
    validation_df.to_csv("./ch06/input_dataset/validation.csv", index=None)
    test_df.to_csv("./ch06/input_dataset/test.csv", index=None)

    
    



if __name__=="__main__":
    main()